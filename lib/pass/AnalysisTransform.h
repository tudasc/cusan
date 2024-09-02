// cusan library
// Copyright (c) 2023-2024 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUSAN_ANALYSISTRANSFORM_H
#define CUSAN_ANALYSISTRANSFORM_H

#include "FunctionDecl.h"
#include "analysis/KernelAnalysis.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <utility>

using namespace llvm;
namespace cusan {

namespace analysis {

using KernelArgInfo = cusan::FunctionArg;

struct CudaKernelInvokeCollector {
  KernelModel& model;
  struct KernelInvokeData {
    llvm::SmallVector<KernelArgInfo, 4> args{};
    llvm::Value* void_arg_array{nullptr};
    llvm::Value* cu_stream{nullptr};
  };
  using Data = KernelInvokeData;

  CudaKernelInvokeCollector(KernelModel& current_stub_model) : model(current_stub_model) {
  }

  llvm::Optional<KernelInvokeData> match(llvm::CallBase& cb, Function& callee) const {
    if (callee.getName() == "cudaLaunchKernel") {
      errs() << "Func:" << callee.getFunction() << "\n";
      auto* cu_stream_handle      = std::prev(cb.arg_end())->get();
      auto* void_kernel_arg_array = std::prev(cb.arg_end(), 3)->get();
      // auto* cb_parent_function    = cb.getFunction();
      auto kernel_args = extract_kernel_args_for(void_kernel_arg_array);

      return KernelInvokeData{kernel_args, void_kernel_arg_array, cu_stream_handle};
    }
    return llvm::NoneType();
  }

  llvm::SmallVector<KernelArgInfo, 4> extract_kernel_args_for(llvm::Value* void_kernel_arg_array) const {
    unsigned index = 0;

    llvm::SmallVector<Value*, 4> real_args;

    for (auto* array_user : void_kernel_arg_array->users()) {
      if (auto* gep = dyn_cast<GetElementPtrInst>(array_user)) {
        for (auto* gep_user : gep->users()) {
          if (auto* store = dyn_cast<StoreInst>(gep_user)) {
            assert(index < model.args.size());
            if (auto* cast = dyn_cast<BitCastInst>(store->getValueOperand())) {
              real_args.push_back(*cast->operand_values().begin());
            } else {
              assert(false);
            }
            index++;
          }
        }
      }
    }

    llvm::SmallVector<KernelArgInfo, 4> result = model.args;
    for (auto& res : result) {
      Value* val = real_args[real_args.size() - 1 - res.arg_pos];
      // because of ABI? clang might convert struct argument to a (byval)pointer
      // but the actual cuda argument is by value so we double check if the expected type matches the actual type
      // and only if then we load it. I think this should handle all cases since the only case it would fail
      // is if we do strct* and send that (byval)pointer but that shouldn't be a thing?
      bool real_ptr =
          res.is_pointer &&
          (dyn_cast<PointerType>(dyn_cast<PointerType>(val->getType())->getPointerElementType()) != nullptr);

      // not fake pointer from clang so load it before getting subargs
      for (auto& sub_arg : res.subargs) {
        if (real_ptr) {
          sub_arg.indices.insert(sub_arg.indices.begin(), -1);
        }
        sub_arg.value = val;
      }
      res.value = val;
    }
    return result;
  }
};

}  // namespace analysis

namespace transform {

struct KernelInvokeTransformer {
  callback::FunctionDecl* decls_;

  KernelInvokeTransformer(callback::FunctionDecl* decls) : decls_(decls) {
  }

  bool transform(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb) const {
    using namespace llvm;
    return generate_compound_cb(data, irb);
  }

 private:
  static short access_cast(AccessState access, bool is_ptr) {
    auto value = static_cast<short>(access);
    value <<= 1;
    if (is_ptr) {
      value |= 1;
    }
    return value;
  }

  static llvm::Value* get_cu_stream_ptr(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb) {
    auto* cu_stream = data.cu_stream;
    assert(cu_stream != nullptr && "Require cuda stream!");
    auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(cu_stream, irb.getInt8PtrTy());
    return cu_stream_void_ptr;
  }

  bool generate_compound_cb(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb) const {
    const bool should_transform =
        llvm::count_if(data.args, [&](const auto& elem) {
          return llvm::count_if(elem.subargs, [&](const auto& sub_elem) { return sub_elem.is_pointer; }) > 0;
        }) > 0;

    uint32_t n_subargs = 0;
    for (const auto& arg : data.args) {
      n_subargs += arg.subargs.size();
    }

    if (!should_transform) {
      return false;
    }

    auto target_callback = decls_->cusan_register_access;

    auto* i16_ty      = Type::getInt16Ty(irb.getContext());
    auto* i32_ty      = Type::getInt32Ty(irb.getContext());
    auto* void_ptr_ty = Type::getInt8PtrTy(irb.getContext());
    // auto* void_ptr_ptr_ty = Type::getInt8PtrTy(irb.getContext())->getPointerTo();

    auto* cu_stream_void_ptr = get_cu_stream_ptr(data, irb);
    auto* arg_size           = irb.getInt32(n_subargs);
    auto* arg_access_array   = irb.CreateAlloca(i16_ty, arg_size);
    auto* arg_value_array    = irb.CreateAlloca(void_ptr_ty, arg_size);

    size_t arg_array_index = 0;
    for (const auto& arg : data.args) {
      errs() << "Handling Arg: " << arg << "\n";
      for (const auto& sub_arg : arg.subargs) {
        errs() << "   subarg: " << sub_arg << "\n";
        const auto access = access_cast(sub_arg.state, sub_arg.is_pointer);
        Value* idx        = ConstantInt::get(i32_ty, arg_array_index);
        Value* acc        = ConstantInt::get(i16_ty, access);
        auto* gep_acc     = irb.CreateGEP(i16_ty, arg_access_array, idx);
        irb.CreateStore(acc, gep_acc);
        // only if it is a pointer store the actual pointer in the value array
        if (sub_arg.is_pointer) {
          assert(arg.value.hasValue());

          auto* value_ptr = arg.value.getValue();

          // TODO: parts of a struct might be null if they are only executed conditionally so we should check the parent
          // for null before gep/load
          for (auto gep_index : sub_arg.indices) {
            auto* subtype = dyn_cast<PointerType>(value_ptr->getType())->getPointerElementType();
            if (gep_index == -1) {
              value_ptr = irb.CreateLoad(subtype, value_ptr);
            } else {
              value_ptr = irb.CreateStructGEP(subtype, value_ptr, gep_index);
            }
          }

          auto* voided_ptr    = irb.CreatePointerCast(value_ptr, void_ptr_ty);
          auto* gep_val_array = irb.CreateGEP(void_ptr_ty, arg_value_array, idx);
          irb.CreateStore(voided_ptr, gep_val_array);
          arg_array_index += 1;
        }
      }
    }

    Value* args_cusan_register[] = {arg_value_array, arg_access_array, arg_size, cu_stream_void_ptr};
    irb.CreateCall(target_callback.f, args_cusan_register);
    return true;
  }
};

template <class Collector, class Transformer>
class CallInstrumenter {
  Function& f_;
  Collector collector_;
  Transformer transformer_;
  struct InstrumentationData {
    typename Collector::Data user_data;
    CallBase* cb;
  };
  llvm::SmallVector<InstrumentationData, 4> data_vec_;

 public:
  CallInstrumenter(Collector c, Transformer t, Function& f) : f_(f), collector_(c), transformer_(t) {
  }

  bool instrument() {
    for (auto& I : instructions(f_)) {
      if (auto* cb = dyn_cast<CallBase>(&I)) {
        if (auto* f = cb->getCalledFunction()) {
          auto t = collector_.match(*cb, *f);
          if (t.hasValue()) {
            data_vec_.push_back({t.getValue(), cb});
          }
        }
      }
    }

    bool modified = false;
    if (data_vec_.size() > 0) {
      IRBuilder<> irb{data_vec_[0].cb};
      for (auto data : data_vec_) {
        irb.SetInsertPoint(data.cb);
        modified |= transformer_.transform(data.user_data, irb);
      }
    }
    return modified;
  }
};

template <typename T, typename = int>
struct WantsReturnValue : std::false_type {};

template <typename T>
struct WantsReturnValue<T, decltype(&T::map_return_value, 0)> : std::true_type {};

template <class T>
class SimpleInstrumenter {
  enum class InsertLocation {
    // insert before or after the call that were instrumenting
    kBefore,
    kAfter
  };

  const FunctionCallee* callee_;
  StringRef func_name_;
  SmallVector<llvm::CallBase*, 4> target_callsites_;

 public:
  void setup(StringRef name, FunctionCallee* callee) {
    func_name_ = name;
    callee_    = callee;
  }

  bool instrument(Function& func, InsertLocation loc = InsertLocation::kAfter) {
    for (auto& I : instructions(func)) {
      if (auto* cb = dyn_cast<CallBase>(&I)) {
        if (auto* f = cb->getCalledFunction()) {
          if (func_name_ == f->getName()) {
            target_callsites_.push_back(cb);
          }
        }
      }
    }

    if (!target_callsites_.empty()) {
      IRBuilder<> irb{target_callsites_[0]};
      for (CallBase* cb : target_callsites_) {
        if (loc == InsertLocation::kBefore) {
          irb.SetInsertPoint(cb);
        } else {
          if (auto* invoke = dyn_cast<InvokeInst>(cb)) {
            irb.SetInsertPoint(invoke->getNormalDest()->getFirstNonPHI());
          } else {
            irb.SetInsertPoint(cb->getNextNonDebugInstruction());
          }
        }

        SmallVector<Value*> v;
        for (auto& arg : cb->args()) {
          v.push_back(arg.get());
        }
        auto args = T::map_arguments(irb, v);
        if constexpr (WantsReturnValue<T>::value) {
          assert(loc == InsertLocation::kAfter && "Can only capture return value if insertion location is after");
          args.append(T::map_return_value(irb, cb));
        }
        irb.CreateCall(*callee_, args);
      }
    }
    return !target_callsites_.empty();
  }
};

class DeviceSyncInstrumenter : public SimpleInstrumenter<DeviceSyncInstrumenter> {
 public:
  DeviceSyncInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaDeviceSynchronize", &decls->cusan_sync_device.f);
  }
  static llvm::SmallVector<Value*, 4> map_arguments(IRBuilder<>&, llvm::ArrayRef<Value*>) {
    return {};
  }
};
class StreamSyncInstrumenter : public SimpleInstrumenter<StreamSyncInstrumenter> {
 public:
  StreamSyncInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaStreamSynchronize", &decls->cusan_sync_stream.f);
  }
  static llvm::SmallVector<Value*, 1> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 1);
    Value* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {cu_stream_void_ptr};
  }
};
class EventSyncInstrumenter : public SimpleInstrumenter<EventSyncInstrumenter> {
 public:
  EventSyncInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaEventSynchronize", &decls->cusan_sync_event.f);
  }
  static llvm::SmallVector<Value*, 1> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 1);
    auto* cu_event_void_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {cu_event_void_ptr};
  }
};
class EventRecordInstrumenter : public SimpleInstrumenter<EventRecordInstrumenter> {
 public:
  EventRecordInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaEventRecord", &decls->cusan_event_record.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 2);
    auto* cu_event_void_ptr  = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
    return {cu_event_void_ptr, cu_stream_void_ptr};
  }
};
class EventRecordFlagsInstrumenter : public SimpleInstrumenter<EventRecordFlagsInstrumenter> {
 public:
  EventRecordFlagsInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaEventRecordWithFlags", &decls->cusan_event_record.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 3);
    auto* cu_event_void_ptr  = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
    return {cu_event_void_ptr, cu_stream_void_ptr};
  }
};

class CudaMemcpyAsyncInstrumenter : public SimpleInstrumenter<CudaMemcpyAsyncInstrumenter> {
 public:
  CudaMemcpyAsyncInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaMemcpyAsync", &decls->cusan_memcpy_async.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    // void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0
    assert(args.size() == 5);
    auto* dst_ptr   = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* src_ptr   = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
    auto* count     = args[2];
    auto* kind      = args[3];
    auto* cu_stream = irb.CreateBitOrPointerCast(args[4], irb.getInt8PtrTy());
    return {dst_ptr, src_ptr, count, kind, cu_stream};
  }
};

class CudaMemcpyInstrumenter : public SimpleInstrumenter<CudaMemcpyInstrumenter> {
 public:
  CudaMemcpyInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaMemcpy", &decls->cusan_memcpy.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    // void* dst, const void* src, size_t count, cudaMemcpyKind kind
    assert(args.size() == 4);
    auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* src_ptr = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
    auto* count   = args[2];
    auto* kind    = args[3];
    return {dst_ptr, src_ptr, count, kind};
  }
};

class CudaMemcpy2DInstrumenter : public SimpleInstrumenter<CudaMemcpy2DInstrumenter> {
 public:
  CudaMemcpy2DInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaMemcpy2D", &decls->cusan_memcpy_2d.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    // void* target, size_t dpitch, const void* from, size_t spitch, size_t width, size_t height, cusan_MemcpyKind kind
    assert(args.size() == 7);
    auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* dpitch  = args[1];
    auto* src_ptr = irb.CreateBitOrPointerCast(args[2], irb.getInt8PtrTy());
    auto* spitch  = args[3];
    auto* width   = args[4];
    auto* height  = args[5];
    auto* kind    = args[6];
    return {dst_ptr, dpitch, src_ptr, spitch, width, height, kind};
  }
};

class CudaMemcpy2DAsyncInstrumenter : public SimpleInstrumenter<CudaMemcpy2DAsyncInstrumenter> {
 public:
  CudaMemcpy2DAsyncInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaMemcpy2DAsync", &decls->cusan_memcpy_2d_async.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    // void* target, size_t dpitch, const void* from, size_t spitch, size_t width, size_t height, cusan_MemcpyKind kind,
    // stream
    assert(args.size() == 8);
    auto* dst_ptr   = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* dpitch    = args[1];
    auto* src_ptr   = irb.CreateBitOrPointerCast(args[2], irb.getInt8PtrTy());
    auto* spitch    = args[3];
    auto* width     = args[4];
    auto* height    = args[5];
    auto* kind      = args[6];
    auto* cu_stream = irb.CreateBitOrPointerCast(args[7], irb.getInt8PtrTy());
    return {dst_ptr, dpitch, src_ptr, spitch, width, height, kind, cu_stream};
  }
};

class CudaMemsetAsyncInstrumenter : public SimpleInstrumenter<CudaMemsetAsyncInstrumenter> {
 public:
  CudaMemsetAsyncInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaMemsetAsync", &decls->cusan_memset_async.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* devPtr, int  value, size_t count, cudaStream_t stream = 0 )
    assert(args.size() == 4);
    auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    // auto* value     = args[1];
    auto* count     = args[2];
    auto* cu_stream = irb.CreateBitOrPointerCast(args[3], irb.getInt8PtrTy());
    return {dst_ptr, count, cu_stream};
  }
};
class CudaMemsetInstrumenter : public SimpleInstrumenter<CudaMemsetInstrumenter> {
 public:
  CudaMemsetInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaMemset", &decls->cusan_memset.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* devPtr, int  value, size_t count,)
    assert(args.size() == 3);
    auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    // auto* value   = args[1];
    auto* count = args[2];
    return {dst_ptr, count};
  }
};

class CudaMemset2dAsyncInstrumenter : public SimpleInstrumenter<CudaMemset2dAsyncInstrumenter> {
 public:
  CudaMemset2dAsyncInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaMemset2DAsync", &decls->cusan_memset_2d_async.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    // void* devPtr, size_t pitch, int  value, size_t width, size_t height, cudaStream_t stream = 0
    assert(args.size() == 6);
    auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* pitch   = args[1];
    // auto* value     = args[2];
    auto* height    = args[3];
    auto* width     = args[4];
    auto* cu_stream = irb.CreateBitOrPointerCast(args[5], irb.getInt8PtrTy());
    return {dst_ptr, pitch, height, width, cu_stream};
  }
};
class CudaMemset2dInstrumenter : public SimpleInstrumenter<CudaMemset2dInstrumenter> {
 public:
  CudaMemset2dInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaMemset2D", &decls->cusan_memset_2d.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    // void* devPtr, size_t pitch, int  value, size_t width, size_t height
    assert(args.size() == 5);
    auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* pitch   = args[1];
    // auto* value   = args[2];
    auto* height = args[3];
    auto* width  = args[4];
    ;
    return {dst_ptr, pitch, height, width};
  }
};

class CudaHostAlloc : public SimpleInstrumenter<CudaHostAlloc> {
 public:
  CudaHostAlloc(callback::FunctionDecl* decls) {
    setup("cudaHostAlloc", &decls->cusan_host_alloc.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void** ptr, size_t size, unsigned int flags )
    assert(args.size() == 3);
    auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* size    = args[1];
    auto* flags   = args[2];
    return {dst_ptr, size, flags};
  }
};

class CudaMallocHost : public SimpleInstrumenter<CudaMallocHost> {
 public:
  CudaMallocHost(callback::FunctionDecl* decls) {
    setup("cudaMallocHost", &decls->cusan_host_alloc.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void** ptr, size_t size)
    assert(args.size() == 2);
    auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* size    = args[1];
    auto* flags   = llvm::ConstantInt::get(Type::getInt32Ty(irb.getContext()), 0, false);
    return {dst_ptr, size, flags};
  }
};

class CudaEventCreateInstrumenter : public SimpleInstrumenter<CudaEventCreateInstrumenter> {
 public:
  CudaEventCreateInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaEventCreate", &decls->cusan_event_create.f);
  }
  static llvm::SmallVector<Value*, 1> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 1);
    // auto* cu_event_void_ptr = irb.CreateLoad(irb.getInt8PtrTy(), args[0], "");
    auto* cu_event_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {cu_event_void_ptr_ptr};
  }
};

class CudaEventCreateWithFlagsInstrumenter : public SimpleInstrumenter<CudaEventCreateWithFlagsInstrumenter> {
 public:
  CudaEventCreateWithFlagsInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaEventCreateWithFlags", &decls->cusan_event_create.f);
  }
  static llvm::SmallVector<Value*, 1> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //cudaEvent_t* event, unsigned int  flags
    assert(args.size() == 2);
    auto* cu_event_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {cu_event_void_ptr_ptr};
  }
};



class StreamCreateInstrumenter : public SimpleInstrumenter<StreamCreateInstrumenter> {
 public:
  StreamCreateInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaStreamCreate", &decls->cusan_stream_create.f);
  }
  static llvm::SmallVector<Value*, 1> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 1);
    auto* flags                  = llvm::ConstantInt::get(Type::getInt32Ty(irb.getContext()), 0, false);
    auto* cu_stream_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {cu_stream_void_ptr_ptr, flags};
  }
};

class StreamCreateWithFlagsInstrumenter : public SimpleInstrumenter<StreamCreateWithFlagsInstrumenter> {
 public:
  StreamCreateWithFlagsInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaStreamCreateWithFlags", &decls->cusan_stream_create.f);
  }
  static llvm::SmallVector<Value*, 1> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 2);
    auto* cu_stream_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* flags                  = args[1];
    return {cu_stream_void_ptr_ptr, flags};
  }
};

class StreamCreateWithPriorityInstrumenter : public SimpleInstrumenter<StreamCreateWithPriorityInstrumenter> {
 public:
  StreamCreateWithPriorityInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaStreamCreateWithPriority", &decls->cusan_stream_create.f);
  }
  static llvm::SmallVector<Value*, 1> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 3);
    auto* cu_stream_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* flags                  = args[1];
    return {cu_stream_void_ptr_ptr, flags};
  }
};


class StreamWaitEventInstrumenter : public SimpleInstrumenter<StreamWaitEventInstrumenter> {
 public:
  StreamWaitEventInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaStreamWaitEvent", &decls->cusan_stream_wait_event.f);
  }
  static llvm::SmallVector<Value*, 1> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 3);
    // auto* cu_stream_void_ptr = irb.CreateLoad(irb.getInt8PtrTy(), args[0], "");
    auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* cu_event_void_ptr  = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
    return {cu_stream_void_ptr, cu_event_void_ptr, args[2]};
  }
};

class CudaHostRegister : public SimpleInstrumenter<CudaHostRegister> {
 public:
  CudaHostRegister(callback::FunctionDecl* decls) {
    setup("cudaHostRegister", &decls->cusan_host_register.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* ptr)
    assert(args.size() == 3);
    auto* ptr   = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* size  = args[1];
    auto* flags = args[2];
    return {ptr, size, flags};
  }
};

class CudaHostUnregister : public SimpleInstrumenter<CudaHostUnregister> {
 public:
  CudaHostUnregister(callback::FunctionDecl* decls) {
    setup("cudaHostUnregister", &decls->cusan_host_unregister.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* ptr)
    assert(args.size() == 1);
    auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {ptr};
  }
};

class CudaHostFree : public SimpleInstrumenter<CudaHostFree> {
 public:
  CudaHostFree(callback::FunctionDecl* decls) {
    setup("cudaFreeHost", &decls->cusan_host_free.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* ptr)
    assert(args.size() == 1);
    auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {ptr};
  }
};

class CudaMallocManaged : public SimpleInstrumenter<CudaMallocManaged> {
 public:
  CudaMallocManaged(callback::FunctionDecl* decls) {
    setup("cudaMallocManaged", &decls->cusan_managed_alloc.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* ptr, size_t size, u32 flags)
    assert(args.size() == 3);
    auto* ptr   = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* size  = args[1];
    auto* flags = args[2];
    return {ptr, size, flags};
  }
};

class CudaMalloc : public SimpleInstrumenter<CudaMalloc> {
 public:
  CudaMalloc(callback::FunctionDecl* decls) {
    setup("cudaMalloc", &decls->cusan_device_alloc.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* ptr, size_t size)
    assert(args.size() == 2);
    auto* ptr  = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* size = args[1];
    return {ptr, size};
  }
};

class CudaFree : public SimpleInstrumenter<CudaFree> {
 public:
  CudaFree(callback::FunctionDecl* decls) {
    setup("cudaFree", &decls->cusan_device_free.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* ptr)
    assert(args.size() == 1);
    auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {ptr};
  }
};

class CudaMallocPitch : public SimpleInstrumenter<CudaMallocPitch> {
 public:
  CudaMallocPitch(callback::FunctionDecl* decls) {
    setup("cudaMallocPitch", &decls->cusan_device_alloc.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //(void** devPtr, size_t* pitch, size_t width, size_t height )
    assert(args.size() == 4);
    auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());

    //"The function may pad the allocation"
    //"*pitch by cudaMallocPitch() is the width in bytes of the allocation"
    auto* pitch = irb.CreateLoad(irb.getIntPtrTy(irb.GetInsertBlock()->getModule()->getDataLayout()), args[1]);
    // auto* width = args[2];
    auto* height = args[3];

    auto* real_size = irb.CreateMul(pitch, height);
    return {ptr, real_size};
  }
};

class CudaStreamQuery : public SimpleInstrumenter<CudaStreamQuery> {
 public:
  CudaStreamQuery(callback::FunctionDecl* decls) {
    setup("cudaStreamQuery", &decls->cusan_stream_query.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* stream)
    assert(args.size() == 1);
    auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {ptr};
  }
  static llvm::SmallVector<Value*, 1> map_return_value(IRBuilder<>& irb, Value* result) {
    (void)irb;
    return {result};
  }
};

class CudaEventQuery : public SimpleInstrumenter<CudaEventQuery> {
 public:
  CudaEventQuery(callback::FunctionDecl* decls) {
    setup("cudaEventQuery", &decls->cusan_event_query.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* event)
    assert(args.size() == 1);
    auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {ptr};
  }
  static llvm::SmallVector<Value*, 1> map_return_value(IRBuilder<>& irb, Value* result) {
    (void)irb;
    return {result};
  }
};

}  // namespace transform
}  // namespace cusan

#endif
