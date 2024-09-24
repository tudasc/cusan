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

inline auto get_void_ptr_type(IRBuilder<>& irb) {
#if LLVM_VERSION_MAJOR >= 15
  return irb.getPtrTy();
#else
  return irb.getInt8PtrTy();
#endif
}

namespace analysis {

using KernelArgInfo = cusan::FunctionArg;

struct CudaKernelInvokeCollector {
  KernelModel& model;
  struct KernelInvokeData {
    llvm::SmallVector<KernelArgInfo, 4> args;
    llvm::Value* void_arg_array{nullptr};
    llvm::Value* cu_stream{nullptr};
  };
  using Data = KernelInvokeData;

  CudaKernelInvokeCollector(KernelModel& current_stub_model) : model(current_stub_model) {
  }

  std::optional<KernelInvokeData> match(llvm::CallBase& cb, Function& callee) const {
    if (callee.getName() == "cudaLaunchKernel") {
      errs() << "Func:" << callee.getFunction() << "\n";
      auto* cu_stream_handle      = std::prev(cb.arg_end())->get();
      auto* void_kernel_arg_array = std::prev(cb.arg_end(), 3)->get();
      // auto* cb_parent_function    = cb.getFunction();
      auto kernel_args = extract_kernel_args_for(void_kernel_arg_array);

      return KernelInvokeData{kernel_args, void_kernel_arg_array, cu_stream_handle};
    }
    return std::nullopt;
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
      // but the actual cuda argument is just a value. So we doulbe check that it actually allocates a pointer
      bool real_ptr =
          res.is_pointer &&
          dyn_cast<AllocaInst>(val)->getAllocatedType()->isPointerTy();

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
    auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(cu_stream, get_void_ptr_type(irb));
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
    auto* void_ptr_ty = get_void_ptr_type(irb);


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
          assert(arg.value.has_value());

          auto* value_ptr = arg.value.value();

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
          if (t.has_value()) {
            data_vec_.push_back({t.value(), cb});
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

#ifndef BasicInstrumenterDecl
#define BasicInstrumenterDecl(name)                                                       \
  class name : public SimpleInstrumenter<name> {                                          \
   public:                                                                                \
    name(callback::FunctionDecl* decls);                                                  \
    static llvm::SmallVector<Value*> map_arguments(IRBuilder<>&, llvm::ArrayRef<Value*>); \
  };
#endif

BasicInstrumenterDecl(DeviceSyncInstrumenter);
BasicInstrumenterDecl(StreamSyncInstrumenter);
BasicInstrumenterDecl(EventSyncInstrumenter);
BasicInstrumenterDecl(EventRecordInstrumenter);
BasicInstrumenterDecl(EventRecordFlagsInstrumenter);
BasicInstrumenterDecl(CudaMemcpyAsyncInstrumenter);
BasicInstrumenterDecl(CudaMemcpyInstrumenter);
BasicInstrumenterDecl(CudaMemcpy2DInstrumenter);
BasicInstrumenterDecl(CudaMemcpy2DAsyncInstrumenter);
BasicInstrumenterDecl(CudaMemsetAsyncInstrumenter);
BasicInstrumenterDecl(CudaMemsetInstrumenter);
BasicInstrumenterDecl(CudaMemset2dAsyncInstrumenter);
BasicInstrumenterDecl(CudaMemset2dInstrumenter);
BasicInstrumenterDecl(CudaHostAlloc);
BasicInstrumenterDecl(CudaMallocHost);
BasicInstrumenterDecl(CudaEventCreateInstrumenter);
BasicInstrumenterDecl(CudaEventCreateWithFlagsInstrumenter);
BasicInstrumenterDecl(StreamCreateInstrumenter);
BasicInstrumenterDecl(StreamCreateWithFlagsInstrumenter);
BasicInstrumenterDecl(StreamCreateWithPriorityInstrumenter);
BasicInstrumenterDecl(StreamWaitEventInstrumenter);
BasicInstrumenterDecl(CudaHostRegister);
BasicInstrumenterDecl(CudaHostUnregister);
BasicInstrumenterDecl(CudaHostFree);
BasicInstrumenterDecl(CudaMallocManaged);
BasicInstrumenterDecl(CudaMalloc);
BasicInstrumenterDecl(CudaFree);
BasicInstrumenterDecl(CudaMallocPitch);

class CudaStreamQuery : public SimpleInstrumenter<CudaStreamQuery> {
 public:
  CudaStreamQuery(callback::FunctionDecl* decls);
  static llvm::SmallVector<Value*> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args);
  static llvm::SmallVector<Value*, 1> map_return_value(IRBuilder<>& irb, Value* result);
};

class CudaEventQuery : public SimpleInstrumenter<CudaEventQuery> {
 public:
  CudaEventQuery(callback::FunctionDecl* decls);
  static llvm::SmallVector<Value*> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args);
  static llvm::SmallVector<Value*, 1> map_return_value(IRBuilder<>& irb, Value* result);
};

}  // namespace transform
}  // namespace cusan

#endif
