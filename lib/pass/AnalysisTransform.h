// cusan library
// Copyright (c) 2023-2024 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUSAN_ANALYSISTRANSFORM_H
#define CUSAN_ANALYSISTRANSFORM_H

#include "FunctionDecl.h"
#include "analysis/KernelAnalysis.h"

#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>

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

namespace helper {
template <typename... Strings>
bool ends_with_any_of(const std::string& name, Strings&&... searching_names) {
  const llvm::StringRef name_ref{name};
#if LLVM_VERSION_MAJOR > 15
  return (name_ref.ends_with(searching_names) || ...);
#else
  return (name_ref.endswith(searching_names) || ...);
#endif
}

}  // namespace helper

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

  std::optional<KernelInvokeData> match(llvm::CallBase& cb, Function& callee) const;

  llvm::SmallVector<KernelArgInfo, 4> extract_kernel_args_for(llvm::Value* void_kernel_arg_array) const;
};

}  // namespace analysis

namespace transform {

struct KernelInvokeTransformer {
  callback::FunctionDecl* decls_;

  KernelInvokeTransformer(callback::FunctionDecl* decls) : decls_(decls) {
  }

  bool transform(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb) const;

 private:
  static short access_cast(AccessState access, bool is_ptr);

  static llvm::Value* get_cu_stream_ptr(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb);
  bool generate_compound_cb(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb) const;
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

class CudaMallocPitch : public SimpleInstrumenter<CudaMallocPitch> {
 public:
  CudaMallocPitch(callback::FunctionDecl* decls) {
    setup("cudaMallocPitch", &decls->cusan_device_alloc.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //(void** devPtr, size_t* pitch, size_t width, size_t height )
    assert(args.size() == 4);
    auto* ptr  = irb.CreateBitOrPointerCast(args[0], irb.getInt8Ty()->getPointerTo());

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
