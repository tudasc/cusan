// cusan library
// Copyright (c) 2023-2024 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUSAN_ANALYSISTRANSFORM_H
#define CUSAN_ANALYSISTRANSFORM_H

#include "../analysis/KernelAnalysis.h"
#include "FunctionDecl.h"
#include "support/Logger.h"
#include "support/Util.h"

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

inline bool does_name_match(const std::string& model_kernel_name, llvm::CallBase& cb) {
  assert(cb.getFunction() != nullptr && "Callbase requires function.");
  const auto stub_name      = util::try_demangle_fully(*cb.getFunction());
  const auto searching_name = util::try_demangle_fully(model_kernel_name);

  StringRef searching_without_type{searching_name};
  if (StringRef{stub_name}.contains("lambda")) {
    LOG_DEBUG("Detected lambda function in stub name " << stub_name);
    // if we got a lambda it has a return type included that we want to shave off
    const auto first_space = searching_name.find(' ');
    searching_without_type = llvm::StringRef(searching_name).substr(first_space + 1);
  }

  LOG_DEBUG("Check stub \"" << stub_name << "\" ends with \"" << searching_name << "\" or \"" << searching_without_type
                            << "\"");
  return util::ends_with_any_of(stub_name, searching_name, searching_without_type);
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

struct HipKernelInvokeCollector {
  KernelModel& model;
  struct KernelInvokeData {
    llvm::SmallVector<KernelArgInfo, 4> args;
    llvm::Value* void_arg_array{nullptr};
    llvm::Value* hip_stream{nullptr};
  };
  using Data = KernelInvokeData;

  HipKernelInvokeCollector(KernelModel& current_stub_model) : model(current_stub_model) {
  }

  std::optional<KernelInvokeData> match(llvm::CallBase& cb, Function& callee) const;

  llvm::SmallVector<KernelArgInfo, 4> extract_kernel_args_for(llvm::Value* void_kernel_arg_array) const;
};

}  // namespace analysis

namespace transform {

struct CudaKernelInvokeTransformer {
  callback::FunctionDecl* decls_;

  CudaKernelInvokeTransformer(callback::FunctionDecl* decls) : decls_(decls) {
  }

  bool transform(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb) const;

 private:
  static short access_cast(AccessState access, bool is_ptr);

  static llvm::Value* get_cu_stream_ptr(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb);
  bool generate_compound_cb(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb) const;
};

struct HipKernelInvokeTransformer {
  callback::FunctionDecl* decls_;

  HipKernelInvokeTransformer(callback::FunctionDecl* decls) : decls_(decls) {
  }

  bool transform(const analysis::HipKernelInvokeCollector::Data& data, IRBuilder<>& irb) const;

 private:
  static short access_cast(AccessState access, bool is_ptr);

  static llvm::Value* get_hip_stream_ptr(const analysis::HipKernelInvokeCollector::Data& data, IRBuilder<>& irb);
  bool generate_compound_cb(const analysis::HipKernelInvokeCollector::Data& data, IRBuilder<>& irb) const;
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
    auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8Ty()->getPointerTo());

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

BasicInstrumenterDecl(HipDeviceSyncInstrumenter);
BasicInstrumenterDecl(HipMemcpyAsyncInstrumenter);
BasicInstrumenterDecl(HipMemcpyInstrumenter);
BasicInstrumenterDecl(HipMalloc);
BasicInstrumenterDecl(HipMallocManaged);
BasicInstrumenterDecl(HipMemsetInstrumenter);
BasicInstrumenterDecl(HipMemcpy2DInstrumenter);
BasicInstrumenterDecl(HipMemcpy2DAsyncInstrumenter);
BasicInstrumenterDecl(HipStreamSyncInstrumenter);
BasicInstrumenterDecl(HipFree);
BasicInstrumenterDecl(HipStreamCreateInstrumenter);
BasicInstrumenterDecl(HipStreamCreateWithFlagsInstrumenter);
BasicInstrumenterDecl(HipStreamCreateWithPriorityInstrumenter);

BasicInstrumenterDecl(HipEventCreateInstrumenter);
BasicInstrumenterDecl(HipEventCreateWithFlagsInstrumenter);
BasicInstrumenterDecl(HipEventRecordInstrumenter);
BasicInstrumenterDecl(HipEventSyncInstrumenter);

BasicInstrumenterDecl(HipMemsetAsyncInstrumenter);
BasicInstrumenterDecl(HipMemset2dAsyncInstrumenter);
BasicInstrumenterDecl(HipMemset2dInstrumenter);
BasicInstrumenterDecl(HipStreamWaitEventInstrumenter);

class HipMallocPitch : public SimpleInstrumenter<HipMallocPitch> {
 public:
  HipMallocPitch(callback::FunctionDecl* decls) {
    setup("hipMallocPitch", &decls->cusan_device_alloc.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //(void** devPtr, size_t* pitch, size_t width, size_t height )
    assert(args.size() == 4);
    auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8Ty()->getPointerTo());

    //"The function may pad the allocation"
    //"*pitch by hipMallocPitch() is the width in bytes of the allocation"
    auto* pitch = irb.CreateLoad(irb.getIntPtrTy(irb.GetInsertBlock()->getModule()->getDataLayout()), args[1]);
    // auto* width = args[2];
    auto* height = args[3];

    auto* real_size = irb.CreateMul(pitch, height);
    return {ptr, real_size};
  }
};

class HipEventQuery : public SimpleInstrumenter<HipEventQuery> {
 public:
  HipEventQuery(callback::FunctionDecl* decls);
  static llvm::SmallVector<Value*> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args);
  static llvm::SmallVector<Value*, 1> map_return_value(IRBuilder<>& irb, Value* result);
};
class HipStreamQuery : public SimpleInstrumenter<HipStreamQuery> {
 public:
  HipStreamQuery(callback::FunctionDecl* decls);
  static llvm::SmallVector<Value*> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args);
  static llvm::SmallVector<Value*, 1> map_return_value(IRBuilder<>& irb, Value* result);
};

// TODO

// BasicInstrumenterDecl(EventRecordFlagsInstrumenter);
// BasicInstrumenterDecl(CudaHostAlloc);
// BasicInstrumenterDecl(CudaMallocHost);

// BasicInstrumenterDecl(CudaHostRegister);
// BasicInstrumenterDecl(CudaHostUnregister);
// BasicInstrumenterDecl(CudaHostFree);

}  // namespace transform
}  // namespace cusan

#endif
