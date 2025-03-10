// cusan library
// Copyright (c) 2023-2024 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#include "CusanPass.h"

#include "../analysis/KernelAnalysis.h"
#include "AnalysisTransform.h"
#include "CommandLine.h"
#include "FunctionDecl.h"
#include "support/CudaUtil.h"
#include "support/Logger.h"
#include "support/Util.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>

using namespace llvm;

namespace cusan {

class CusanPass : public llvm::PassInfoMixin<CusanPass> {
  cusan::ModelHandler kernel_models_;
  callback::FunctionDecl cusan_decls_;

 public:
  llvm::PreservedAnalyses run(llvm::Module&, llvm::ModuleAnalysisManager&);

  bool runOnModule(llvm::Module&);

  bool runOnFunc(llvm::Function&);

  bool runOnKernelFunc(llvm::Function&);
};

class LegacyCusanPass : public llvm::ModulePass {
 private:
  CusanPass pass_impl_;

 public:
  static char ID;  // NOLINT

  LegacyCusanPass() : ModulePass(ID){};

  bool runOnModule(llvm::Module& module) override;

  ~LegacyCusanPass() override = default;
};

bool LegacyCusanPass::runOnModule(llvm::Module& module) {
  const auto modified = pass_impl_.runOnModule(module);
  return modified;
}

llvm::PreservedAnalyses CusanPass::run(llvm::Module& module, llvm::ModuleAnalysisManager& AM) {
  auto promote_pass_preserved = llvm::PreservedAnalyses::all();
  const bool is_device_code   = llvm::StringRef(module.getTargetTriple()).contains("nvptx64-nvidia-cuda");
  if (is_device_code) {
    ModulePassManager module_pass_manager;
    module_pass_manager.addPass(createModuleToFunctionPassAdaptor(llvm::PromotePass()));
    promote_pass_preserved = module_pass_manager.run(module, AM);
  }

  const auto changed = runOnModule(module);

  if (!is_device_code) {
    cusan::util::dump_module_if(module, "CUSAN_DUMP_HOST_IR");
  } else {
    cusan::util::dump_module_if(module, "CUSAN_DUMP_DEVICE_IR");
  }

  return changed ? llvm::PreservedAnalyses::none() : promote_pass_preserved;
}

bool CusanPass::runOnModule(llvm::Module& module) {
  cusan_decls_.initialize(module);
  const auto kernel_models_file = [&]() {
    if (cl_cusan_kernel_file.getNumOccurrences()) {
      return cl_cusan_kernel_file.getValue();
    }

    const auto* data_file = getenv("CUSAN_KERNEL_DATA_FILE");
    if (data_file) {
      return std::string{data_file};
    }

    for (llvm::DICompileUnit* cu : module.debug_compile_units()) {
      if (!cu->getFilename().empty()) {
        return std::string{cu->getFilename()} + "-data.yaml";
      }
    }
    return std::string{"cusan-kernel.yaml"};
  }();

  LOG_DEBUG("Using model data file " << kernel_models_file)
  const auto result = io::load(this->kernel_models_, kernel_models_file);

  const auto changed      = llvm::count_if(module.functions(), [&](auto& func) {
                         if (cuda::is_kernel(&func)) {
                           return runOnKernelFunc(func);
                         }
                         return runOnFunc(func);
                       }) > 1;
  const auto store_result = io::store(this->kernel_models_, kernel_models_file);
  return changed;
}

bool CusanPass::runOnKernelFunc(llvm::Function& function) {
  if (function.isDeclaration()) {
    return false;
  }
  LOG_DEBUG("[Device] running on kernel: " << function.getName());
  auto data = device::analyze_device_kernel(&function);
  if (data) {
    if (!cl_cusan_quiet.getValue()) {
      LOG_DEBUG("[Device] Kernel data: " << data.value())
    }
    this->kernel_models_.insert(data.value());
  }

  return false;
}

bool CusanPass::runOnFunc(llvm::Function& function) {
  const auto stub_name = util::try_demangle(function);

  if (util::starts_with_any_of(stub_name, "__tsan", "__typeart", "_cusan_", "MPI::", "std::", "MPI_")) {
    return false;
  }

  bool modified = false;

  modified |= transform::DeviceSyncInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::StreamSyncInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::EventSyncInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::EventRecordInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::EventRecordFlagsInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::CudaEventCreateInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::CudaEventCreateWithFlagsInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::StreamCreateInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::CudaMemset2dAsyncInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::CudaMemsetAsyncInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::CudaMemcpyAsyncInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::CudaMemset2dInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::CudaMemsetInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::CudaMemcpyInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::CudaMemcpy2DInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::CudaMemcpy2DAsyncInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::StreamWaitEventInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::CudaMallocHost(&cusan_decls_).instrument(function);
  modified |= transform::CudaHostAlloc(&cusan_decls_).instrument(function);
  modified |= transform::CudaHostFree(&cusan_decls_).instrument(function);
  modified |= transform::CudaHostRegister(&cusan_decls_).instrument(function);
  modified |= transform::CudaHostUnregister(&cusan_decls_).instrument(function);
  modified |= transform::CudaMallocManaged(&cusan_decls_).instrument(function);
  modified |= transform::CudaMalloc(&cusan_decls_).instrument(function);
  modified |= transform::CudaFree(&cusan_decls_).instrument(function);
  modified |= transform::CudaStreamQuery(&cusan_decls_).instrument(function);
  modified |= transform::CudaEventQuery(&cusan_decls_).instrument(function);
  modified |= transform::StreamCreateWithFlagsInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::StreamCreateWithPriorityInstrumenter(&cusan_decls_).instrument(function);
  modified |= transform::CudaMallocPitch(&cusan_decls_).instrument(function);
  modified |= transform::CudaChooseDevice(&cusan_decls_).instrument(function);
  modified |= transform::CudaSetDevice(&cusan_decls_).instrument(function);

  //callbacks
  modified |= transform::CudaDeviceSyncCallback(&cusan_decls_).instrument(function);
  modified |= transform::CudaEventSyncCallback(&cusan_decls_).instrument(function);
  modified |= transform::CudaStreamSyncCallback(&cusan_decls_).instrument(function);

  auto data_for_host = host::kernel_model_for_stub(&function, this->kernel_models_);
  if (data_for_host) {
    LOG_FATAL("Found kernel data for " << util::try_demangle_fully(function) << ": "
                                       << data_for_host.value().kernel_name)
    modified |= transform::CallInstrumenter(analysis::CudaKernelInvokeCollector{data_for_host.value()},
                                            transform::KernelInvokeTransformer{&cusan_decls_}, function)
                    .instrument();
  }
  return modified;
}

}  // namespace cusan

#define DEBUG_TYPE "cusan-pass"

//.....................
// New PM
//.....................
llvm::PassPluginLibraryInfo getCusanPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "cusan", LLVM_VERSION_STRING, [](PassBuilder& pass_builder) {
            pass_builder.registerPipelineStartEPCallback([](auto& MPM, OptimizationLevel) {
              // LOG_DEBUG("Opt " << l.getSizeLevel() << " " << l.getSpeedupLevel() << " " << l.O0.getSpeedupLevel())
              MPM.addPass(cusan::CusanPass());
            });
#if (LLVM_VERSION_MAJOR == 14) && !defined(CUSAN_TYPEART)
            pass_builder.registerPipelineParsingCallback(
                [](StringRef name, ModulePassManager& module_pm, ArrayRef<PassBuilder::PipelineElement>) {
                  if (name == "cusan") {
                    module_pm.addPass(cusan::CusanPass());
                    return true;
                  }
                  return false;
                });
#endif
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getCusanPassPluginInfo();
}

//.....................
// Old PM
//.....................
char cusan::LegacyCusanPass::ID = 0;  // NOLINT

static RegisterPass<cusan::LegacyCusanPass> x("cusan", "Cusan Pass");  // NOLINT

ModulePass* createCusanPass() {
  return new cusan::LegacyCusanPass();
}

extern "C" void AddCusanPass(LLVMPassManagerRef pass_manager) {
  unwrap(pass_manager)->add(createCusanPass());
}
