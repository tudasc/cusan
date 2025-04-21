// cusan library
// Copyright (c) 2023-2025 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUSAN_CUDAUTIL_H
#define CUSAN_CUDAUTIL_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

namespace cusan::cuda {

// bool AA::isGPU(const Module &M) {
//   Triple T(M.getTargetTriple());
//   return T.isAMDGPU() || T.isNVPTX();
// }

inline bool is_kernel(const llvm::Function* function) {
  const auto* module   = function->getParent();
  const auto* named_md = module->getNamedMetadata("nvvm.annotations");
  if (named_md == nullptr) {
    return false;
  }

  const auto any = llvm::any_of(named_md->operands(), [&](auto* operand) {
    const auto* md_value = llvm::dyn_cast_or_null<llvm::ValueAsMetadata>(operand->getOperand(0).get());
    if (md_value == nullptr) {
      return false;
    }
    return llvm::dyn_cast<llvm::Function>(md_value->getValue()) == function;
  });

  return any;
}

}  // namespace cusan::cuda

#endif  // CUSAN_CUDAUTIL_H
