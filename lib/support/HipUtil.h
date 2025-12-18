// cusan library
// Copyright (c) 2023-2024 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUSAN_HIPUTIL_H
#define CUSAN_HIPUTIL_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

namespace cusan::hip {

inline bool is_kernel(const llvm::Function* function) {
  return function->getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL;
}

}  // namespace cusan::hip

#endif  // CUSAN_HIPUTIL_H
