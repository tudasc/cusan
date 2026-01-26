// cusan library
// Copyright (c) 2023-2024 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#include "CusanRuntime.h"

#include <cassert>
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

namespace cusan::runtime {

DeviceID get_current_device_id() {
  DeviceID res;
  hipGetDevice(&res);
  return res;
}

cusan_memcpy_kind infer_memcpy_direction(const void* target, const void* from) {
  // Note: unlike CUDA they do not specify that unified addressing is needed

  hipPointerAttribute_t target_attribs;
  hipPointerGetAttributes(&target_attribs, target);
  hipPointerAttribute_t from_attribs;
  hipPointerGetAttributes(&from_attribs, target);
  bool targetIsHostMem = target_attribs.type == hipMemoryType::hipMemoryTypeUnregistered ||
                         target_attribs.type == hipMemoryType::hipMemoryTypeHost;
  bool fromIsHostMem = target_attribs.type == hipMemoryType::hipMemoryTypeUnregistered ||
                       target_attribs.type == hipMemoryType::hipMemoryTypeHost;

  if (!fromIsHostMem && !targetIsHostMem) {
    return cusan_MemcpyDeviceToDevice;
  }
  if (!fromIsHostMem && targetIsHostMem) {
    return cusan_MemcpyDeviceToHost;
  }
  if (fromIsHostMem && !targetIsHostMem) {
    return cusan_MemcpyHostToDevice;
  }
  // if (fromIsHostMem && targetIsHostMem) {
  return cusan_MemcpyHostToHost;
  // }
}
}  // namespace cusan::runtime