// cusan library
// Copyright (c) 2023-2025 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#include "CusanRuntime.h"

#include <cassert>
#include <cuda_runtime_api.h>

namespace cusan::runtime {

DeviceID get_current_device_id() {
  DeviceID res;
  cudaGetDevice(&res);
  return res;
}

cusan_memcpy_kind infer_memcpy_direction(const void* target, const void* from) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  assert(prop.unifiedAddressing && "Can only use default direction for memcpy when Unified memory is supported.");

  cudaPointerAttributes target_attribs;
  cudaPointerGetAttributes(&target_attribs, target);
  cudaPointerAttributes from_attribs;
  cudaPointerGetAttributes(&from_attribs, target);
  bool targetIsHostMem = target_attribs.type == cudaMemoryType::cudaMemoryTypeUnregistered ||
                         target_attribs.type == cudaMemoryType::cudaMemoryTypeHost;
  bool fromIsHostMem = target_attribs.type == cudaMemoryType::cudaMemoryTypeUnregistered ||
                       target_attribs.type == cudaMemoryType::cudaMemoryTypeHost;

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