// cusan library
// Copyright (c) 2023-2025 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUSAN_FUNCTIONDECL_H
#define CUSAN_FUNCTIONDECL_H

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

namespace cusan {
namespace callback {
struct CusanFunction {
  const std::string name;
  llvm::FunctionCallee f{nullptr};
  llvm::SmallVector<llvm::Type*, 4> arg_types{};
};

struct FunctionDecl {
  CusanFunction cusan_register_access{"_cusan_kernel_register"};
  CusanFunction cusan_event_record{"_cusan_event_record"};
  CusanFunction cusan_sync_device{"_cusan_sync_device"};
  CusanFunction cusan_sync_stream{"_cusan_sync_stream"};
  CusanFunction cusan_sync_event{"_cusan_sync_event"};
  CusanFunction cusan_event_create{"_cusan_create_event"};
  CusanFunction cusan_stream_create{"_cusan_create_stream"};
  CusanFunction cusan_memset_2d_async{"_cusan_memset_2d_async"};
  CusanFunction cusan_memset_async{"_cusan_memset_async"};
  CusanFunction cusan_memcpy_async{"_cusan_memcpy_async"};
  CusanFunction cusan_memset{"_cusan_memset"};
  CusanFunction cusan_memset_2d{"_cusan_memset_2d"};
  CusanFunction cusan_memcpy{"_cusan_memcpy"};
  CusanFunction cusan_memcpy_2d{"_cusan_memcpy_2d"};
  CusanFunction cusan_memcpy_2d_async{"_cusan_memcpy_2d_async"};
  CusanFunction cusan_stream_wait_event{"_cusan_stream_wait_event"};
  CusanFunction cusan_host_alloc{"_cusan_host_alloc"};
  CusanFunction cusan_managed_alloc{"_cusan_managed_alloc"};
  CusanFunction cusan_host_free{"_cusan_host_free"};
  CusanFunction cusan_host_register{"_cusan_host_register"};
  CusanFunction cusan_host_unregister{"_cusan_host_unregister"};
  CusanFunction cusan_device_alloc{"_cusan_device_alloc"};
  CusanFunction cusan_set_device{"_cusan_set_device"};
  CusanFunction cusan_choose_device{"_cusan_choose_device"};
  CusanFunction cusan_device_free{"_cusan_device_free"};
  CusanFunction cusan_stream_query{"_cusan_stream_query"};
  CusanFunction cusan_event_query{"_cusan_event_query"};
  CusanFunction cusan_sync_callback{"cusan_sync_callback"};
  void initialize(llvm::Module& m);
};

}  // namespace callback
}  // namespace cusan

#endif
