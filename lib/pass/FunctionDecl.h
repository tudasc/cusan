// cusan library
// Copyright (c) 2023-2024 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUSAN_FUNCTIONDECL_H
#define CUSAN_FUNCTIONDECL_H

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

using namespace llvm;
namespace cusan {
namespace callback {
struct CusanFunction {
  const std::string name;
  FunctionCallee f{nullptr};
  SmallVector<Type*, 4> arg_types{};
};

struct FunctionDecl {
  CusanFunction cusan_register_access{"_cusan_kernel_register"};

  CusanFunction cusan_event_record{"_cusan_event_record"};

  CusanFunction cusan_sync_device{"_cusan_sync_device"};
  CusanFunction cusan_sync_stream{"_cusan_sync_stream"};
  CusanFunction cusan_sync_event{"_cusan_sync_event"};
  CusanFunction cusan_event_create{"_cusan_create_event"};
  CusanFunction cusan_stream_create{"_cusan_create_stream"};
  CusanFunction cusan_memset_async{"_cusan_memset_async"};
  CusanFunction cusan_memcpy_async{"_cusan_memcpy_async"};
  CusanFunction cusan_memset{"_cusan_memset"};
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
  CusanFunction cusan_device_free{"_cusan_device_free"};
  CusanFunction cusan_stream_query{"_cusan_stream_query"};
  CusanFunction cusan_event_query{"_cusan_event_query"};

  void initialize(Module& m) {
    using namespace llvm;
    auto& c = m.getContext();

    const auto add_optimizer_attributes = [&](auto& arg) {
      arg.addAttr(Attribute::NoCapture);
      arg.addAttr(Attribute::ReadOnly);
    };

    const auto make_function = [&](auto& f_struct, auto f_types) {
      auto func_type     = f_types.empty() ? FunctionType::get(Type::getVoidTy(c), false)
                                           : FunctionType::get(Type::getVoidTy(c), f_types, false);
      auto func_callee   = m.getOrInsertFunction(f_struct.name, func_type);
      f_struct.f         = func_callee;
      f_struct.arg_types = std::move(f_types);
      if (auto f = dyn_cast<Function>(f_struct.f.getCallee())) {
        f->setLinkage(GlobalValue::ExternalLinkage);
        if (f->arg_size() == 0) {
          return;
        }
        auto& first_param = *(f->arg_begin());
        if (first_param.getType()->isPointerTy()) {
          add_optimizer_attributes(first_param);
        }
      }
    };
    using ArgTypes = decltype(CusanFunction::arg_types);
    // TODO address space?
    ArgTypes arg_types_cusan_register = {PointerType::get(Type::getInt8PtrTy(c), 0), Type::getInt16PtrTy(c),
                                         Type::getInt32Ty(c), Type::getInt8PtrTy(c)};
    make_function(cusan_register_access, arg_types_cusan_register);

    ArgTypes arg_types_sync_device = {};
    make_function(cusan_sync_device, arg_types_sync_device);

    ArgTypes arg_types_sync_stream = {Type::getInt8PtrTy(c)};
    make_function(cusan_sync_stream, arg_types_sync_stream);

    ArgTypes arg_types_sync_event = {Type::getInt8PtrTy(c)};
    make_function(cusan_sync_event, arg_types_sync_event);
    ArgTypes arg_types_event_record = {Type::getInt8PtrTy(c), Type::getInt8PtrTy(c)};
    make_function(cusan_event_record, arg_types_event_record);

    ArgTypes arg_types_event_create = {Type::getInt8PtrTy(c)};
    make_function(cusan_event_create, arg_types_event_create);

    ArgTypes arg_types_stream_create = {Type::getInt8PtrTy(c), Type::getInt32Ty(c)};
    make_function(cusan_stream_create, arg_types_stream_create);

    auto size_t_ty = m.getDataLayout().getIntPtrType(c);

    // void* devPtr, int  value, size_t count, RawStream* stream
    ArgTypes arg_types_memset_async = {Type::getInt8PtrTy(c), Type::getInt32Ty(c), size_t_ty, Type::getInt8PtrTy(c)};
    make_function(cusan_memset_async, arg_types_memset_async);

    // void* dst, const void* src
    ArgTypes arg_types_memcpy_async = {Type::getInt8PtrTy(c), Type::getInt8PtrTy(c),
                                       // size_t count, MemcpyKind kind, RawStream stream
                                       size_t_ty, Type::getInt32Ty(c), Type::getInt8PtrTy(c)};
    make_function(cusan_memcpy_async, arg_types_memcpy_async);

    // void* devPtr, int  value, size_t count
    ArgTypes arg_types_memset = {Type::getInt8PtrTy(c), Type::getInt32Ty(c), size_t_ty};
    make_function(cusan_memset, arg_types_memset);

    // void* dst, const void* src
    ArgTypes arg_types_memcpy = {Type::getInt8PtrTy(c), Type::getInt8PtrTy(c),
                                 // size_t count, MemcpyKind kind
                                 size_t_ty, Type::getInt32Ty(c)};
    make_function(cusan_memcpy, arg_types_memcpy);

    ArgTypes arg_types_stream_wait_event = {Type::getInt8PtrTy(c), Type::getInt8PtrTy(c), Type::getInt32Ty(c)};
    make_function(cusan_stream_wait_event, arg_types_stream_wait_event);

    ArgTypes arg_types_host_alloc = {Type::getInt8PtrTy(c), size_t_ty, Type::getInt32Ty(c)};
    make_function(cusan_host_alloc, arg_types_host_alloc);

    ArgTypes arg_types_host_register = {Type::getInt8PtrTy(c), size_t_ty, Type::getInt32Ty(c)};
    make_function(cusan_host_register, arg_types_host_register);

    ArgTypes arg_types_host_unregister = {Type::getInt8PtrTy(c)};
    make_function(cusan_host_unregister, arg_types_host_unregister);

    ArgTypes arg_types_host_free = {Type::getInt8PtrTy(c)};
    make_function(cusan_host_free, arg_types_host_free);

    ArgTypes arg_types_managed_alloc = {Type::getInt8PtrTy(c), size_t_ty, Type::getInt32Ty(c)};
    make_function(cusan_managed_alloc, arg_types_managed_alloc);

    ArgTypes arg_device_alloc = {Type::getInt8PtrTy(c), size_t_ty};
    make_function(cusan_device_alloc, arg_device_alloc);

    ArgTypes arg_device_free = {Type::getInt8PtrTy(c)};
    make_function(cusan_device_free, arg_device_free);

    // RawStream stream, u32 return_errType
    ArgTypes arg_stream_query = {Type::getInt8PtrTy(c), Type::getInt32Ty(c)};
    make_function(cusan_stream_query, arg_stream_query);

    // Event stream, u32 return_errType
    ArgTypes arg_event_query = {Type::getInt8PtrTy(c), Type::getInt32Ty(c)};
    make_function(cusan_event_query, arg_event_query);

    // void* target, size_t dpitch, const void* from, size_t spitch, size_t width, size_t height, cusan_MemcpyKind kind
    ArgTypes arg_types_memcpy_2d = {
        Type::getInt8PtrTy(c), size_t_ty, Type::getInt8PtrTy(c), size_t_ty, size_t_ty, size_t_ty, Type::getInt32Ty(c)};
    make_function(cusan_memcpy_2d, arg_types_memcpy_2d);

        // void* target, size_t dpitch, const void* from, size_t spitch, size_t width, size_t height, cusan_MemcpyKind kind
    ArgTypes arg_types_memcpy_2d_async = {
        Type::getInt8PtrTy(c), size_t_ty, Type::getInt8PtrTy(c), size_t_ty, size_t_ty, size_t_ty, Type::getInt32Ty(c), Type::getInt8PtrTy(c)};
    make_function(cusan_memcpy_2d_async, arg_types_memcpy_2d_async);
  }
};

}  // namespace callback
}  // namespace cusan

#endif
