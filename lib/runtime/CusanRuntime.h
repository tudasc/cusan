// cusan library
// Copyright (c) 2023-2024 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LIB_RUNTIME_CUSAN_H_
#define LIB_RUNTIME_CUSAN_H_
#include <cstddef>

#ifdef __cplusplus

extern "C" {
#endif

typedef enum cusan_memcpy_kind_t : unsigned int {
  cusan_MemcpyHostToHost     = 0,
  cusan_MemcpyHostToDevice   = 1,
  cusan_MemcpyDeviceToHost   = 2,
  cusan_MemcpyDeviceToDevice = 3,
  cusan_MemcpyDefault        = 4,
} cusan_MemcpyKind;

typedef enum cusan_stream_create_flags_t : unsigned int {
  cusan_StreamFlagsDefault     = 0,
  cusan_StreamFlagsNonBlocking = 1,
} cusan_StreamCreateFlags;
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace cusan::runtime {
using TsanFiber = void*;
using Event     = const void*;
using RawStream = const void*;
using DeviceID  = int;
cusan_MemcpyKind infer_memcpy_direction(const void* target, const void* from);
DeviceID get_current_device_id();
}  // namespace cusan::runtime
using cusan::runtime::DeviceID;
using cusan::runtime::Event;
using cusan::runtime::RawStream;
using cusan::runtime::TsanFiber;
#else
#define TsanFiber void*
#define Event const void*
#define RawStream const void*
#define DeviceID int
#endif

#ifdef __cplusplus

extern "C" {
#endif

void _cusan_kernel_register(void** kernel_args, short* modes, int n, RawStream stream);

void _cusan_sync_device();
void _cusan_set_device(DeviceID device);
void _cusan_choose_device(DeviceID* device);

void _cusan_sync_stream(RawStream stream);
void _cusan_create_stream(RawStream* stream, cusan_StreamCreateFlags flags);
void _cusan_stream_query(RawStream stream, unsigned int err);

void _cusan_sync_event(Event event);
void _cusan_event_record(Event event, RawStream stream);
void _cusan_create_event(RawStream* event);
void _cusan_event_query(Event event, unsigned int err);
void _cusan_stream_wait_event(RawStream stream, Event event, unsigned int flags);
void _cusan_stream_wait_event(RawStream stream, Event event, unsigned int flags);

void _cusan_memcpy(void* target, const void* from, size_t count, cusan_MemcpyKind);
void _cusan_memcpy_async(void* target, const void* from, size_t count, cusan_MemcpyKind kind, RawStream stream);
void _cusan_memcpy_2d(void* target, size_t dpitch, const void* from, size_t spitch, size_t width, size_t height,
                      cusan_MemcpyKind);
void _cusan_memcpy_2d_async(void* target, size_t dpitch, const void* from, size_t spitch, size_t width, size_t height,
                            cusan_MemcpyKind, RawStream stream);

void _cusan_memset(void* target, size_t count);
void _cusan_memset_async(void* target, size_t count, RawStream stream);
void _cusan_memset_2d(void* target, size_t pitch, size_t width, size_t height, cusan_MemcpyKind);
void _cusan_memset_2d_async(void* target, size_t pitch, size_t width, size_t height, cusan_MemcpyKind,
                            RawStream stream);

void _cusan_host_alloc(void** ptr, size_t size, unsigned int flags);
void _cusan_host_free(void* ptr);
void _cusan_managed_alloc(void** ptr, size_t size, unsigned int flags);
void _cusan_managed_free(void* ptr);
void _cusan_host_register(void* ptr, size_t size, unsigned int flags);
void _cusan_host_unregister(void* ptr);
void _cusan_device_alloc(void** ptr, size_t size);
void _cusan_device_free(void* ptr);

typedef enum cusan_sync_type_t : unsigned char {
  cusan_Device = 0,
  cusan_Stream = 1,
  cusan_Event  = 2,
} cusan_SyncType;

void cusan_sync_callback(cusan_SyncType /*type*/, unsigned int /*return_value*/);

#ifdef __cplusplus
}
#endif

#endif /* LIB_RUNTIME_CUSAN_H_ */
