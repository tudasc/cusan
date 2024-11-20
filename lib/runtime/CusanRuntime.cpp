// cusan library
// Copyright (c) 2023-2024 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#include "CusanRuntime.h"
// clang-format off
#ifdef CUSAN_TYPEART
#include "RuntimeInterface.h"
#endif
#include "analysis/KernelModel.h"
#include "support/Logger.h"
#include "StatsCounter.h"
#if CUSAN_SOFTCOUNTER
#include "support/Table.h"
#endif
#include "TSanInterface.h"
// clang-format on
#include <cstddef>
#include <iostream>
#include <map>

#ifndef CUSAN_SYNC_DETAIL_LEVEL
#define CUSAN_SYNC_DETAIL_LEVEL 0
#endif

namespace cusan::runtime {

struct Stream {
  RawStream handle;

  // blocks the default stream till its done
  bool isBlocking;

  constexpr explicit Stream(const void* h = nullptr, bool isBlocking = true) : handle(h), isBlocking(isBlocking) {
  }
  constexpr bool operator<(const Stream& rhs) const {
    return this->handle < rhs.handle;
  }
  [[nodiscard]] constexpr bool isDefaultStream() const {
    return handle == nullptr;
  }
};

struct AllocationInfo {
  size_t size;
  bool is_pinned  = false;
  bool is_managed = false;
  bool on_device  = false;

  static constexpr AllocationInfo Device(size_t size) {
    return AllocationInfo{size, false, false, true};
  }
  static constexpr AllocationInfo Pinned(size_t size) {
    return AllocationInfo{size, true, false, false};
  }
  static constexpr AllocationInfo Managed(size_t size) {
    return AllocationInfo{size, false, true, false};
  }
};

struct PtrAttribute {
  AccessState state{AccessState::kRW};
  bool is_ptr{false};
};

PtrAttribute access_cast_back(short cb_value) {
  const short access = (cb_value >> 1);
  const bool ptr     = (cb_value & 1) != 0;
  return PtrAttribute{AccessState{access}, ptr};
}

struct PointerAccess {
  size_t alloc_size{0};
  AccessState mode{AccessState::kRW};
};

class Runtime;

class Device {
  friend Runtime;
  // NOTE: assumed to be a ordered map so we can iterate in ascending order
  std::map<const void*, AllocationInfo> allocations_;
  std::map<Stream, TsanFiber> streams_;
  std::map<Event, Stream> events_;

  Device() {
    // every device has a default stream
  }
};

class Runtime {
  std::map<int32_t, Device> devices_;
  int32_t current_device_;
  TsanFiber cpu_fiber_;
  TsanFiber curr_fiber_;
  bool init_;

 public:
  static constexpr Stream kDefaultStream = Stream();
  Recorder stats_recorder;

  static Runtime& get() {
    static Runtime run_t;
    if (!run_t.init_) {
      run_t.current_device_ = get_current_device();
      run_t.cpu_fiber_   = TsanGetCurrentFiber();
      run_t.devices_.insert({run_t.current_device_, {}});
      { run_t.register_stream(kDefaultStream); }
      run_t.init_ = true;
    }
    return run_t;
  }

  Runtime(const Runtime&) = delete;

  void operator=(const Runtime&) = delete;

  Device& get_device(DeviceID id) {
    if (devices_.find(id) == devices_.end()) {
      devices_.insert({id, {}});
    }
    return devices_.at(id);
  }

  void set_device(DeviceID device){
    if (devices_.find(device) == devices_.end()) {
      devices_.insert({device, {}});
    }
    current_device_ = device;
  }

  [[nodiscard]] const std::map<const void*, AllocationInfo>& get_allocations() {
    return get_device(current_device_).allocations_;
  }

  void happens_before() {
    LOG_TRACE("[cusan]    HappensBefore of curr fiber")
    auto& gpu = get_device(current_device_);
    TsanHappensBefore(curr_fiber_);
    stats_recorder.inc_TsanHappensBefore();
  }

  void switch_to_cpu() {
    LOG_TRACE("[cusan]    Switch to cpu")
    auto& gpu = get_device(current_device_);

    auto search_result = gpu.streams_.find(Runtime::kDefaultStream);
    assert(search_result != gpu.streams_.end() && "Tried using stream that wasn't created prior");
    if (curr_fiber_ == search_result->second) {
      LOG_TRACE("[cusan]        syncing all other blocking GPU streams to run after since its default stream")
      for (const auto& [s, sync_var] : gpu.streams_) {
        if (s.isBlocking && !s.isDefaultStream()) {
          LOG_TRACE("[cusan]        happens before " << s.handle)
          TsanHappensBefore(sync_var);
          stats_recorder.inc_TsanHappensBefore();
        }
      }
    }

    // without synchronization
    TsanSwitchToFiber(cpu_fiber_, 1);
    stats_recorder.inc_TsanSwitchToFiber();
    curr_fiber_ = cpu_fiber_;
  }

  void register_stream(Stream stream) {
    auto& gpu = get_device(current_device_);

    static uint32_t n_streams = 0;
    auto search_result        = gpu.streams_.find(stream);
    assert(search_result == gpu.streams_.end() && "Registered stream twice");
    TsanFiber fiber = TsanCreateFiber(0);
    stats_recorder.inc_TsanCreateFiber();
    char name[32];
    snprintf(name, 32, "cuda_stream %u", n_streams++);
    TsanSetFiberName(fiber, name);
    gpu.streams_.insert({stream, fiber});
  }

  void switch_to_stream(Stream stream) {
    LOG_TRACE("[cusan]    Switching to stream: " << stream.handle)
    auto& gpu          = get_device(current_device_);
    auto search_result = gpu.streams_.find(stream);
    assert(search_result != gpu.streams_.end() && "Tried using stream that wasn't created prior");
    TsanSwitchToFiber(search_result->second, 0);
    stats_recorder.inc_TsanSwitchToFiber();
    if (search_result->first.isDefaultStream()) {
      LOG_TRACE("[cusan]        syncing all other blocking GPU streams since its default stream")
      // then we are on the default stream and as such want to synchronize behind all other streams
      // unless they are nonBlocking
      for (auto& [s, sync_var] : gpu.streams_) {
        if (s.isBlocking && !s.isDefaultStream()) {
          LOG_TRACE("[cusan]        happens after " << s.handle)
          TsanHappensAfter(sync_var);
          stats_recorder.inc_TsanHappensAfter();
        }
      }
    }
    curr_fiber_ = search_result->second;
  }

  void happens_after_all_streams(bool onlyBlockingStreams = false) {
    LOG_TRACE("[cusan]    happens_after_all_streams but only blocking ones: " << onlyBlockingStreams)
    auto& gpu = get_device(current_device_);
    for (const auto& [stream, fiber] : gpu.streams_) {
      if (!onlyBlockingStreams || stream.isBlocking) {
        LOG_TRACE("[cusan]        happens after " << stream.handle)
        TsanHappensAfter(fiber);
        stats_recorder.inc_TsanHappensAfter();
      }
    }
  }

  void happens_after_stream(Stream stream) {
    auto& gpu = get_device(current_device_);

    auto search_result = gpu.streams_.find(stream);
    assert(search_result != gpu.streams_.end() && "Tried using stream that wasn't created prior");
    TsanHappensAfter(search_result->second);
    stats_recorder.inc_TsanHappensAfter();
  }

  void record_event(Event event, Stream stream) {
    LOG_TRACE("[cusan]    Record event: " << event << " stream:" << stream.handle);
    auto& gpu          = get_device(current_device_);
    gpu.events_[event] = stream;
  }

  // Sync the event on the current stream
  void sync_event(Event event) {
    auto& gpu = get_device(current_device_);

    auto search_result = gpu.events_.find(event);
    assert(search_result != gpu.events_.end() && "Tried using event that wasn't recorded to prior");
    LOG_TRACE("[cusan]    Sync event: " << event << " recorded on stream:" << gpu.events_[event].handle)
    happens_after_stream(gpu.events_[event]);
  }

  void insert_allocation(void* ptr, AllocationInfo info) {
    auto& gpu = get_device(current_device_);
    assert(gpu.allocations_.find(ptr) == gpu.allocations_.end() && "Registered an allocation multiple times");
    gpu.allocations_[ptr] = info;
  }

  void free_allocation(void* ptr, bool must_exist = true) {
    auto& gpu  = get_device(current_device_);
    bool found = gpu.allocations_.find(ptr) != gpu.allocations_.end();
    if (must_exist) {
      assert(found && "Tried to delete a non existent allocation");
    }
    if (found) {
      gpu.allocations_.erase(ptr);
    }
  }

  AllocationInfo* get_allocation_info(const void* ptr) {
    auto& gpu = get_device(current_device_);
    auto res  = gpu.allocations_.find(ptr);
    if (res == gpu.allocations_.end()) {
      // fallback find if it lies within a region
      // for(auto [alloc_ptr, alloc_info]: allocations_){
      //   if(alloc_ptr < ptr && ((const char*)alloc_ptr) + alloc_info.size > ptr){
      //     return &allocations_[ptr];
      //   }
      // }
      return nullptr;
    }
    return &res->second;
  }

 private:
  Runtime() = default;

  ~Runtime() {
#undef cusan_stat_handle
#define cusan_stat_handle(name) table.put(Row::make(#name, stats_recorder.get_##name()));
#if CUSAN_SOFTCOUNTER
    Table table{"Cusan runtime statistics"};
#ifdef CUSAN_FIBERPOOL
    table.put(Row::make("Fiberpool", 1));
#else
    table.put(Row::make("Fiberpool", 0));
#endif
    CUSAN_CUDA_EVENT_LIST
#include "TsanEvents.inc"
    table.put(Row::make("TsanMemoryReadSize[KB]", stats_recorder.stats_r.getAvg() / 1024.0));
    table.put(Row::make("TsanMemoryWriteSize[KB]", stats_recorder.stats_w.getAvg() / 1024.0));
    table.print(std::cout);
    std::cout << "TsanMemRead stats (size in b): \n";
    stats_recorder.stats_r.printHist(std::cout);
    std::cout << "TsanMemWrite stats (size in b): \n";
    stats_recorder.stats_w.printHist(std::cout);
#endif
#undef cusan_stat_handle
#undef CUSAN_CUDA_EVENT_LIST

#ifdef CUSAN_FIBERPOOL
    // TsanFiberPoolFini();
#endif
  }
};

}  // namespace cusan::runtime

using namespace cusan::runtime;

namespace helper {
#ifndef CUSAN_TYPEART
inline std::optional<size_t> find_memory_alloc_size(Runtime& runtime, const void* ptr) {
  const auto& allocs = runtime.get_allocations();

  // if there exists any allocation
  if (allocs.size() > 0) {
    // find the first allocation that is bigger or equal then what we search for
    const auto subsequent_alloc = allocs.lower_bound(ptr);

    // if its equal we got our match
    if (subsequent_alloc->first == ptr) {
      return subsequent_alloc->second.size;
    }
    // else if there exists a previous allocation
    else if (subsequent_alloc != allocs.begin()) {
      // it is the only one that might include our pointer
      // since all allocations are non overlapping and the start of the allocation needs to be smaller then our ptr
      const auto& alloc = *std::prev(subsequent_alloc);
      assert(alloc.first <= ptr);
      // still got to verify were inside though
      if (((const char*)alloc.first + alloc.second.size) >= ptr) {
        return alloc.second.size;
      }
    }
  }
  return {};
}
#else
inline std::optional<size_t> find_memory_alloc_size(const Runtime&, const void* ptr) {
  size_t alloc_size{0};
  int alloc_id{0};
  auto query_status = typeart_get_type(ptr, &alloc_id, &alloc_size);
  if (query_status != TYPEART_OK) {
    LOG_TRACE(" [cusan]    Querying allocation length failed on " << ptr << ". Code: " << int(query_status))
    return {};
  }
  const auto bytes_for_type = typeart_get_type_size(alloc_id);
  const auto total_bytes    = bytes_for_type * alloc_size;
  LOG_TRACE(" [cusan]    Querying allocation length of " << ptr << ". Code: " << int(query_status) << "  with size"
                                                         << total_bytes)
  return total_bytes;
}
#endif
}  // namespace helper

void _cusan_kernel_register(void** kernel_args, short* modes, int n, RawStream stream) {
  LOG_TRACE("[cusan]Kernel Register with " << n << " Args and on stream:" << stream)
  auto& runtime = Runtime::get();

  llvm::SmallVector<size_t, 4> sizes;
  for (int i = 0; i < n; ++i) {
    const auto mode = cusan::runtime::access_cast_back(modes[i]);
    if (!mode.is_ptr) {
      sizes.push_back(0);
      continue;
    }

    const auto* ptr          = kernel_args[i];
    const auto size_in_bytes = helper::find_memory_alloc_size(runtime, ptr);
    if (!size_in_bytes) {
      LOG_TRACE(" [cusan]    Querying allocation length failed on " << ptr);
      sizes.push_back(0);
      continue;
    }

    sizes.push_back(size_in_bytes.value());
  }

  runtime.stats_recorder.inc_kernel_register_calls();
  runtime.switch_to_stream(Stream(stream));
  for (int i = 0; i < n; ++i) {
    const auto mode = cusan::runtime::access_cast_back(modes[i]);

    auto* ptr              = kernel_args[i];
    const auto total_bytes = sizes[i];
    if (total_bytes == 0) {
      continue;
    }

    if (mode.state == cusan::AccessState::kRW || mode.state == cusan::AccessState::kWritten) {
      LOG_TRACE("[cusan]    Write to " << ptr << " with size " << total_bytes)
      TsanMemoryWritePC(ptr, total_bytes, __builtin_return_address(0));
      runtime.stats_recorder.inc_TsanMemoryWriteCount(total_bytes);
    } else if (mode.state == cusan::AccessState::kRead) {
      LOG_TRACE("[cusan]    Read from " << ptr << " with size " << total_bytes)
      TsanMemoryReadPC(ptr, total_bytes, __builtin_return_address(0));
      runtime.stats_recorder.inc_TsanMemoryReadCount(total_bytes);
    }
  }

  runtime.happens_before();
  runtime.switch_to_cpu();
}

void _cusan_sync_device() {
  LOG_TRACE("[cusan]Sync Device\n")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_sync_device_calls();
  runtime.happens_after_all_streams();
}

void _cusan_event_record(Event event, RawStream stream) {
  LOG_TRACE("[cusan]Event Record")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_event_record_calls();
  runtime.record_event(event, Stream(stream));
}

void _cusan_sync_stream(RawStream raw_stream) {
  LOG_TRACE("[cusan]Sync Stream" << raw_stream)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_sync_stream_calls();
  const auto stream = Stream(raw_stream);
  if (stream.isDefaultStream()) {
    // if we sync the default stream then we implicitly sync and wait on all blocking streams
    LOG_TRACE("[cusan] Since default stream it happens after all")
    runtime.happens_after_all_streams(true);
  } else {
    runtime.happens_after_stream(stream);
  }
}

void _cusan_sync_event(Event event) {
  LOG_TRACE("[cusan]Sync Event" << event)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_sync_event_calls();
  runtime.sync_event(event);
}

void _cusan_create_event(Event*) {
  LOG_TRACE("[cusan]create event")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_create_event_calls();
}

void _cusan_create_stream(RawStream* stream, cusan_StreamCreateFlags flags) {
  LOG_TRACE("[cusan]create stream with flags: " << flags
                                                << " isNonBlocking: " << (bool)(flags & cusan_StreamFlagsNonBlocking))
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_create_stream_calls();
  runtime.register_stream(Stream(*stream, !(bool)(flags & cusan_StreamFlagsNonBlocking)));
}

void _cusan_stream_wait_event(RawStream stream, Event event, unsigned int) {
  LOG_TRACE("[cusan]StreamWaitEvent stream:" << stream << " on event:" << event)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_stream_wait_event_calls();
  runtime.switch_to_stream(Stream(stream));
  runtime.sync_event(event);
  runtime.happens_before();
  runtime.switch_to_cpu();
}

void _cusan_host_alloc(void** ptr, size_t size, unsigned int) {
  // at least based of this presentation and some comments in the cuda forums this syncs the whole device
  //  https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
  LOG_TRACE("[cusan]host alloc " << *ptr << " with size " << size)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_host_alloc_calls();
  // runtime.happens_after_all_streams();

  runtime.insert_allocation(*ptr, AllocationInfo{size, true, false});
}

void _cusan_host_free(void* ptr) {
  LOG_TRACE("[cusan]host free")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_host_free_calls();
  runtime.free_allocation(ptr);
}

void _cusan_host_register(void* ptr, size_t size, unsigned int) {
  LOG_TRACE("[cusan]host register " << ptr << " with size:" << size);
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_host_register_calls();
  runtime.insert_allocation(ptr, AllocationInfo::Pinned(size));
}
void _cusan_host_unregister(void* ptr) {
  LOG_TRACE("[cusan]host unregister " << ptr);
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_host_unregister_calls();
  runtime.free_allocation(ptr);
}

void _cusan_managed_alloc(void** ptr, size_t size, unsigned int) {
  LOG_TRACE("[cusan]Managed host alloc " << *ptr << " with size " << size << " -> implicit device sync")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_managed_alloc_calls();
  runtime.happens_after_all_streams();
  runtime.insert_allocation(*ptr, AllocationInfo::Managed(size));
}

void _cusan_device_alloc(void** ptr, size_t size) {
  // implicit syncs device
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator
  LOG_TRACE("[cusan]Device alloc " << *ptr << " with size " << size << " -> implicit device sync")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_device_alloc_calls();

  runtime.insert_allocation(*ptr, AllocationInfo::Device(size));
  // runtime.switch_to_stream(Stream());
  // runtime.switch_to_cpu();
}
void _cusan_device_free(void* ptr) {
  // implicit syncs device
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator
  LOG_TRACE("[cusan]Device free " << ptr << " -> TODO maybe implicit device sync")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_device_free_calls();
  runtime.happens_after_all_streams();
}

// TODO: get rid of cudaSpecifc check for cudaSuccess 0
void _cusan_stream_query(RawStream stream, unsigned int err) {
  LOG_TRACE("[cusan] Stream query " << stream << " -> " << err)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_stream_query_calls();
  if (err == 0) {
    LOG_TRACE("[cusan]    syncing")

    runtime.happens_after_stream(Stream{stream});
  }
}

// TODO: get rid of cudaSpecifc check for cudaSuccess 0
void _cusan_event_query(Event event, unsigned int err) {
  LOG_TRACE("[cusan] Event query " << event << " -> " << err)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_event_query_calls();
  if (err == 0) {
    LOG_TRACE("[cusan]    syncing")
    runtime.sync_event(event);
  }
}

void _cusan_set_device(DeviceID device) {
  Runtime::get().set_device(device);
}

void _cusan_choose_device(DeviceID* device) {
  // does this function ever return a nullptr?
  // and what would that mean
  assert(device);
  Runtime::get().set_device(*device);
}

void _cusan_memset_async_impl(void* target, size_t count, RawStream stream) {
  // The Async versions are always asynchronous with respect to the host.
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_memset_async_calls();
  runtime.switch_to_stream(Stream(stream));
  TsanMemoryWritePC(target, count, __builtin_return_address(0));
  runtime.stats_recorder.inc_TsanMemoryWrite();
  runtime.happens_before();
  runtime.switch_to_cpu();
}
void _cusan_memset_impl(void* target, size_t count) {
  // The cudaMemset functions are asynchronous with respect to the host except when the target memory is pinned host
  // memory.
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_memset_calls();
  runtime.switch_to_stream(Runtime::kDefaultStream);
  LOG_TRACE("[cusan]    " << "Write to " << target << " with size: " << count)
  TsanMemoryWritePC(target, count, __builtin_return_address(0));
  runtime.stats_recorder.inc_TsanMemoryWrite();
  runtime.happens_before();
  runtime.switch_to_cpu();

  auto* alloc_info = runtime.get_allocation_info(target);
  // if we couldn't find alloc info we just assume the worst and don't sync
  if ((alloc_info && (alloc_info->is_pinned || alloc_info->is_managed)) || CUSAN_SYNC_DETAIL_LEVEL == 0) {
    LOG_TRACE("[cusan]    " << "Memset is blocking")
    runtime.happens_after_stream(Runtime::kDefaultStream);
  } else {
    LOG_TRACE("[cusan]    " << "Memset is not blocking")
    if (!alloc_info) {
      LOG_DEBUG("[cusan]    Failed to get alloc info " << target);
    } else if (!alloc_info->is_pinned && !alloc_info->is_managed) {
      LOG_TRACE("[cusan]    Pinned:" << alloc_info->is_pinned << " Managed:" << alloc_info->is_managed)
    }
  }

  // r.happens_after_stream(Runtime::default_stream));
}

void _cusan_memset_2d(void* target, size_t pitch, size_t, size_t height, cusan_MemcpyKind) {
  _cusan_memset_impl(target, pitch * height);
}
void _cusan_memset_2d_async(void* target, size_t pitch, size_t, size_t height, cusan_MemcpyKind, RawStream stream) {
  _cusan_memset_async_impl(target, pitch * height, stream);
}

void _cusan_memset(void* target, size_t count) {
  LOG_TRACE("[cusan]Memset " << count << " bytes to:" << target)
  _cusan_memset_impl(target, count);
}

void _cusan_memset_async(void* target, size_t count, RawStream stream) {
  LOG_TRACE("[cusan]MemsetAsync" << count << " bytes to:" << target)
  _cusan_memset_async_impl(target, count, stream);
}

void _cusan_memcpy_async_impl(void* target, size_t write_size, const void* from, size_t read_size,
                              cusan_MemcpyKind kind, RawStream stream) {
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_memcpy_async_calls();
  if (kind == cusan_MemcpyHostToHost && CUSAN_SYNC_DETAIL_LEVEL == 1) {
    // 2. For transfers from any host memory to any host memory, the function is fully synchronous with respect to the
    // host.
    LOG_TRACE("[cusan]   Blocking")
    runtime.switch_to_stream(Stream(stream));
    TsanMemoryReadPC(from, read_size, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryRead();
    TsanMemoryWritePC(target, write_size, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryWrite();
    runtime.happens_before();
    runtime.switch_to_cpu();
    runtime.happens_after_stream(Stream(stream));
  } else {
    // 1. For transfers between device memory and pageable host memory, the function *might* be synchronous with respect
    // to host.
    // 2. If pageable memory must first be staged to pinned memory, the driver *may* synchronize with the stream and
    // stage the copy into pinned memory.
    // 4. For all other transfers, the function should be fully asynchronous.
    LOG_TRACE("[cusan]   not Blocking")
    runtime.switch_to_stream(Stream(stream));
    TsanMemoryReadPC(from, read_size, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryRead();
    TsanMemoryWritePC(target, write_size, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryWrite();
    runtime.happens_before();
    runtime.switch_to_cpu();
  }
}

void _cusan_memcpy_impl(void* target, size_t write_size, const void* from, size_t read_size, cusan_MemcpyKind kind) {
  // TODO verify that the memcpy2d beheaviour is actually the same as normal memcpy

  if (kind == cusan_MemcpyDefault) {
    kind = infer_memcpy_direction(target, from);
  }

  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_memcpy_calls();
  if (CUSAN_SYNC_DETAIL_LEVEL == 0) {
    LOG_TRACE("[cusan]   DefaultStream+Blocking")
    // In this mode: Memcpy always blocks, no detailed view w.r.t. memory direction
    runtime.switch_to_stream(Runtime::kDefaultStream);
    TsanMemoryReadPC(from, read_size, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryRead();
    TsanMemoryWritePC(target, write_size, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryWrite();
    runtime.happens_before();
    runtime.switch_to_cpu();
    runtime.happens_after_stream(Runtime::kDefaultStream);
  } else if (kind == cusan_MemcpyDeviceToDevice) {
    // 4. For transfers from device memory to device memory, no host-side synchronization is performed.
    LOG_TRACE("[cusan]   DefaultStream")
    runtime.switch_to_stream(Runtime::kDefaultStream);
    TsanMemoryReadPC(from, read_size, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryRead();
    TsanMemoryWritePC(target, write_size, __builtin_return_address(0));
    runtime.happens_before();
    runtime.switch_to_cpu();
  } else if (kind == cusan_MemcpyDeviceToHost) {
    // 3. For transfers from device to either pageable or pinned host memory, the function returns only once the copy
    // has completed.
    LOG_TRACE("[cusan]   DefaultStream+Blocking")
    runtime.switch_to_stream(Runtime::kDefaultStream);
    TsanMemoryReadPC(from, read_size, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryRead();
    TsanMemoryWritePC(target, write_size, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryWrite();
    runtime.happens_before();
    runtime.switch_to_cpu();
    runtime.happens_after_stream(Runtime::kDefaultStream);
  } else if (kind == cusan_MemcpyHostToDevice) {
    // 1. For transfers from pageable host memory to device memory, a stream sync is performed before the copy is
    // initiated.

    auto* alloc_info = runtime.get_allocation_info(from);
    // if we couldn't find alloc info we just assume the worst and don't sync
    if (alloc_info && !alloc_info->is_pinned) {
      runtime.happens_after_stream(Runtime::kDefaultStream);
      LOG_TRACE("[cusan]   DefaultStream+Blocking")
    } else {
      LOG_TRACE("[cusan]   DefaultStream")
    }
    //   The function will return once the pageable buffer has been copied to the staging memory for DMA transfer to
    //   device memory
    TsanMemoryReadPC(from, read_size, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryRead();
    runtime.switch_to_stream(Runtime::kDefaultStream);
    TsanMemoryWritePC(target, write_size, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryWrite();
    runtime.happens_before();
    runtime.switch_to_cpu();
    runtime.happens_after_stream(Runtime::kDefaultStream);
  } else if (kind == cusan_MemcpyHostToHost) {
    // 5. For transfers from any host memory to any host memory, the function is fully synchronous with respect to the
    // host.
    LOG_TRACE("[cusan]   DefaultStream+Blocking")
    runtime.switch_to_stream(Runtime::kDefaultStream);
    runtime.happens_before();
    runtime.switch_to_cpu();
    runtime.happens_after_stream(Runtime::kDefaultStream);
    TsanMemoryReadPC(from, read_size, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryRead();
    TsanMemoryWritePC(target, write_size, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryWrite();
  } else {
    assert(false && "Should be unreachable");
  }
}

void _cusan_memcpy_2d_async(void* target, size_t dpitch, const void* from, size_t spitch, size_t width, size_t height,
                            cusan_MemcpyKind kind, RawStream stream) {
  LOG_TRACE("[cusan]MemcpyAsync" << width * height << " bytes to:" << target)

  size_t read_size  = spitch * height;
  size_t write_size = dpitch * height;
  _cusan_memcpy_async_impl(target, write_size, from, read_size, kind, stream);
}

void _cusan_memcpy_async(void* target, const void* from, size_t count, cusan_MemcpyKind kind, RawStream stream) {
  LOG_TRACE("[cusan]MemcpyAsync" << count << " bytes to:" << target)
  _cusan_memcpy_async_impl(target, count, from, count, kind, stream);
}

void _cusan_memcpy_2d(void* target, size_t dpitch, const void* from, size_t spitch, size_t width, size_t height,
                      cusan_MemcpyKind kind) {
  LOG_TRACE("[cusan]Memcpy2d " << width * height << " from:" << from << " to:" << target);
  size_t read_size  = spitch * height;
  size_t write_size = dpitch * height;
  _cusan_memcpy_impl(target, write_size, from, read_size, kind);
}

void _cusan_memcpy(void* target, const void* from, size_t count, cusan_MemcpyKind kind) {
  LOG_TRACE("[cusan]Memcpy " << count << " from:" << from << " to:" << target);
  _cusan_memcpy_impl(target, count, from, count, kind);
}
