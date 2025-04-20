// cusan library
// Copyright (c) 2023-2024 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LIB_STATSCOUNTER_CUSAN_H_
#define LIB_STATSCOUNTER_CUSAN_H_

#include "support/Logger.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits.h>
#include <map>
#include <numeric>
#include <vector>

#define CUSAN_CUDA_EVENT_LIST                 \
  cusan_stat_handle(event_query_calls);       \
  cusan_stat_handle(stream_query_calls);      \
  cusan_stat_handle(device_free_calls);       \
  cusan_stat_handle(device_alloc_calls);      \
  cusan_stat_handle(managed_alloc_calls);     \
  cusan_stat_handle(host_unregister_calls);   \
  cusan_stat_handle(host_register_calls);     \
  cusan_stat_handle(stream_wait_event_calls); \
  cusan_stat_handle(memset_async_calls);      \
  cusan_stat_handle(memcpy_async_calls);      \
  cusan_stat_handle(memset_calls);            \
  cusan_stat_handle(memcpy_calls);            \
  cusan_stat_handle(create_event_calls);      \
  cusan_stat_handle(create_stream_calls);     \
  cusan_stat_handle(sync_event_calls);        \
  cusan_stat_handle(sync_stream_calls);       \
  cusan_stat_handle(sync_device_calls);       \
  cusan_stat_handle(event_record_calls);      \
  cusan_stat_handle(kernel_register_calls);   \
  cusan_stat_handle(host_free_calls);         \
  cusan_stat_handle(host_alloc_calls);

namespace cusan::runtime {

namespace softcounter {

using Counter       = long long int;
using AtomicCounter = std::atomic<Counter>;

class Statistics {
 private:
  std::vector<unsigned> numbers{};
  unsigned min{UINT_MAX};
  unsigned max{0};

 public:
  void addNumber(unsigned number) {
    numbers.push_back(number);

    if (number < min) {
      min = number;
    }
    if (number > max) {
      max = number;
    }
  }

  double getAvg() const {
    if (numbers.empty()) {
      return 0.0;
    }
    const double sum   = std::accumulate(numbers.begin(), numbers.end(), 0.0);
    const auto average = sum / double(numbers.size());
    return average;
  }

  void printHist(std::ostream& s) const {
    unsigned bucketSize{12};
    const auto bucket_f = [&](unsigned number) { return unsigned(std::floor(double(number) / double(bucketSize))); };

    std::map<unsigned, unsigned> histogram{};
    for (const auto number : numbers) {
      const auto bucket = bucket_f(number);
      histogram[bucket]++;
    }

    for (const auto& pair : histogram) {
      const auto bytes_low  = pair.first * bucketSize;
      const auto bytes_high = bytes_low + bucketSize - 1;
      s << "[" << bytes_low << " - " << bytes_high << "]: " << pair.second << ", ";
    }
    s << "\n";
  }
};

#define cusan_stat_handle(name) \
  inline void inc_##name() {    \
  }                             \
  inline Counter get_##name() { \
    return 0;                   \
  }

class NoneRecorder {
 public:
  Statistics stats_w;
  Statistics stats_r;
  CUSAN_CUDA_EVENT_LIST
#include "TsanEvents.inc"
  void inc_TsanMemoryReadCount(unsigned) {
  }
  void inc_TsanMemoryWriteCount(unsigned) {
  }
};

#undef cusan_stat_handle
#define cusan_stat_handle(name) \
  AtomicCounter name = 0;       \
  inline void inc_##name() {    \
    this->name++;               \
  }                             \
  inline Counter get_##name() { \
    return this->name;          \
  }

struct AccessRecorder {
 public:
  Statistics stats_w;
  Statistics stats_r;

  CUSAN_CUDA_EVENT_LIST
#include "TsanEvents.inc"

  void inc_TsanMemoryReadCount(unsigned count) {
    this->TsanMemoryRead++;
    stats_r.addNumber(count);
  }
  void inc_TsanMemoryWriteCount(unsigned count) {
    this->TsanMemoryWrite++;
    stats_w.addNumber(count);
  }
};
#undef cusan_stat_handle
}  // namespace softcounter

#ifdef CUSAN_SOFTCOUNTER
using Recorder = softcounter::AccessRecorder;
#else
using Recorder = softcounter::NoneRecorder;
#endif

}  // namespace cusan::runtime

#endif