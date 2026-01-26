// clang-format off
// RUN: %wrapper-hip %clang_args -x hip %s -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck --allow-empty %s

// REQUIRES: hip

// clang-format on

// CHECK-NOT: data race
// CHECK-NOT: [Error] sync

#include <hip/hip_runtime.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
  const int size = 512;
  int* h_data1;
  hipMallocHost((void**)&h_data1, size * sizeof(int));
  int* h_data2;
  hipHostAlloc(&h_data2, size * sizeof(int), hipHostAllocDefault);
  hipFreeHost(h_data1);
  hipFreeHost(h_data2);
  return 0;
}
