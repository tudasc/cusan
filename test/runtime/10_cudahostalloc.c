// clang-format off
// RUN: %wrapper-cxx %tsan-compile-flags -O1 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck --allow-empty %s
// clang-format on

// CHECK-NOT: data race
// CHECK-NOT: [Error] sync



#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
  const int size = 512;
  int* h_data1;
  cudaMallocHost((void**)&h_data1, size * sizeof(int));
  int* h_data2;
  cudaHostAlloc(&h_data2, size * sizeof(int), cudaHostAllocDefault);
  cudaFreeHost(h_data1);
  cudaFreeHost(h_data2);
  return 0;
}
