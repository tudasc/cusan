// clang-format off
// RUN: %rm-file %t.yaml 

// RUN: %wrapper-cc %clang-pass-only-args --cusan-kernel-data=%t.yaml -x cuda --cuda-gpu-arch=sm_72 %s 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR

// CHECK-LLVM-IR: @main(
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaMallocHost
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_host_alloc
// CHECK-LLVM-IR: {{(call|invoke)}}{{.*}} @_ZL13cudaHostAllocIiE9cudaErrorPPT_mj
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaFreeHost({{.*}}[[free_ptr1:%[0-9a-z]+]])
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_host_free({{.*}}[[free_ptr1]])
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaFreeHost({{.*}}[[free_ptr2:%[0-9a-z]+]])
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_host_free({{.*}}[[free_ptr2]])

// CHECK-LLVM-IR: define{{.*}} @_ZL13cudaHostAllocIiE9cudaErrorPPT_mj
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaHostAlloc({{.*}}, i64 {{.*}}[[host_alloc_size:[0-9]+]],
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_host_alloc({{.*}}, i64 {{.*}}[[host_alloc_size]],

// clang-format on

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
