// clang-format off

// RUN: %wrapper-cc %emit-host-only --cusan-kernel-data=%t.yaml -x cuda --cuda-gpu-arch=sm_72 %s 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR

// CHECK-LLVM-IR: @main(i32 noundef %0, {{i8\*\*|ptr}} noundef %1)
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaMallocHost
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_host_alloc
// CHECK-LLVM-IR: {{(call|invoke)}} noundef i32 @_ZL13cudaHostAllocIiE9cudaErrorPPT_mj
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaFreeHost({{.*}}[[free_ptr1:%[0-9a-z]+]])
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_host_free({{.*}}[[free_ptr1]])
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaFreeHost({{.*}}[[free_ptr2:%[0-9a-z]+]])
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_host_free({{.*}}[[free_ptr2]])

// CHECK-LLVM-IR: _ZL13cudaHostAllocIiE9cudaErrorPPT_mj
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaHostAlloc({{.*}}[[host_alloc_ptr:%[0-9a-z]+]])
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_host_alloc({{.*}}[[host_alloc_ptr]])

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
