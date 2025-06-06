// clang-format off
// RUN: %rm-file %t.yaml 
// RUN: %wrapper-cc %clang-pass-only-args --cusan-kernel-data=%t.yaml -x cuda --cuda-gpu-arch=sm_72 %s 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR

// CHECK-LLVM-IR: @main(
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaHostRegister({{i8\*|ptr}} {{.*}}[[unregister_ptr:%[0-9a-z]+]]
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_host_register({{i8\*|ptr}} {{.*}}[[unregister_ptr]]
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaMemcpy({{i8\*|ptr}} {{.*}}[[target:%[0-9a-z]+]], {{i8\*|ptr}} {{.*}}[[from:%[0-9a-z]+]],
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_memcpy({{i8\*|ptr}} {{.*}}[[target]], {{i8\*|ptr}} {{.*}}[[from]], 
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaHostUnregister({{i8\*|ptr}} {{.*}}[[unregister_ptr:%[0-9a-z]+]]
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_host_unregister({{i8\*|ptr}} {{.*}}[[unregister_ptr]]

// clang-format on

#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
  const int size = 512;
  int* h_data    = (int*)malloc(size * sizeof(int));
  cudaHostRegister(h_data, size * sizeof(int), cudaHostRegisterDefault);
  int* h_data2;
  cudaHostAlloc(&h_data2, size * sizeof(int), cudaHostAllocDefault);

  memset(h_data, 0, size * sizeof(int));
  cudaMemcpy(h_data, h_data, size * sizeof(int), cudaMemcpyDefault);
  for (int i = 0; i < size; i++) {
    const int buf_v = h_data[i];
    if (buf_v != 0) {
      printf("[Error] sync\n");
      break;
    }
  }
  cudaHostUnregister(h_data);
  cudaFreeHost(h_data2);

  free(h_data);
  return 0;
}
