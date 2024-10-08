// clang-format off
// RUN: %apply %s -strip-debug --cusan-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR
// clang-format on

// CHECK-LLVM-IR: @main(i32 noundef %0, i8** noundef %1)
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaHostRegister(i8* {{.*}}[[unregister_ptr:%[0-9a-z]+]]
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_host_register(i8* {{.*}}[[unregister_ptr]]
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaMemcpy(i8* {{.*}}[[target:%[0-9a-z]+]], i8* {{.*}}[[from:%[0-9a-z]+]],
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_memcpy(i8* {{.*}}[[target]], i8* {{.*}}[[from]],
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaHostUnregister(i8* {{.*}}[[unregister_ptr:%[0-9a-z]+]]
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_host_unregister(i8* {{.*}}[[unregister_ptr]]

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
