// clang-format off
// RUN: %wrapper-cxx -O1 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck --allow-empty %s
// clang-format on

// CHECK-NOT: data race
// CHECK-NOT: [Error] sync

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
