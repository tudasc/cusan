// clang-format off
// RUN: %wrapper-cc %clang_args0 -x cuda %s -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %wrapper-cc -DCUSAN_SYNC %clang_args0 -x cuda %s -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// clang-format on

// CHECK-DAG: data race
// CHECK-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync

// XFAIL:*

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel(int** data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
#if __CUDA_ARCH__ >= 700
  for (int i = 0; i < tid; i++) {
    __nanosleep(1000000U);
  }
#else
  printf(">>> __CUDA_ARCH__ !\n");
#endif
  data[-1][tid] = (tid + 1);
}

int main() {
  const int size            = 256;
  const int threadsPerBlock = 256;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int** d_data;  // Unified Memory pointer
  cudaMallocManaged(&d_data, 2 * sizeof(int*));

  // Allocate Unified Memory
  cudaMallocManaged(&d_data[0], size * sizeof(int));
  cudaMallocManaged(&d_data[1], size * sizeof(int));
  cudaMemset(d_data[0], 0, size * sizeof(int));
  cudaMemset(d_data[1], 0, size * sizeof(int));

  kernel<<<blocksPerGrid, threadsPerBlock>>>(&d_data[1]);

#ifdef CUSAN_SYNC
  cudaDeviceSynchronize();
#endif

  for (int i = 0; i < size; i++) {
    if (d_data[0][i] < 1) {
      printf("[Error] sync\n");
      break;
    }
  }

  cudaFree(d_data);

  return 0;
}
