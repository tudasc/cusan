// clang-format off
// RUN: %apply %s -strip-debug --cusan-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 2>&1 | %filecheck %s  -DFILENAME=%s

// clang-format on

// CHECK-NOT: Handling Arg:
// CHECK: Handling Arg:
// CHECK-NEXT: subarg: {{.*}} ptr: 1, rw: Read
// CHECK-NEXT: subarg: {{.*}}, is_loading, gep_indices:[], ptr: 1, rw: Write
// CHECK-NOT: Handling Arg:

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