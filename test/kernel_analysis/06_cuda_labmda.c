// RUN: %apply  %s -strip-debug --cusan-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 2>&1 | \
// RUN: %filecheck %s

// CHECK-NOT: Handling Arg:
// CHECK: Handling Arg:
// CHECK-NEXT: subarg: {{.*}}indices:[], ptr: 1, rw: ReadWrite
// CHECK-NEXT: subarg: {{.*}}indices:{{.}}[0, 0, ], L, ], ptr: 1, rw: Write
// CHECK-NOT: Handling Arg:

#include <cstdio>
#include <cuda_runtime.h>

template <typename F>
__global__ void kernel_functor(F functor) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
#if __CUDA_ARCH__ >= 700
  for (int i = 0; i < tid; i++) {
    __nanosleep(1000000U);
  }
#else
  printf(">>> __CUDA_ARCH__ !\n");
#endif
  functor(tid);
}

int main() {
  const int size            = 256;
  const int threadsPerBlock = 256;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int* d_data;  // Unified Memory pointer

  // Allocate Unified Memory
  cudaMallocManaged(&d_data, size * sizeof(int));
  cudaMemset(d_data, 0, size * sizeof(int));
  cudaDeviceSynchronize();
  cudaEvent_t endEvent;
  cudaEventCreate(&endEvent);
  const auto lamba_kernel = [=] __host__ __device__(const int tid) { d_data[tid] = (tid + 1); };
  kernel_functor<decltype(lamba_kernel)><<<blocksPerGrid, threadsPerBlock>>>(lamba_kernel);
  cudaEventRecord(endEvent);

#ifdef CUSAN_SYNC
  // Wait for the end event to complete (alternative to querying)
  cudaEventSynchronize(endEvent);
#endif

  for (int i = 0; i < size; i++) {
    if (d_data[i] < 1) {
      printf("[Error] sync\n");
      break;
    }
  }

  cudaEventDestroy(endEvent);
  cudaFree(d_data);

  return 0;
}
