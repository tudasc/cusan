// clang-format off

// RUN: %apply %s -strip-debug --cusan-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR


// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamCreate 
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_create_stream 
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamCreate 
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_create_stream
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaMemcpyAsync({{i8\*|ptr}} {{.*}}[[mcpyasy_target:%[0-9a-z]+]], {{i8\*|ptr}} {{.*}}[[mcpyasy_from:%[0-9a-z]+]],
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_memcpy_async({{i8\*|ptr}} {{.*}}[[mcpyasy_target]], {{i8\*|ptr}} {{.*}}[[mcpyasy_from]], 
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamSynchronize
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamDestroy 
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamDestroy

// clang-format on

#include "../support/gpu_mpi.h"

#include <unistd.h>

__global__ void kernel(int* arr, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
#if __CUDA_ARCH__ >= 700
    for (int i = 0; i < tid; i++) {
      __nanosleep(1000000U);
    }
#else
    printf(">>> __CUDA_ARCH__ !\n");
#endif
    arr[tid] = tid + 1;
  }
}

int main(int argc, char* argv[]) {
  if (!has_gpu_aware_mpi()) {
    printf("This example is designed for CUDA-aware MPI. Exiting.\n");
    return 1;
  }

  cudaEvent_t first_finished_event;
  cudaEventCreate(&first_finished_event);
  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int* h_data = (int*)malloc(size * sizeof(int));
  int* d_data;
  cudaMalloc(&d_data, size * sizeof(int));

  kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, size);
#ifdef CUSAN_SYNC
  cudaStreamSynchronize(stream1);
#endif
  cudaMemcpyAsync(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost, stream2);
  cudaStreamSynchronize(stream2);
  for (int i = 0; i < size; i++) {
    const int buf_v = h_data[i];
    if (buf_v == 0) {
      printf("[Error] sync\n");
      break;
    }
  }
  free(h_data);

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaEventDestroy(first_finished_event);
  cudaFree(d_data);
  return 0;
}
