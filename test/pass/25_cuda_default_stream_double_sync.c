// clang-format off

// RUN: %wrapper-cc %emit-host-only --cusan-kernel-data=%t.yaml -x cuda --cuda-gpu-arch=sm_72 %s 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR

// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaStreamCreate
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_create_stream
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaStreamCreate
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_create_stream
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaMemset
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_memset
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaDeviceSynchronize
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_sync_device
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaFree
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_device_free
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaFree
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_device_free

// clang-format on

#include "../support/gpu_mpi.h"

#include <unistd.h>

__global__ void write_kernel_delay(int* arr, const int N, int value, const unsigned int delay) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
#if __CUDA_ARCH__ >= 700
  for (int i = 0; i < tid; i++) {
    __nanosleep(delay);
  }
#else
  printf(">>> __CUDA_ARCH__ !\n");
#endif
  if (tid < N) {
    arr[tid] = value;
  }
}

int main(int argc, char* argv[]) {
  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int* managed_data;
  cudaMallocManaged(&managed_data, size * sizeof(int));
  cudaMemset(managed_data, 0, size * sizeof(int));

  int* d_data2;
  cudaMalloc(&d_data2, size * sizeof(int));
  cudaDeviceSynchronize();

  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(managed_data, size, 128, 9999999);
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, 0>>>(d_data2, size, 0, 1);
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_data2, size, 128, 1);
#ifdef CUSAN_SYNC
  cudaStreamSynchronize(stream2);
#endif
  for (int i = 0; i < size; i++) {
    if (managed_data[i] == 0) {
      printf("[Error] sync %i %i\n", managed_data[i], i);
      break;
    }
  }

  cudaFree(managed_data);
  cudaFree(d_data2);
  return 0;
}
