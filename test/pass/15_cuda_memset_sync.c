// clang-format off
// RUN: %wrapper-cc %emit-host-only --cusan-kernel-data=%t.yaml -x cuda --cuda-gpu-arch=sm_72 %s 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR


// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamCreate
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_create_stream
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamCreate
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_create_stream
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaMemset
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_memset
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaMemset
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_memset
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamSynchronize
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_sync_stream
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaFree
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_device_free
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaFree
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_device_free

// clang-format on

#include <cstdio>
#include <cuda_runtime.h>

__global__ void write_kernel_delay(int* arr, const int N, const unsigned int delay) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
#if __CUDA_ARCH__ >= 700
  for (int i = 0; i < tid; i++) {
    __nanosleep(delay);
  }
#else
  printf(">>> __CUDA_ARCH__ !\n");
#endif
  if (tid < N) {
    arr[tid] = (tid + 1);
  }
}

int main() {
  const int size            = 256;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;
  int* managed_data;
  int* managed_data2;
  int* fake_data;
  int* d_data2;
  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  cudaMallocManaged(&managed_data, size * sizeof(int));
  cudaMallocManaged(&managed_data2, size * sizeof(int));
  cudaMallocManaged(&fake_data, 4);
  cudaMemset(managed_data, 0, size * sizeof(int));
  cudaMemset(managed_data2, 0, size * sizeof(int));

  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(managed_data, size, 1316134912);
#ifdef CUSAN_SYNC
  cudaMemset(fake_data, 0, 4);
#endif
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(managed_data2, size, 1);
  cudaStreamSynchronize(stream2);
  for (int i = 0; i < size; i++) {
    if (managed_data[i] == 0) {
      printf("[Error] sync %i\n", managed_data[i]);
      break;
    }
  }

  cudaFree(d_data2);
  cudaFree(managed_data);

  return 0;
}
