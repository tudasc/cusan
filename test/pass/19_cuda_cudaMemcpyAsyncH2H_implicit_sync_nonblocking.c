// clang-format off

// RUN: %apply %s -strip-debug --cusan-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR

// clang-format on

// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamCreateWithFlags
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_create_stream
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamCreateWithFlags
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_create_stream
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaMemset
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_memset
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaDeviceSynchronize
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_sync_device
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaMemcpy
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_memcpy
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaMemcpyAsync
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_memcpy_async
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamSynchronize
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_sync_stream
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaFree
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_device_free
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamDestroy
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamDestroy

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
  int* data;
  // int* data2;
  int* d_data2;
  int* h_data  = (int*)malloc(sizeof(int));
  int* h_data2 = (int*)malloc(sizeof(int));

  int* h_data3 = (int*)malloc(size * sizeof(int));
  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

  cudaMalloc(&data, size * sizeof(int));
  cudaMemset(data, 0, size * sizeof(int));

  cudaDeviceSynchronize();

  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(data, size, 1316134912);
  cudaMemcpy(h_data, h_data2, sizeof(int), cudaMemcpyHostToHost);
#ifdef CUSAN_SYNC
  cudaStreamSynchronize(stream1);
#endif
  cudaMemcpyAsync(h_data3, data, size * sizeof(int), cudaMemcpyDefault, stream2);
  cudaStreamSynchronize(stream2);
  for (int i = 0; i < size; i++) {
    if (h_data3[i] == 0) {
      printf("[Error] sync %i\n", h_data3[i]);
      break;
    }
  }

  cudaFree(data);

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

  return 0;
}
