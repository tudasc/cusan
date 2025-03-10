// clang-format off
// RUN: %rm-file %t.yaml 

// RUN: %wrapper-cc %clang-pass-only-args --cusan-kernel-data=%t.yaml -x cuda --cuda-gpu-arch=sm_72 %s 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR

// clang-format on

// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamCreate
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_create_stream
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaEventCreateWithFlags
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_create_event
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaMemset
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_memset
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaEventRecord
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_event_record
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaFree
// CHECK-LLVM-IR: {{(call|invoke)}} void @_cusan_device_free
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
  int* managed_data;
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cudaEvent_t event1;
  cudaEventCreateWithFlags(&event1, cudaEventBlockingSync);

  cudaMallocManaged(&managed_data, size * sizeof(int));
  cudaMemset(managed_data, 0, size * sizeof(int));

  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(managed_data, size, 1316134912);
  cudaEventRecord(event1, stream1);

#ifdef CUSAN_SYNC
  while (cudaEventQuery(event1) != cudaSuccess) {
  }
#endif

  for (int i = 0; i < size; i++) {
    if (managed_data[i] == 0) {
      printf("[Error] sync %i\n", managed_data[i]);
      // break;
    }
  }

  cudaFree(managed_data);
  cudaStreamDestroy(stream1);
  return 0;
}
