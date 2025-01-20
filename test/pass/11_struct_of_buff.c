// clang-format off
// RUN: %rm-file %t.yaml 
// RUN: %wrapper-cc %clang-pass-only-args --cusan-kernel-data=%t.yaml -x cuda --cuda-gpu-arch=sm_72 %s 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR

// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaStreamCreate
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_create_stream
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaStreamCreate
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_create_stream
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaDeviceSynchronize
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_sync_device
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaStreamDestroy
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaStreamDestroy
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaFree
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_device_free
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaFree
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_device_free

// clang-format on

#include "../support/gpu_mpi.h"

struct BufferStorage {
  int* buff1;
  int* buff2;
};

__global__ void kernel1(BufferStorage storage, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    storage.buff1[tid] = tid * 32;
  }
}
__global__ void kernel2(BufferStorage storage, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    storage.buff2[tid] = tid * 32;
  }
}
__global__ void kernel3(BufferStorage storage, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    storage.buff2[tid] = tid * 32;
  }
}

int main(int argc, char* argv[]) {
  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  BufferStorage buffStor;
  cudaMalloc(&buffStor.buff1, size * sizeof(int));
  cudaMalloc(&buffStor.buff2, size * sizeof(int));

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  kernel1<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(buffStor, size);
  kernel3<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(buffStor,
                                                          size);  // no problem since kernel 1 and 3 write to different
  kernel2<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(buffStor, size);  // also no problem since they on same stream
#ifdef CUSAN_SYNC
  cudaDeviceSynchronize();
#endif
  kernel3<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
      buffStor, size);  // problem since different stream but same write target

  cudaDeviceSynchronize();

  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream1);
  cudaFree(buffStor.buff1);
  cudaFree(buffStor.buff2);
  return 0;
}
