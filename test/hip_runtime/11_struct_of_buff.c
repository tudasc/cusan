// clang-format off
// RUN: %wrapper-hip %clang_args -x hip %s -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %wrapper-hip -DCUSAN_SYNC %clang_args -x hip %s -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC
// clang-format on

// CHECK-DAG: data race

// CHECK-SYNC-NOT: data race

#include <hip/hip_runtime.h>

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
  hipMalloc(&buffStor.buff1, size * sizeof(int));
  hipMalloc(&buffStor.buff2, size * sizeof(int));

  hipStream_t stream1, stream2;
  hipStreamCreate(&stream1);
  hipStreamCreate(&stream2);

  kernel1<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(buffStor, size);
  kernel3<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(buffStor,
                                                          size);  // no problem since kernel 1 and 3 write to different
  kernel2<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(buffStor, size);  // also no problem since they on same stream
#ifdef CUSAN_SYNC
  hipDeviceSynchronize();
#endif
  kernel3<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
      buffStor, size);  // problem since different stream but same write target

  hipDeviceSynchronize();

  hipStreamDestroy(stream2);
  hipStreamDestroy(stream1);
  hipFree(buffStor.buff1);
  hipFree(buffStor.buff2);
  return 0;
}
