// clang-format off
// RUN: %wrapper-cxx %clang_args -x cuda -g %s -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s -DFILENAME=%s

// RUN: %wrapper-cxx -DCUSAN_SYNC %clang_args -x cuda -g %s -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options%cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-SYNC

// clang-format on

// CHECK-DAG: data race

// CHECK-SYNC-NOT: data race

#include "../support/gpu_mpi.h"

#include <unistd.h>

__global__ void kernel(int* arr, const int N) {  // CHECK-DAG: [[FILENAME]]:[[@LINE]]
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    arr[tid] = arr[tid] + 1;
  }
}

int main(int argc, char* argv[]) {
  cudaEvent_t first_finished_event;
  cudaEventCreate(&first_finished_event);
  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int* d_data;
  cudaMalloc(&d_data, size * sizeof(int));

  kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, size);
  cudaEventRecord(first_finished_event, stream1);

#ifdef CUSAN_SYNC
  cudaEventSynchronize(first_finished_event);
#endif

  kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_data, size);

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaEventDestroy(first_finished_event);
  cudaFree(d_data);
  return 0;
}
