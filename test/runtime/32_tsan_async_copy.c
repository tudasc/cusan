// clang-format off
// RUN: %wrapper-mpicxx %tsan-compile-flags -O1 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %mpi-exec -n 2 %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s -DFILENAME=%s

// RUN: %wrapper-mpicxx %tsan-compile-flags -DCUSAN_SYNC -O1 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %mpi-exec -n 2 %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-SYNC

// CHECK-DAG: data race
// CHECK-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync

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
