// clang-format off
// RUN: %wrapper-cxx %clang_args -x cuda -g %s -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s -DFILENAME=%s

// RUN: %wrapper-cxx -DCUSAN_SYNC %clang_args -x cuda -g %s -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-SYNC

// CHECK-DAG: data race

// CHECK-SYNC-NOT: data race

// clang-format on

// #include "../support/gpu_mpi.h"

#include <unistd.h>

__global__ void writing_kernel(float* arr, const int N, float value) {  // CHECK-DAG: [[FILENAME]]:[[@LINE]]
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
#if __CUDA_ARCH__ >= 700
    for (int i = 0; i < tid; i++) {
      __nanosleep(1000000U);
    }
#else
    printf(">>> __CUDA_ARCH__ !\n");
#endif
    arr[tid] = (float)tid + value;
  }
}

__global__ void reading_kernel(float* res, const float* read, const int N,
                               float value) {  // CHECK-DAG: [[FILENAME]]:[[@LINE]]
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    res[tid] = read[tid] + value;
  }
}

int main(int argc, char* argv[]) {
  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  float* h_data = (float*)malloc(size * sizeof(float));
  memset(h_data, 0, size * sizeof(float));
  // Allocate device memory
  float* d_data;
  float* res_data;
  cudaMalloc(&res_data, size * sizeof(float));
  cudaMalloc(&d_data, size * sizeof(float));

  // Copy host memory to device
  cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  // Create CUDA streams
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  // Create an event
  cudaEvent_t event;
  cudaEventCreate(&event);
  // Launch first kernel in stream1
  writing_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, size, 5.0f);

  // Record event after kernel in stream1
  cudaEventRecord(event, stream1);
  // Make stream2 wait for the event
#ifdef CUSAN_SYNC
  cudaStreamWaitEvent(stream2, event, 0);
#endif

  // Launch second kernel in stream2
  reading_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(res_data, d_data, size, 10.0f);

  // Copy data back to host
  cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

  // Wait for stream2 to finish
  cudaStreamSynchronize(stream2);

  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream1);
  cudaEventDestroy(event);
  cudaFree(d_data);
  free(h_data);
  return 0;
}
