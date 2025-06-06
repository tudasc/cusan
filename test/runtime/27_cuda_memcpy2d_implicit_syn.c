// clang-format off

// RUN: %wrapper-cxx %clang_args -x cuda %s -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %wrapper-cxx -DCUSAN_SYNC %clang_args -x cuda %s -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC
// clang-format on

// CHECK-DAG: data race
// CHECK-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync

// REQUIRES: !typeart

#include <assert.h>
#include <stdio.h>

__global__ void kernel(int* arr, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
#if __CUDA_ARCH__ >= 700
    for (int i = 0; i < tid; i++) {
      __nanosleep(99000000U);
    }
#else
    printf(">>> __CUDA_ARCH__ !\n");
#endif
    arr[tid] = (tid + 1);
  }
}

int main(int argc, char* argv[]) {
  const int width  = 64;
  const int height = 8;

  int* d_data;
  size_t pitch;
  // allocations
  cudaMallocPitch(&d_data, &pitch, width * sizeof(int), height);
  int* h_data       = (int*)malloc(width * sizeof(int) * height);
  int* dummy_h_data = (int*)malloc(width * sizeof(int) * height);

  size_t true_buffer_size = pitch * height;
  size_t true_n_elements  = true_buffer_size / sizeof(int);
  assert(true_buffer_size % sizeof(int) == 0);
  const int threadsPerBlock = true_n_elements;
  const int blocksPerGrid   = (true_n_elements + threadsPerBlock - 1) / threadsPerBlock;

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cudaStream_t stream2;
  cudaStreamCreate(&stream2);

  // null out all the data
  cudaMemset2D(d_data, pitch, 0, width, height);
  memset(h_data, 0, width * sizeof(int) * height);
  cudaDeviceSynchronize();

  kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, true_n_elements);

#ifdef CUSAN_SYNC
  // copy into dummy data buffer causing implicit sync
  cudaMemcpy2D(dummy_h_data, width * sizeof(int), d_data, pitch, width * sizeof(int), height, cudaMemcpyDeviceToHost);
#endif

  // do async non blocking copy which will fail if there was no sync between this and the writing kernel
  cudaMemcpy2DAsync(h_data, width * sizeof(int), d_data, pitch, width * sizeof(int), height, cudaMemcpyDeviceToHost,
                    stream2);
  cudaStreamSynchronize(stream2);
  for (int i = 0; i < width * height; i++) {
    const int buf_v = h_data[i];
    // printf("buf[%d] = %d\n", i, buf_v);
    if (buf_v == 0) {
      printf("[Error] sync\n");
      break;
    }
  }

  free(h_data);
  free(dummy_h_data);
  cudaFree(d_data);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  return 0;
}
