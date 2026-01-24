// clang-format off

// RUN: %wrapper-hip %clang_args -x hip %s -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %wrapper-hip -DCUSAN_SYNC %clang_args -x hip %s -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// REQUIRES: hip && !typeart

// clang-format on

// CHECK-DAG: data race

// CHECK-SYNC-NOT: data race

// REQUIRES:

#include <assert.h>
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void kernel(int* arr, const int N, const unsigned int delay) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    for (unsigned int x = 0; x < delay; x++) {
      arr[tid] += x * arr[tid];
    }
    arr[tid] = (tid + 1);
  }
}

int main(int argc, char* argv[]) {
  const int width  = 64;
  const int height = 8;

  int* d_data;
  size_t pitch;
  // allocations
  hipMallocPitch(&d_data, &pitch, width * sizeof(int), height);
  int* h_data       = (int*)malloc(width * sizeof(int) * height);
  int* dummy_h_data = (int*)malloc(width * sizeof(int) * height);

  size_t true_buffer_size = pitch * height;
  size_t true_n_elements  = true_buffer_size / sizeof(int);
  assert(true_buffer_size % sizeof(int) == 0);
  const int threadsPerBlock = true_n_elements;
  const int blocksPerGrid   = (true_n_elements + threadsPerBlock - 1) / threadsPerBlock;

  hipStream_t stream1;
  hipStreamCreate(&stream1);
  hipStream_t stream2;
  hipStreamCreate(&stream2);

  // null out all the data
  hipMemset2D(d_data, pitch, 0, width, height);
  memset(h_data, 0, width * sizeof(int) * height);
  hipDeviceSynchronize();

  kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, true_n_elements, 2005912);

#ifdef CUSAN_SYNC
  // copy into dummy data buffer causing implicit sync
  hipMemcpy2D(dummy_h_data, width * sizeof(int), d_data, pitch, width * sizeof(int), height, hipMemcpyDeviceToHost);
#endif

  // do async non blocking copy which will fail if there was no sync between this and the writing kernel
  hipMemcpy2DAsync(h_data, width * sizeof(int), d_data, pitch, width * sizeof(int), height, hipMemcpyDeviceToHost,
                   stream2);
  hipStreamSynchronize(stream2);
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
  hipFree(d_data);
  hipStreamDestroy(stream1);
  hipStreamDestroy(stream2);
  return 0;
}
