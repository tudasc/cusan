// clang-format off
// RUN: %wrapper-hip %clang_args -x hip %s -o %cusan_test_dir/%basename_t.exe
// RUN: %cusan_ldpreload %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s --check-prefixes=%sync-check

// RUN: %wrapper-hip -DCUSAN_SYNC %clang_args -x hip %s -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %cusan_ldpreload %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// REQUIRES: hip

// clang-format on

// CHECK-DAG: data race
// SYNC-ERROR-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync

#include <cstdio>
#include <hip/hip_runtime.h>

__global__ void write_kernel_delay(int* arr, const int N, const unsigned int delay) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    for (unsigned int x = 0; x < delay; x++) {
      arr[tid] += x * arr[tid];
    }
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
  hipStream_t stream1;
  hipStream_t stream2;
  hipStreamCreate(&stream1);
  hipStreamCreate(&stream2);

  hipMalloc(&data, size * sizeof(int));
  hipMemset(data, 0, size * sizeof(int));

  hipDeviceSynchronize();

  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(data, size, 545912);
#ifdef CUSAN_SYNC
  hipMemcpy(h_data, h_data2, sizeof(int), hipMemcpyHostToHost);
#endif
  hipMemcpyAsync(h_data3, data, size * sizeof(int), hipMemcpyDefault, stream2);
  hipStreamSynchronize(stream2);
  for (int i = 0; i < size; i++) {
    if (h_data3[i] == 0) {
      printf("[Error] sync %i\n", h_data3[i]);
      break;
    }
  }

  hipFree(data);

  hipStreamDestroy(stream1);
  hipStreamDestroy(stream2);

  return 0;
}
