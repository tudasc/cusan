// clang-format off
// RUN: %wrapper-hip %clang_args -x hip %s -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %wrapper-hip -DCUSAN_SYNC %clang_args -x hip %s -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// REQUIRES: hip

// clang-format on

// CHECK-DAG: data race
// CHECK-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync

#include <cstdio>
#include <hip/hip_runtime.h>

#define HIP_CHECK(expression)                                                                     \
  {                                                                                               \
    const hipError_t status = expression;                                                         \
    if (status != hipSuccess) {                                                                   \
      printf("HIP error %i: %s; %s:%i\n", status, hipGetErrorString(status), __FILE__, __LINE__); \
      exit(1);                                                                                    \
    }                                                                                             \
  }

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
  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;
  int* managed_data;
  hipStream_t stream1;
  HIP_CHECK(hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking));

  HIP_CHECK(hipMallocManaged(&managed_data, size * sizeof(int)));
  HIP_CHECK(hipMemset(managed_data, 0, size * sizeof(int)));
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(managed_data, size, 545912);
#ifdef CUSAN_SYNC
  hipStreamSynchronize(stream1);
#endif
  for (int i = 0; i < size; i++) {
    const int data_i = managed_data[i];
    if (data_i == 0) {
      printf("[Error] sync %i: %i\n", i, data_i);
      break;
    }
  }

  HIP_CHECK(hipFree(managed_data));
  return 0;
}
