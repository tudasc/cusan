// clang-format off
// RUN: %wrapper-hip %clang_args -x hip %s -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s --check-prefixes=%sync-check

// RUN: %wrapper-hip -DCUSAN_SYNC %clang_args -x hip %s -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

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
  int* managed_data;
  int* managed_data2;
  int* fake_data;
  int* d_data2;
  hipStream_t stream1;
  hipStream_t stream2;
  hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking);
  hipStreamCreate(&stream2);

  hipMallocManaged(&managed_data, size * sizeof(int));
  hipMallocManaged(&managed_data2, size * sizeof(int));
  hipMallocManaged(&fake_data, 4);
  hipMemset(managed_data, 0, size * sizeof(int));
  hipMemset(managed_data2, 0, size * sizeof(int));

  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(managed_data, size, 545912);
  hipMemset(fake_data, 0, 4);
#ifdef CUSAN_SYNC
  hipStreamSynchronize(stream1);
#endif
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(managed_data2, size, 1);
  hipStreamSynchronize(stream2);
  for (int i = 0; i < size; i++) {
    const int data_i = managed_data[i];
    if (data_i == 0) {
      printf("[Error] sync\n");
      break;
    }
  }

  hipFree(d_data2);
  hipFree(managed_data);

  return 0;
}
