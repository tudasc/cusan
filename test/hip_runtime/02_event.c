// clang-format off
// RUN: %wrapper-hip %clang_args -x hip %s -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s --check-prefixes=%sync-check

// RUN: %wrapper-hip %clang_args -x hip -DCUSAN_SYNC %s -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// REQUIRES: hip

// clang-format on

// CHECK-DAG: data race
// SYNC-ERROR-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync

#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void kernel(int* data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (unsigned int x = 0; x < 545912; x++) {
    data[tid] += x * data[tid];
  }
  data[tid] = (tid + 1);
}

int main() {
  const int size            = 256;
  const int threadsPerBlock = 256;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int* d_data;  // Unified Memory pointer

  // Allocate Unified Memory
  hipMallocManaged(&d_data, size * sizeof(int));
  hipMemset(d_data, 0, size * sizeof(int));

  hipEvent_t endEvent;
  hipEventCreate(&endEvent);
  kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data);
  hipEventRecord(endEvent);

#ifdef CUSAN_SYNC
  // Wait for the end event to complete (alternative to querying)
  hipEventSynchronize(endEvent);
#endif

  for (int i = 0; i < size; i++) {
    if (d_data[i] < 1) {
      printf("[Error] sync\n");
      break;
    }
  }

  hipEventDestroy(endEvent);
  hipFree(d_data);

  return 0;
}
