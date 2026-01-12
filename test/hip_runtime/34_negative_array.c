// clang-format off
// RUN: %wrapper-hip %clang_args0 -x hip %s -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %wrapper-hip -DCUSAN_SYNC %clang_args0 -x hip %s -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// clang-format on

// CHECK-DAG: data race
// CHECK-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync

// XFAIL:*

#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void kernel(int** data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (unsigned int x = 0; x < tid; x++) {
    data[-1][tid] += x * data[-1][tid];
  }
  data[-1][tid] = (tid + 1);
}

int main() {
  const int size            = 256;
  const int threadsPerBlock = 256;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int** d_data;  // Unified Memory pointer
  hipMallocManaged(&d_data, 2 * sizeof(int*));

  // Allocate Unified Memory
  hipMallocManaged(&d_data[0], size * sizeof(int));
  hipMallocManaged(&d_data[1], size * sizeof(int));
  hipMemset(d_data[0], 0, size * sizeof(int));
  hipMemset(d_data[1], 0, size * sizeof(int));

  kernel<<<blocksPerGrid, threadsPerBlock>>>(&d_data[1]);

#ifdef CUSAN_SYNC
  hipDeviceSynchronize();
#endif

  for (int i = 0; i < size; i++) {
    if (d_data[0][i] < 1) {
      printf("[Error] sync\n");
      break;
    }
  }

  hipFree(d_data);

  return 0;
}
