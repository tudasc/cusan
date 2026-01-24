// clang-format off
// RUN: %wrapper-hip %clang_args -x hip %s -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s -DFILENAME=%s

// RUN: %wrapper-hip -DCUSAN_SYNC %clang_args -x hip %s -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-SYNC

// REQUIRES: hip

// clang-format on

// CHECK-DAG: data race
// CHECK-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync

#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void write_kernel_delay(int* arr, const int N, int value, const unsigned int delay) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    for (unsigned int x = 0; x < delay; x++) {
      arr[tid] += x * arr[tid];
    }
    arr[tid] = value;
  }
}

int main(int argc, char* argv[]) {
  hipStream_t stream;
  hipStreamCreate(&stream);

  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int* managed_data;
  hipMallocManaged(&managed_data, size * sizeof(int));
  hipMemset(managed_data, 0, size * sizeof(int));

  int* d_data2;
  hipMalloc(&d_data2, size * sizeof(int));
  hipDeviceSynchronize();

  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, 0>>>(managed_data, size, 128, 545912);
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data2, size, 0, 1);

#ifdef CUSAN_SYNC
  hipStreamSynchronize(stream);
#endif
  for (int i = 0; i < size; i++) {
    if (managed_data[i] == 0) {
      printf("[Error] sync %i %i\n", managed_data[i], i);
      break;
    }
  }

  hipStreamDestroy(stream);
  hipFree(managed_data);
  hipFree(d_data2);
  return 0;
}
