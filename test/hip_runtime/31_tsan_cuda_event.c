// clang-format off
// RUN: %wrapper-hip %clang_args -x hip -g %s -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s -DFILENAME=%s

// RUN: %wrapper-hip -DCUSAN_SYNC %clang_args -x hip -g %s -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options%cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-SYNC

// REQUIRES: hip

// clang-format on

// CHECK-DAG: data race

// CHECK-SYNC-NOT: data race

#include <hip/hip_runtime.h>
#include <unistd.h>

__global__ void kernel(int* arr, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    arr[tid] = arr[tid] + 1;
  }
}

int main(int argc, char* argv[]) {
  hipEvent_t first_finished_event;
  hipEventCreate(&first_finished_event);
  hipStream_t stream1;
  hipStream_t stream2;
  hipStreamCreate(&stream1);
  hipStreamCreate(&stream2);

  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int* d_data;
  hipMalloc(&d_data, size * sizeof(int));

  kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, size);
  hipEventRecord(first_finished_event, stream1);

#ifdef CUSAN_SYNC
  hipEventSynchronize(first_finished_event);
#endif

  kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_data, size);

  hipStreamDestroy(stream1);
  hipStreamDestroy(stream2);
  hipEventDestroy(first_finished_event);
  hipFree(d_data);
  return 0;
}
