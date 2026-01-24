// clang-format off
//  RUN: %wrapper-hip %clang_args -x hip %s -o %cusan_test_dir/%basename_t.exe
//  RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %wrapper-hip -DCUSAN_SYNC %clang_args -x hip %s -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// REQUIRES: hip

// clang-format on

// CHECK-DAG: data race

// CHECK-SYNC-NOT: data race

#include <hip/hip_runtime.h>
#include <stdio.h>

struct BufferStorage {
  int* buff1;
  // a list of pointers
  int** buff2;
};

__global__ void kernel(BufferStorage storage, const int N, bool write_second) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    storage.buff1[tid] = (tid + 1) * 32;
    if (write_second) {
      storage.buff2[0][tid] = tid * 32;
    }
  }
}

int main(int argc, char* argv[]) {
  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  BufferStorage buffStor;
  hipMallocManaged(&buffStor.buff1, size * sizeof(int));

  buffStor.buff2 = NULL;

  kernel<<<blocksPerGrid, threadsPerBlock, 0>>>(buffStor, size, false);
#ifdef CUSAN_SYNC
  hipDeviceSynchronize();
#endif

  for (int i = 0; i < size; i++) {
    if (buffStor.buff1[i] < 1) {
      printf("[Error] sync\n");
      break;
    }
  }

  hipFree(buffStor.buff1);
  if (buffStor.buff2 != NULL) {
    hipFree(buffStor.buff2);
  }
  return 0;
}
