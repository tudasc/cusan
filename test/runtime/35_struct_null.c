// clang-format off
// RUN: %wrapper-cxx %clang_args %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options timeout 1 %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %wrapper-cxx -DCUSAN_SYNC %clang_args %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options timeout 1 %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC
// clang-format on

// CHECK-DAG: data race

// CHECK-SYNC-NOT: data race

// XFAIL: *

#include "../support/gpu_mpi.h"

struct BufferStorage {
  int* buff1;
  // a list of pointers
  int** buff2;
};

__global__ void kernel(BufferStorage storage, const int N, bool write_second) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    storage.buff1[tid] = tid * 32;
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
  cudaMallocManaged(&buffStor.buff1, size * sizeof(int));
  /// cudaMallocManaged(&buffStor.buff2, sizeof(int*));

  buffStor.buff2 = 0;

  // since we set the bolean argument to false buff2 could contain nullptrs since we dont use it
  //  but the pass analyses based on the static code and so it doesnt know this runtime information
  kernel<<<blocksPerGrid, threadsPerBlock, 0>>>(buffStor, size, false);
#ifdef CUSAN_SYNC
  cudaDeviceSynchronize();
#endif

  for (int i = 0; i < size; i++) {
    if (buffStor.buff1[i] < 1) {
      printf("[Error] sync\n");
      break;
    }
  }

  cudaFree(buffStor.buff1);
  cudaFree(buffStor.buff2);
  return 0;
}
