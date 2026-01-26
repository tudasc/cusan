// clang-format off
// RUN: %wrapper-cc %clang_args -x cuda -gencode arch=compute_70,code=sm_70 %s -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %wrapper-cc -DCUSAN_SYNC %clang_args -x cuda -gencode arch=compute_70,code=sm_70 %s -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// REQUIRES: multigpu

// clang-format on

// CHECK-DAG: data race
// CHECK-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void write_kernel_delay(int* arr, const int N, const unsigned int delay) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
#if __CUDA_ARCH__ >= 700
  for (int i = 0; i < tid; i++) {
    __nanosleep(delay);
  }
#else
  printf(">>> __CUDA_ARCH__ !\n");
#endif
  if (tid < N) {
    arr[tid] = (tid + 1);
  }
}

int main() {
  const int size            = 256;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;
  int* managed_data;
  int* managed_data2;

  int n_devices = 0;
  cudaGetDeviceCount(&n_devices);
  if (n_devices < 2) {
    printf("This test is designed for CUDA on multiple devices but there is only one or none here. Exiting.\n");
    return 1;
  }

  cudaSetDevice(0);
  cudaMallocManaged(&managed_data, size * sizeof(int));
  cudaMemset(managed_data, 0, size * sizeof(int));

  cudaSetDevice(1);
  cudaMallocManaged(&managed_data2, size * sizeof(int));
  cudaMemset(managed_data2, 0, size * sizeof(int));

  cudaSetDevice(0);
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock>>>(managed_data, size, 1316134912);

  // if we only have the later synchronize we will only synchronize the second device
#ifdef CUSAN_SYNC
  cudaDeviceSynchronize();
#endif

  cudaSetDevice(1);
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock>>>(managed_data2, size, 1);
  cudaDeviceSynchronize();

  for (int i = 0; i < size; i++) {
    if (managed_data[i] == 0) {
      printf("[Error] sync managed_data %i\n", managed_data[i]);
      break;
    }
  }
  for (int i = 0; i < size; i++) {
    if (managed_data2[i] == 0) {
      printf("[Error] sync managed_data2 %i\n", managed_data[i]);
      break;
    }
  }

  cudaFree(managed_data);
  cudaFree(managed_data2);
  return 0;
}
