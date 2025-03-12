// clang-format off
// RUN: %wrapper-mpicxx %clang_args -x cuda %s -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t.exe
// RUN: %cusan_ldpreload %tsan-options %mpi-exec -n 2 %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %wrapper-mpicxx -DCUSAN_SYNC %clang_args -x cuda %s -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %cusan_ldpreload %tsan-options %mpi-exec -n 2 %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// REQUIRES: mpiexec

// clang-format on

// CHECK-DAG: data race
// CHECK-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync

// REQUIRES: !typeart

#include "../support/gpu_mpi.h"

#include <assert.h>

__global__ void kernel(int* arr, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
#if __CUDA_ARCH__ >= 700
    for (int i = 0; i < tid; i++) {
      __nanosleep(1000000U);
    }
#else
    printf(">>> __CUDA_ARCH__ !\n");
#endif
    arr[tid] = (tid + 1);
  }
}

int main(int argc, char* argv[]) {
  if (!has_gpu_aware_mpi()) {
    printf("This example is designed for CUDA-aware MPI. Exiting.\n");
    return 1;
  }

  const int width  = 4;
  const int height = 8;

  int* d_data;
  size_t pitch;
  cudaMallocPitch(&d_data, &pitch, width * sizeof(int), height);

  size_t true_buffer_size = pitch * height;
  size_t true_n_elements  = true_buffer_size / sizeof(int);
  // printf("%zu %zu %zu\n", true_buffer_size, true_n_elements, pitch);
  assert(true_buffer_size % sizeof(int) == 0);

  const int threadsPerBlock = true_n_elements;
  const int blocksPerGrid   = (true_n_elements + threadsPerBlock - 1) / threadsPerBlock;

  MPI_Init(&argc, &argv);
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_size != 2) {
    printf("This example is designed for 2 MPI processes. Exiting.\n");
    MPI_Finalize();
    return 1;
  }

  if (world_rank == 0) {
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, true_n_elements);
#ifdef CUSAN_SYNC
    cudaDeviceSynchronize();  // FIXME: uncomment for correct execution
#endif
    MPI_Send(d_data, true_n_elements, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else if (world_rank == 1) {
    MPI_Recv(d_data, true_n_elements, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (world_rank == 1) {
    int* h_data = (int*)malloc(true_buffer_size);
    cudaMemcpy(h_data, d_data, true_buffer_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < true_n_elements; i++) {
      const int buf_v = h_data[i];
      // printf("buf[%d] = %d (r%d)\n", i, buf_v, world_rank);
      if (buf_v == 0) {
        printf("[Error] sync\n");
        break;
      }
    }
    free(h_data);
  }

  cudaFree(d_data);
  MPI_Finalize();
  return 0;
}
