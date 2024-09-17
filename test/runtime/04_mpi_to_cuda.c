// clang-format off
// RUN: %wrapper-mpicxx %tsan-compile-flags -O1 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t.exe
// RUN: %cusan_ldpreload %tsan-options %mpi-exec -n 2 %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %wrapper-mpicxx %tsan-compile-flags -DCUSAN_SYNC -O1 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %cusan_ldpreload %tsan-options %mpi-exec -n 2 %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// clang-format on

// CHECK-DAG: data race
// CHECK-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync

// FLAKYPASS: *
// ALLOW_RETRIES: 5

#include "../support/gpu_mpi.h"

__global__ void kernel_init(int* arr, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    arr[tid] = -(tid + 1);
  }
}

__global__ void kernel(int* arr, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
#if __CUDA_ARCH__ >= 700
    for (int i = 0; i < tid; i++) {
      __nanosleep(100U);
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

  const int size            = 256;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  MPI_Init(&argc, &argv);
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_size != 2) {
    printf("This example is designed for 2 MPI processes. Exiting.\n");
    MPI_Finalize();
    return 1;
  }

  int* d_data;
  cudaMalloc(&d_data, size * sizeof(int));

  if (world_rank == 0) {
    kernel_init<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 0) {
    MPI_Send(d_data, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else if (world_rank == 1) {
    MPI_Request request;
    // Recv all negative numbers:
    MPI_Irecv(d_data, size, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
#ifdef CUSAN_SYNC
    MPI_Wait(&request, MPI_STATUS_IGNORE);
#endif
    // FIXME: MPI_Wait here to avoid racy d_data access
    // Set all numbers to positive value:
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
#ifndef CUSAN_SYNC
    MPI_Wait(&request, MPI_STATUS_IGNORE);
#endif
  }

  if (world_rank == 1) {
    int* h_data = (int*)malloc(size * sizeof(int));
    // cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
      const int buf_v = h_data[i];
      // Expect: all values should be positive, given the p_1 kernel sets them to tid.
      if (buf_v < 1) {
        printf("[Error] sync\n");
        break;
      }
      //      printf("buf[%d] = %d (r%d)\n", i, buf_v, world_rank);
    }
    free(h_data);
  }

  cudaFree(d_data);
  MPI_Finalize();
  return 0;
}
