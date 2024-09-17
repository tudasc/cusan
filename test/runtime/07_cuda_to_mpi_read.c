// clang-format off
// RUN: %wrapper-mpicxx %clang_args %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t.exe
// RUN: %cusan_ldpreload %tsan-options %mpi-exec -n 2 %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck --allow-empty %s

// clang-format on

// CHECK-NOT: data race
// CHECK-NOT: [Error] sync

#include "../support/gpu_mpi.h"

__global__ void kernel_init(int* arr, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    arr[tid] = -(tid + 1);
  }
}

__global__ void kernel(int* arr, const int N, int* result) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
#if __CUDA_ARCH__ >= 700
    for (int i = 0; i < tid; i++) {
      __nanosleep(10000U);
    }
#else
    printf(">>> __CUDA_ARCH__ !\n");
#endif
    result[tid] = arr[tid];
  }
}

int main(int argc, char* argv[]) {
  if (!has_gpu_aware_mpi()) {
    printf("This example is designed for CUDA-aware MPI. Exiting.\n");
    return 1;
  }

  const int size            = 512;
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
    int* d_result;
    cudaMalloc(&d_result, size * sizeof(int));

    // kernel and Send both only read d_data
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size, d_result);
    MPI_Send(d_data, size, MPI_INT, 1, 0, MPI_COMM_WORLD);

    cudaFree(d_result);
    cudaDeviceSynchronize();
  } else if (world_rank == 1) {
    MPI_Recv(d_data, size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (world_rank == 1) {
    int* h_data = (int*)malloc(size * sizeof(int));
    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
      const int buf_v = h_data[i];
      if (buf_v >= 0) {
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