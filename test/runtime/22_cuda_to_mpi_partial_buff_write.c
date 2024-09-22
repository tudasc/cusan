// clang-format off
// RUN: %wrapper-mpicxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t.exe
// RUN: %cusan_ldpreload %tsan-options %mpi-exec -n 2 %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s --allow-empty

// clang-format on

// CHECK-NOT: data race
// CHECK-NOT: [Error] sync


#include "../support/gpu_mpi.h"

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

  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;
  static_assert(size % 2 == 0, "Needs to be divisble by 2");
  const int half_size = size / 2;

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

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  if (world_rank == 0) {
    kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, half_size);
    cudaStreamSynchronize(stream);
    kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(&d_data[half_size], half_size);
    MPI_Send(d_data, half_size, MPI_INT, 1, 0, MPI_COMM_WORLD);
    cudaStreamSynchronize(stream);
    MPI_Send(&d_data[half_size], half_size, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else if (world_rank == 1) {
    MPI_Recv(d_data, half_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&d_data[half_size], half_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int* h_data = (int*)malloc(size * sizeof(int));
    cudaMemcpyAsync(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (int i = 0; i < size; i++) {
      const int buf_v = h_data[i];
      if (buf_v == 0) {
        printf("[Error] sync %i\n", i);
        break;
      }
    }
    free(h_data);
  }

  cudaStreamDestroy(stream);
  cudaFree(d_data);
  MPI_Finalize();
  return 0;
}