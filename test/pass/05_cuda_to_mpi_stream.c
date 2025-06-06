// clang-format off
// RUN: %rm-file %t.yaml 
// RUN: %wrapper-mpicc %clang-pass-only-args --cusan-kernel-data=%t.yaml -x cuda --cuda-gpu-arch=sm_72 %s 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR

// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamCreate
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_create_stream

// CHECK-LLVM-IR: cudaMemcpyAsync
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_memcpy_async
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamSynchronize
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_sync_stream

// REQUIRES: mpi

// clang-format on

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
    kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, size);
#ifdef CUSAN_SYNC
    cudaStreamSynchronize(stream);  // FIXME: uncomment for correct execution
#endif
    MPI_Send(d_data, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else if (world_rank == 1) {
    MPI_Recv(d_data, size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (world_rank == 1) {
    int* h_data = (int*)malloc(size * sizeof(int));
    cudaMemcpyAsync(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (int i = 0; i < size; i++) {
      const int buf_v = h_data[i];
      if (buf_v == 0) {
        printf("[Error] sync\n");
        break;
      }
      //      printf("buf[%d] = %d (r%d)\n", i, buf_v, world_rank);
    }
    free(h_data);
  }

  cudaStreamDestroy(stream);
  cudaFree(d_data);
  MPI_Finalize();
  return 0;
}
