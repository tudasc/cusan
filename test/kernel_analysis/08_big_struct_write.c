// clang-format off
// RUN: %rm-file %t.yaml 
// RUN: %wrapper-cc %clang-pass-only-args --cusan-kernel-data=%t.yaml -x cuda --cuda-gpu-arch=sm_72 %s 2>&1 | %filecheck %s

// CHECK-NOT: Handling Arg:
// CHECK: Handling Arg:
// CHECK-NEXT: subarg: {{.*}}gep_indices:[], ptr: 1, rw: Read
// CHECK-NEXT: subarg: {{.*}}, is_loading, gep_indices:[0, 0, ], ptr: 1, rw: Write
// CHECK-NEXT: Handling Arg:
// CHECK-NEXT: subarg: {{.*}}ptr: 0, rw: ReadWrite

// CHECK: Handling Arg:
// CHECK-NEXT: subarg: {{.*}}gep_indices:[], ptr: 1, rw: Read
// CHECK-NEXT: subarg: {{.*}}, is_loading, gep_indices:[0, 1, ], ptr: 1, rw: Write
// CHECK-NEXT: Handling Arg:
// CHECK-NEXT: subarg: {{.*}}ptr: 0, rw: ReadWrite

// CHECK: Handling Arg:
// CHECK-NEXT: subarg: {{.*}}gep_indices:[], ptr: 1, rw: Read
// CHECK-NEXT: subarg: {{.*}}, is_loading, gep_indices:[0, 1, ], ptr: 1, rw: Write
// CHECK-NEXT: Handling Arg:
// CHECK-NEXT: subarg: {{.*}}ptr: 0, rw: ReadWrite
// CHECK-NOT: Handling Arg:

// clang-format on

#include "../support/gpu_mpi.h"

struct BufferStorage {
  int* buff1;
  int* buff2;
  int a;
  int b;
  float c;
};

__global__ void kernel1(BufferStorage storage, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    storage.buff1[tid] = tid * 32;
  }
}
__global__ void kernel2(BufferStorage storage, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    storage.buff2[tid] = tid * 32;
  }
}
__global__ void kernel3(BufferStorage storage, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    storage.buff2[tid] = tid * 32;
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

  BufferStorage buffStor;
  cudaMalloc(&buffStor.buff1, size * sizeof(int));
  cudaMalloc(&buffStor.buff2, size * sizeof(int));

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  if (world_rank == 0) {
    kernel1<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(buffStor, size);
    kernel3<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(
        buffStor, size);  // no problem since kernel 1 and 3 write to different
    kernel2<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(buffStor,
                                                            size);  // also no problem since they on same stream
#ifdef CUSAN_SYNC
    cudaDeviceSynchronize();
#endif
    // MPI_Send(buffStor.buff1, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
    MPI_Send(buffStor.buff2, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else if (world_rank == 1) {
    // MPI_Recv(buffStor.buff1, size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(buffStor.buff2, size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    kernel3<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
        buffStor, size);  // problem since different stream but same write target
  }

  cudaFree(buffStor.buff1);
  cudaFree(buffStor.buff2);
  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream1);
  MPI_Finalize();
  return 0;
}
