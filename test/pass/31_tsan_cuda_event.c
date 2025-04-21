// clang-format off
// RUN: %rm-file %t.yaml 

// RUN: %wrapper-cc %clang-pass-only-args --cusan-kernel-data=%t.yaml -x cuda --cuda-gpu-arch=sm_72 %s 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR

// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaEventCreate
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_create_event
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamCreate
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_create_stream
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamCreate
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_create_stream
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaEventRecord
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_event_record

// clang-format on

#include <unistd.h>

__global__ void kernel(int* arr, const int N) {  // CHECK-DAG: [[FILENAME]]:[[@LINE]]
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    arr[tid] = arr[tid] + 1;
  }
}

int main(int argc, char* argv[]) {
  cudaEvent_t first_finished_event;
  cudaEventCreate(&first_finished_event);
  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int* d_data;
  cudaMalloc(&d_data, size * sizeof(int));

  kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, size);
  cudaEventRecord(first_finished_event, stream1);

#ifdef CUSAN_SYNC
  cudaEventSynchronize(first_finished_event);
#endif

  kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_data, size);

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaEventDestroy(first_finished_event);
  cudaFree(d_data);
  return 0;
}
