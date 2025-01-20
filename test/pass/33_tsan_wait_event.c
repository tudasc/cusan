// clang-format off
// RUN: %rm-file %t.yaml 
// RUN: %wrapper-cc %clang-pass-only-args --cusan-kernel-data=%t.yaml -x cuda --cuda-gpu-arch=sm_72 %s 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR


// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaMemcpy({{i8\*|ptr}} {{.*}}[[mcpy_target:%[0-9a-z]+]], {{i8\*|ptr}} {{.*}}[[mcpy_from:%[0-9a-z]+]],
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_memcpy({{i8\*|ptr}} {{.*}}[[mcpy_target]], {{i8\*|ptr}} {{.*}}[[mcpy_from]],
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamCreate
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_create_stream 
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamCreate
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_create_stream 
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaEventCreate
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_create_event
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaEventRecord
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_event_record
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaMemcpy({{i8\*|ptr}} {{.*}}[[mcpy2_target:%[0-9a-z]+]], {{i8\*|ptr}} {{.*}}[[mcpy2_from:%[0-9a-z]+]],
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_memcpy({{i8\*|ptr}} {{.*}}[[mcpy2_target]], {{i8\*|ptr}} {{.*}}[[mcpy2_from]],
// CHECK-LLVM-IR: {{(call|invoke)}} i32 @cudaStreamSynchronize
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_sync_stream

// clang-format on

#include <unistd.h>

__global__ void writing_kernel(float* arr, const int N, float value) {  // CHECK-DAG: [[FILENAME]]:[[@LINE]]
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
#if __CUDA_ARCH__ >= 700
    for (int i = 0; i < tid; i++) {
      __nanosleep(1000000U);
    }
#else
    printf(">>> __CUDA_ARCH__ !\n");
#endif
    arr[tid] = (float)tid + value;
  }
}

__global__ void reading_kernel(float* res, const float* read, const int N,
                               float value) {  // CHECK-DAG: [[FILENAME]]:[[@LINE]]
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    res[tid] = read[tid] + value;
  }
}

int main(int argc, char* argv[]) {
  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  float* h_data = (float*)malloc(size * sizeof(float));
  memset(h_data, 0, size * sizeof(float));
  // Allocate device memory
  float* d_data;
  float* res_data;
  cudaMalloc(&res_data, size * sizeof(float));
  cudaMalloc(&d_data, size * sizeof(float));

  // Copy host memory to device
  cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  // Create CUDA streams
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  // Create an event
  cudaEvent_t event;
  cudaEventCreate(&event);
  // Launch first kernel in stream1
  writing_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, size, 5.0f);

  // Record event after kernel in stream1
  cudaEventRecord(event, stream1);
  // Make stream2 wait for the event
#ifdef CUSAN_SYNC
  cudaStreamWaitEvent(stream2, event, 0);
#endif

  // Launch second kernel in stream2
  reading_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(res_data, d_data, size, 10.0f);

  // Copy data back to host
  cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

  // Wait for stream2 to finish
  cudaStreamSynchronize(stream2);

  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream1);
  cudaEventDestroy(event);
  cudaFree(d_data);
  free(h_data);
  return 0;
}
