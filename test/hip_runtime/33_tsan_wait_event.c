// clang-format off
// RUN: %wrapper-hip %clang_args -x hip -g %s -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s -DFILENAME=%s

// RUN: %wrapper-hip -DCUSAN_SYNC %clang_args -x hip -g %s -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-SYNC

// CHECK-DAG: data race

// CHECK-SYNC-NOT: data race

// clang-format on

#include <hip/hip_runtime.h>
#include <unistd.h>

__global__ void writing_kernel(float* arr, const int N, float value) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    for (unsigned int x = 0; x < tid; x++) {
      arr[tid] += x * arr[tid];
    }
    arr[tid] = (float)tid + value;
  }
}

__global__ void reading_kernel(float* res, const float* read, const int N,
                               float value) {
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
  hipMalloc(&res_data, size * sizeof(float));
  hipMalloc(&d_data, size * sizeof(float));

  // Copy host memory to device
  hipMemcpy(d_data, h_data, size * sizeof(float), hipMemcpyHostToDevice);

  hipDeviceSynchronize();
  // Create hip streams
  hipStream_t stream1, stream2;
  hipStreamCreate(&stream1);
  hipStreamCreate(&stream2);
  // Create an event
  hipEvent_t event;
  hipEventCreate(&event);
  // Launch first kernel in stream1
  writing_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, size, 5.0f);

  // Record event after kernel in stream1
  hipEventRecord(event, stream1);
  // Make stream2 wait for the event
#ifdef CUSAN_SYNC
  hipStreamWaitEvent(stream2, event, 0);
#endif

  // Launch second kernel in stream2
  reading_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(res_data, d_data, size, 10.0f);

  // Copy data back to host
  hipMemcpy(h_data, d_data, size * sizeof(float), hipMemcpyDeviceToHost);

  // Wait for stream2 to finish
  hipStreamSynchronize(stream2);

  hipStreamDestroy(stream2);
  hipStreamDestroy(stream1);
  hipEventDestroy(event);
  hipFree(d_data);
  free(h_data);
  return 0;
}
