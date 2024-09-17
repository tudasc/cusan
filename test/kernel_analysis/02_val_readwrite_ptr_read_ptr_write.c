// RUN: %apply  %s -strip-debug --cusan-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 2>&1 | \
// RUN: %filecheck %s

// CHECK-NOT: Handling Arg:
// CHECK: Handling Arg:
// CHECK-NEXT: subarg: {{.*}} ptr: 0, rw: ReadWrite
// CHECK-NEXT: Handling Arg:
// CHECK-NEXT: subarg: {{.*}} ptr: 1, rw: Read
// CHECK-NEXT: Handling Arg:
// CHECK-NEXT: subarg: {{.*}} ptr: 1, rw: ReadWrite
// CHECK-NOT: Handling Arg:

#include <stdio.h>
__device__ void axpy_write(float a, float* y) {
  y[threadIdx.x] = a;
}

__global__ void axpy(float a, float* x, float* y) {
  axpy_write(a * x[threadIdx.x], y);
}

int main(int argc, char* argv[]) {
  const int kDataLen = 4;

  float a                = 2.0f;
  float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
  float host_y[kDataLen];

  float* device_x;
  float* device_y;
  cudaMalloc((void**)&device_x, kDataLen * sizeof(float));
  cudaMalloc((void**)&device_y, kDataLen * sizeof(float));

  cudaMemcpy(device_x, host_x, kDataLen * sizeof(float), cudaMemcpyHostToDevice);

  axpy<<<1, kDataLen>>>(a, device_x, device_y);

  cudaDeviceSynchronize();
  cudaMemcpy(host_y, device_y, kDataLen * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < kDataLen; ++i) {
    printf("y[%i] = %f\n", i, host_y[i]);
  }

  cudaDeviceReset();
  return 0;
}