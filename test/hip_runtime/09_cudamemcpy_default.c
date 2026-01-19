// clang-format off
// RUN: %wrapper-hip %clang_args -x hip %s -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck --allow-empty %s
// clang-format on

// CHECK-NOT: data race
// CHECK-NOT: [Error] sync

#include <hip/hip_runtime.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
  const int size = 512;
  int* h_data    = (int*)malloc(size * sizeof(int));
  hipHostRegister(h_data, size * sizeof(int), hipHostRegisterDefault);
  int* h_data2;
  hipHostAlloc(&h_data2, size * sizeof(int), hipHostAllocDefault);

  memset(h_data, 0, size * sizeof(int));
  hipMemcpy(h_data, h_data, size * sizeof(int), hipMemcpyDefault);
  for (int i = 0; i < size; i++) {
    const int buf_v = h_data[i];
    if (buf_v != 0) {
      printf("[Error] sync\n");
      break;
    }
  }
  hipHostUnregister(h_data);
  hipFreeHost(h_data2);

  free(h_data);
  return 0;
}
