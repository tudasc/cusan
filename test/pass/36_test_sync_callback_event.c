// clang-format off
// RUN: %rm-file %t.yaml 
// RUN: %wrapper-cc %clang-pass-only-args --cusan-kernel-data=%t.yaml -x cuda --cuda-gpu-arch=sm_72 %s 2>&1 | %filecheck %s
// REQUIRES: sync_callback
// clang-format on

// CHECK: {{(invoke|call)}} i32 @cudaEventSynchronize
// CHECK: {{(invoke|call)}} void @cusan_sync_callback

int main(int argc, char* argv[]) {
  cudaEvent_t event;
  cudaEventSynchronize(event);
  return 0;
}
