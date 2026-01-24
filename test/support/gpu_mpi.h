#ifndef CUSAN_GPUAWAREMPI_H
#define CUSAN_GPUAWAREMPI_H

#include <mpi.h>
#include <mpi-ext.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

inline bool has_cuda_aware_mpi() {
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (1 == MPIX_Query_cuda_support()) {
    return true;
  }
#endif
  return false;
}

inline bool has_rocm_aware_mpi() {
#if defined(MPIX_HIP_AWARE_SUPPORT)
  if (1 == MPIX_Query_hip_support()) {
    return true;
  }
#endif
  return false;
}

inline void print_cuda_aware_mpi() {
  printf("CUDA Compile time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  printf("This MPI library has CUDA-aware support.\n");
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
  printf("This MPI library does not have CUDA-aware support.\n");
#else
  printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif

  printf("CUDA Run time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (has_cuda_aware_mpi()) {
    printf("This MPI library has CUDA-aware support.\n");
  } else {
    printf("This MPI library does not have CUDA-aware support.\n");
  }
#else
  printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif
}

inline void print_rocm_aware_mpi() {
  printf("HIP Compile time check:\n");
#if defined(MPIX_HIP_AWARE_SUPPORT) && MPIX_HIP_AWARE_SUPPORT
  printf("This MPI library has HIP-aware support.\n");
#elif defined(MPIX_HIP_AWARE_SUPPORT) && !MPIX_HIP_AWARE_SUPPORT
  printf("This MPI library does not have HIP-aware support.\n");
#else
  printf("This MPI library cannot determine if there is HIP-aware support.\n");
#endif

  printf("HIP Run time check:\n");
#if defined(MPIX_HIP_AWARE_SUPPORT)
  if (has_rocm_aware_mpi()) {
    printf("This MPI library has HIP-aware support.\n");
  } else {
    printf("This MPI library does not have HIP-aware support.\n");
  }
#else
  printf("This MPI library cannot determine if there is HIP-aware support.\n");
#endif
}

inline void print_gpu_aware_mpi() {
  print_cuda_aware_mpi();
  print_rocm_aware_mpi();
}

inline bool has_gpu_aware_mpi() {
  if (has_cuda_aware_mpi()) {
    return true;
  }
  if (has_rocm_aware_mpi()) {
    return true;
  }
  return false;
}

#ifdef __cplusplus
}
#endif

#endif
