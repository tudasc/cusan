# CuSan  &middot; [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

CuSan is tool to find data races between (asynchronous) CUDA calls and the host.
To that end, we analyze and instrument CUDA codes to track CUDA domain-specific memory accesses and synchronization semantics during compilation using LLVM.
Our runtime then passes these information appropriately to [ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html) (packaged with Clang/LLVM) for the final data race analysis.


## Usage

Making use of CuSan consists of two phases:

1. Compile your code with Clang/LLVM (version 14) using one the CuSan compiler wrappers, e.g., `cusan-clang++` or `cusan-mpic++`.
This will (a) analyze and instrument the CUDA API appropriately, such as kernel calls and their particular memory access semantics (r/w), (b) add ThreadSanitizer instrumentation automatically (`-fsanitize=thread`), and (c) finally link our runtime library.
2. Execute the target program for the data race analysis. Our runtime internally calls ThreadSanitizer to expose the CUDA synchronization and memory access semantics. 


### Checking CUDA-aware MPI applications
You need to use the MPI correctness checker [MUST](https://hpc.rwth-aachen.de/must/), or preload our (very) simple MPI interceptor `libCusanMPIInterceptor.so` for CUDA-aware MPI data race detection.
These libraries call ThreadSanitizer with the particular access semantics of MPI. 
Therefore, the combined semantics of CUDA and MPI are properly exposed to ThreadSanitizer to detect data races of data dependent MPI and CUDA calls.

#### Example usage for MPI
Given the file [03_cuda_to_mpi.c](test/runtime/03_cuda_to_mpi.c), execute the following for CUDA data race detection:

```bash
$ cusan-mpic++ -O3 03_cuda_to_mpi.c -x cuda -gencode arch=compute_70,code=sm_70 -o cuda_to_mpi.exe
$ LD_PRELOAD=/path/to/libCusanMPIInterceptor.so mpirun -n 2 ./cuda_to_mpi.exe
```

#### Example report
The following is an example report for [03_cuda_to_mpi.c](test/runtime/03_cuda_to_mpi.c) of our test suite, where the necessary synchronization is not called:
```c
L.23  __global__ void kernel(int* arr, const int N)
...
L.58  int* d_data;
L.59  cudaMalloc(&d_data, size * sizeof(int));
L.60
L.61  if (world_rank == 0) {
L.62    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
L.63  #ifdef CUSAN_SYNC
L.64    cudaDeviceSynchronize();  // CUSAN_SYNC needs to be defined
L.65  #endif
L.66    MPI_Send(d_data, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
```
```
==================
WARNING: ThreadSanitizer: data race (pid=689288)
  Read of size 8 at 0x7fb51f200000 by main thread:
    #0 main cusan/test/pass/03_cuda_to_mpi.c:66:5 (03_cuda_to_mpi.c.exe+0x4e8448)

  Previous write of size 8 at 0x7fb51f200000 by thread T6:
    #0 __device_stub__kernel(int*, int) cusan/test/pass/03_cuda_to_mpi.c:23:47 (03_cuda_to_mpi.c.exe+0x4e81ef)

  Thread T6 'cuda_stream' (tid=0, running) created by main thread at:
    #0 __pool_create_fiber_dbg cusan/build/_deps/fiber_pool-src/fiberpool.cpp:538:16 (libCusanFiberpool-d.so+0x1c152)
    #1 main cusan/test/pass/03_cuda_to_mpi.c:59:3 (03_cuda_to_mpi.c.exe+0x4e8331)

SUMMARY: ThreadSanitizer: data race cusan/test/pass/03_cuda_to_mpi.c:66:5 in main
==================
ThreadSanitizer: reported 1 warnings
```

## Building cusan

cusan is tested with LLVM version 14 and 18, and CMake version >= 3.20. Use CMake presets `develop` or `release`
to build.

### Dependencies
CuSan was tested on the TUDa Lichtenberg II cluster with:
- System modules: `1) gcc/11.2.0 2) cuda/11.8 3) openmpi/4.1.6 4) git/2.40.0 5) python/3.10.10 6) clang/14.0.6 or 6) clang/18.1.8`
- Optional external libraries: [TypeART](https://github.com/tudasc/TypeART/tree/v1.9.0b-cuda.1), FiberPool (both default off)
- Testing: llvm-lit, FileCheck
- GPU: Tesla T4 and Tesla V100 (mostly: arch=sm_70)

### Build example

cusan uses CMake to build. Example build recipe (release build, installs to default prefix
`${cusan_SOURCE_DIR}/install/cusan`)

```sh
$> cd cusan
$> cmake --preset release
$> cmake --build build --target install --parallel
```

#### Build options

| Option                       | Default | Description                                                                                       |
|------------------------------|:-------:|---------------------------------------------------------------------------------------------------|
| `CUSAN_TYPEART`              |  `OFF`  | Use TypeART library to track memory allocations.                                      |
| `CUSAN_FIBERPOOL`            |  `OFF`  | Use external library to efficiently manage fibers creation .                                      |
| `CUSAN_SOFTCOUNTER`          |  `OFF`  | Runtime stats for calls to ThreadSanitizer and CUDA-callbacks. Only use for stats collection, not race detection.   |
| `CUSAN_SYNC_DETAIL_LEVEL`    |  `ON`   | Analyze, e.g., memcpy and memcpyasync w.r.t. arguments to determine implicit sync.                |
| `CUSAN_LOG_LEVEL_RT`         |  `3`    | Granularity of runtime logger. 3 is most verbose, 0 is least. For release, set to 0.              |
