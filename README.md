# CuSan  &middot; [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

CuSan is tool to find data races between (asynchronous) CUDA calls and the host.

To that end, during compilation with Clang/LLVM, we analyze and instrument CUDA API usage in the target code to track CUDA-specific memory accesses and synchronization semantics.
Our runtime then exposes these information to [ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html) (packaged with Clang/LLVM) for the final data race analysis.


## Usage

Making use of CuSan consists of two phases:

1. Compile your code using one the CuSan compiler wrappers, e.g., `cusan-clang++` or `cusan-mpic++`.
This will (a) analyze and instrument the CUDA API, such as kernel calls and their particular memory access semantics (r/w), (b) add ThreadSanitizer instrumentation automatically (`-fsanitize=thread`), and (c) finally link our runtime library.
2. Execute the target program for the data race analysis. Our runtime internally calls ThreadSanitizer to expose the CUDA synchronization and memory access semantics. 

#### Example usage
Given the file [02_event.c](test/runtime/02_event.c), execute the following for CUDA data race detection:

```bash
$ cusan-clang -O3 -g 02_event.c -x cuda -gencode arch=compute_70,code=sm_70 -o event.exe
$ export TSAN_OPTIONS=ignore_noninstrumented_modules=1
$ ./event.exe
```

### Checking CUDA-aware MPI applications
You need to use the MPI correctness checker [MUST](https://hpc.rwth-aachen.de/must/), or preload our (very) simple MPI interceptor `libCusanMPIInterceptor.so` for CUDA-aware MPI data race detection.
These libraries call ThreadSanitizer with the particular access semantics of MPI. 
Therefore, the combined semantics of CUDA and MPI are properly exposed to ThreadSanitizer to detect data races of data dependent MPI and CUDA calls.

#### Example usage for MPI
Given the file [03_cuda_to_mpi.c](test/runtime/03_cuda_to_mpi.c), execute the following for CUDA data race detection:

```bash
$ cusan-mpic++ -O3 -g 03_cuda_to_mpi.c -x cuda -gencode arch=compute_70,code=sm_70 -o cuda_to_mpi.exe
$ LD_PRELOAD=/path/to/libCusanMPIInterceptor.so mpirun -n 2 ./cuda_to_mpi.exe
```

*Note*: For avoiding false positives, ThreadSanitizer suppression files might be needed, see for example [suppression.txt](test/runtime/suppressions.txt), or documentation for [sanitizer special case lists](https://clang.llvm.org/docs/SanitizerSpecialCaseList.html).

#### Example report
The following is an example report for [03_cuda_to_mpi.c](test/runtime/03_cuda_to_mpi.c) of our test suite, where the necessary synchronization is not called:
```c
L.18  __global__ void kernel(int* arr, const int N)
...
L.53  int* d_data;
L.54  cudaMalloc(&d_data, size * sizeof(int));
L.55
L.56  if (world_rank == 0) {
L.57    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
L.58  #ifdef CUSAN_SYNC
L.59    cudaDeviceSynchronize();  // CUSAN_SYNC needs to be defined
L.60  #endif
L.61    MPI_Send(d_data, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
```
```
==================
WARNING: ThreadSanitizer: data race (pid=579145)
  Read of size 8 at 0x7f1587200000 by main thread:
    #0 main cusan/test/runtime/03_cuda_to_mpi.c:61:5 (03_cuda_to_mpi.c.exe+0xfad11)

  Previous write of size 8 at 0x7f1587200000 by thread T6:
    #0 __device_stub__kernel(int*, int) cusan/test/runtime/03_cuda_to_mpi.c:18:47 (03_cuda_to_mpi.c.exe+0xfaaed)

  Thread T6 'cuda_stream 0' (tid=0, running) created by main thread at:
    #0 cusan::runtime::Runtime::register_stream(cusan::runtime::Stream) <null> (libCusanRuntime.so+0x3b830)
    #1 main cusan/test/runtime/03_cuda_to_mpi.c:54:3 (03_cuda_to_mpi.c.exe+0xfabc7)

SUMMARY: ThreadSanitizer: data race cusan/test/runtime/03_cuda_to_mpi.c:61:5 in main
==================
ThreadSanitizer: reported 1 warnings
```

#### Caveats ThreadSanitizer and OpenMPI
Known issues (on the Lichtenberg HPC system) to make ThreadSanitizer work with OpenMPI 4.1.6:
- Intel Compute Runtime requires environment flags to work with sanitizers, see [Intel Compute Runtime issue 376](https://github.com/intel/compute-runtime/issues/376):
  ```bash
  export NEOReadDebugKeys=1
  export DisableDeepBind=1
  ```
- The sanitizer memory interceptor does not play well with OpenMPI's, see [OpenMPI issue 12819](https://github.com/open-mpi/ompi/issues/12819). Need to disable *patcher*:
  ```bash
  export OMPI_MCA_memory=^patcher
  ```

## Building CuSan

CuSan is tested with LLVM version 14 and 18, and CMake version >= 3.20. Use CMake presets `develop` or `release`
to build.

### Dependencies
CuSan was tested on the TUDa Lichtenberg II cluster with:
- System modules: `1) gcc/11.2.0 2) cuda/11.8 3) openmpi/4.1.6 4) git/2.40.0 5) python/3.10.10 6) clang/14.0.6 or 6) clang/18.1.8`
- Optional external libraries: [TypeART](https://github.com/tudasc/TypeART/tree/v1.9.0b-cuda.1), FiberPool (both default off)
- Testing: llvm-lit, FileCheck
- GPU: Tesla T4 and Tesla V100 (mostly: arch=sm_70)

### Build example

CuSan uses CMake to build. Example build recipe (release build, installs to default prefix
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
| `CUSAN_LOG_LEVEL_PASS`         |  `3`    | Granularity of pass plugin logger. 3 is most verbose, 0 is least. For release, set to 0.              |
