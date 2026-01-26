# CuSan  &middot; [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


CuSan \[[CU24](#ref-cusan-2024)\] is a tool for detecting data races between (asynchronous) CUDA calls and the host.

To achieve this, we analyze and instrument CUDA API usage in the target code during compilation with Clang/LLVM to track CUDA-specific memory accesses and synchronization semantics.
Our runtime then exposes this information to [ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html) (packaged with Clang/LLVM) for final data race analysis.


## Usage

Using CuSan involves two main steps:

1. **Compile your code** with one of the CuSan compiler wrappers, su ch as `cusan-clang++` or `cusan-mpic++`. This process:
   - Analyzes and instruments the CUDA API, including kernel calls and specific memory access semantics (r/w).
   - Automatically adds ThreadSanitizer instrumentation (`-fsanitize=thread`).
   - Links the CuSan runtime library.
2. **Execute the target program** for data race analysis. Our runtime calls ThreadSanitizer to expose CUDA synchronization and memory access semantics.

##### Compilation limitations
Currently, the compilation must be serialized, e.g., `make -j 1`, to ensure consistent kernel memory access information.
Our analysis writes its kernel-specific data into a specific `.yaml` file during device side compilation (`env CUSAN_KERNEL_DATA_FILE` or wrapper argument `--cusan-kernel-data=`).
This file is subsequently read during the host side compilation.


#### Example usage
Given the file [02_event.c](test/runtime/02_event.c), to detect CUDA data races, execute the following:

```bash
# Set explicit location of kernel memory access data file
$ export CUSAN_KERNEL_DATA_FILE=kernel-data.yaml
# Compile code with CuSan
$ cusan-clang -O3 -g -x cuda -gencode arch=compute_70,code=sm_70 02_event.c -o event.exe
$ export TSAN_OPTIONS=ignore_noninstrumented_modules=1
$ ./event.exe
```

### Checking CUDA-aware MPI applications
To check CUDA-aware MPI applications, use the MPI correctness checker [MUST](https://hpc.rwth-aachen.de/must/) or preload our MPI interceptor `libCusanMPIInterceptor.so`. 
The latter has very limited capabilities and is used mostly for internal testing.
These libraries call ThreadSanitizer with MPI-specific access semantics, ensuring that combined CUDA and MPI semantics are properly exposed to ThreadSanitizer for data race detection between dependent MPI and CUDA calls.

#### Example usage for MPI
Given the file [03_cuda_to_mpi.c](test/runtime/03_cuda_to_mpi.c), execute the following:

```bash
$ cusan-mpic++ -O3 -g -x cuda -gencode arch=compute_70,code=sm_70  03_cuda_to_mpi.c -o cuda_to_mpi.exe
$ LD_PRELOAD=/path/to/libCusanMPIInterceptor.so mpirun -n 2 ./cuda_to_mpi.exe
```

*Note*: To avoid false positives, you may need ThreadSanitizer suppression files.
See [suppression.txt](test/runtime/suppressions.txt), or refer to the [sanitizer special case lists documentation](https://clang.llvm.org/docs/SanitizerSpecialCaseList.html).


#### Example report
The following is an example report for [03_cuda_to_mpi.c](test/runtime/03_cuda_to_mpi.c) of our test suite, where the necessary synchronization is missing:
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
For the Lichtenberg HPC system, some issues may arise when using ThreadSanitizer with OpenMPI 4.1.6:
- Intel Compute Runtime requires specific environment flags, see [Intel Compute Runtime issue 376](https://github.com/intel/compute-runtime/issues/376):
  ```bash
  export NEOReadDebugKeys=1
  export DisableDeepBind=1
  ```
- OpenMPI's memory interceptor may conflict with the sanitizer's., see [OpenMPI issue 12819](https://github.com/open-mpi/ompi/issues/12819). Need to disable *patcher*:
  ```bash
  export OMPI_MCA_memory=^patcher
  ```

### Using CuSan with CMake
For plain Makefiles, the wrapper replaces the Clang compiler variables, e.g., `CC` or `MPICC`. For CMake, during the configuration, it is advised to disable the wrapper temporarily. This is due to CMake executing internal compiler checks, where we do not need CuSan instrumentation:

```bash
# Temporarily disable wrapper with environment flag CUSAN_WRAPPER=OFF:
$> CUSAN_WRAPPER=OFF cmake -B build -DCMAKE_C_COMPILER=cusan-clang 
# Compile with cusan-clang:
$> cmake --build build --target install -- -j1
```

## Building CuSan

CuSan is tested with LLVM version 14, 18 and 19, and CMake version >= 3.20. Use CMake presets `develop` or `release`
to build.

### Dependencies
CuSan was tested on the TUDa Lichtenberg II cluster with:
- System modules: `1) gcc/11.2.0 2) cuda/11.8 3) openmpi/4.1.6 4) git/2.40.0 5) python/3.10.10 6) clang/14.0.6 or 6) clang/18.1.8`
- The MPI dependency is optional
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

| Option                        | Default | Description                                                                                                        |
|-------------------------------|:-------:|--------------------------------------------------------------------------------------------------------------------|
| `CUSAN_TYPEART`               | `OFF`   | Use TypeART library to track memory allocations.                                                                   |
| `CUSAN_FIBERPOOL`             | `OFF`   | Use external library to efficiently manage fibers creation .                                                       |
| `CUSAN_SOFTCOUNTER`           | `OFF`   | Runtime stats for calls to ThreadSanitizer and CUDA-callbacks. Only use for stats collection, not race detection.  |
| `CUSAN_DEVICE_SYNC_CALLBACKS` | `OFF`   | Adds a callback after each CUDA sync call (device, stream, event) to our runtime including the calls return value. |
| `CUSAN_SYNC_DETAIL_LEVEL`     | `ON`    | Analyze, e.g., memcpy and memcpyasync w.r.t. arguments to determine implicit sync.                                 |
| `CUSAN_LOG_LEVEL_RT`          | `0`     | Granularity of runtime logger. 3 is most verbose, 0 is least. For release, set to 0.                               |
| `CUSAN_LOG_LEVEL_PASS`        | `3`     | Granularity of pass plugin logger. 3 is most verbose, 0 is least. For release, set to 0.                           |

### Development 

For debugging, additional (hidden) build options and environment flags exists.


#### Build options
| Option                       | Default | Description                                                                                       |
|------------------------------|:-------:|---------------------------------------------------------------------------------------------------|
| `CUSAN_TEST_WORKAROUNDS`              |  `ON`  | Will set environment flags as described in **Caveats ThreadSanitizer and OpenMPI** for testing.                                      |

#### Environment flags

| Environment Flag                       | Default | Description                                                                                       |
|------------------------------|:-------:|---------------------------------------------------------------------------------------------------|
| `CUSAN_DUMP_HOST_IR`              |  -  | Dumps module IR of host side during compilation to stdout after our transformations. Unsupported with TypeART.                                      |
| `CUSAN_DUMP_DEVICE_IR`              |  -  | Dumps module IR of device during compilation to stdout after our analysis. This includes the applied transformation *mem2reg*. Note: Device analysis happens before host. Unsupported with TypeART.                                     |

## References

<table style="border:0px">
<tr>
    <td valign="top"><a name="ref-cusan-2024"></a>[CU24]</td>
    <td>Hück, Alexander and Ziegler, Tim and Schwitanski, Simon and Jenke, Joachim and Bischof, Christian,
    "Compiler-Aided Correctness Checking of CUDA-Aware MPI Applications",
    In <i>SC24-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis</i>,
    pages 204-213. IEEE, 2024, doi: <a href=https://doi.org/10.1109/SCW63240.2024.00032>10.1109/SCW63240.2024.00032</a></td>
</tr>
</table>