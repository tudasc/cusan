// cusan library
// Copyright (c) 2023-2024 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#include "StatsCounter.h"
#include "TSanInterface.h"
#include "support/Table.h"

#include <iostream>
#include <mpi.h>
#include <threads.h>

#ifndef _EXTERN_C_
#ifdef __cplusplus
#define _EXTERN_C_ extern "C"
#else /* __cplusplus */
#define _EXTERN_C_
#endif /* __cplusplus */
#endif /* _EXTERN_C_ */

namespace cusan::mpi::runtime {

using namespace cusan::runtime::softcounter;

#define CUSAN_MPI_EVENTS        \
  cusan_stat_handle(Send);      \
  cusan_stat_handle(Isend);     \
  cusan_stat_handle(Recv);      \
  cusan_stat_handle(Irecv);     \
  cusan_stat_handle(Wait);      \
  cusan_stat_handle(Waitall);   \
  cusan_stat_handle(SendRecv);  \
  cusan_stat_handle(Reduce);    \
  cusan_stat_handle(AllReduce); \
  cusan_stat_handle(Barrier);

#define cusan_stat_handle(name) \
  inline void inc_##name() {    \
  }                             \
  inline Counter get_##name() { \
    return 0;                   \
  }
class MPINoneRecorder final {
 public:
#include "TsanEvents.inc"
  CUSAN_MPI_EVENTS
};

#undef cusan_stat_handle
#define cusan_stat_handle(name) \
  AtomicCounter name = 0;       \
  inline void inc_##name() {    \
    this->name++;               \
  }                             \
  inline Counter get_##name() { \
    return this->name;          \
  }

struct MPIAccessRecorder final {
 public:
#include "TsanEvents.inc"
  CUSAN_MPI_EVENTS
};

#ifdef CUSAN_SOFTCOUNTER
using MPIRecorder = MPIAccessRecorder;
#else
using MPIRecorder = MPINoneRecorder;
#endif

struct MPIRuntime final {
  MPIRecorder mpi_recorder;
  static MPIRuntime& get() {
    static MPIRuntime run_t;
    return run_t;
  }

  ~MPIRuntime() {
#undef cusan_stat_handle
#define cusan_stat_handle(name) table.put(Row::make(#name, mpi_recorder.get_##name()));
#if CUSAN_SOFTCOUNTER
    Table table{"Cusan MPI runtime statistics"};
#include "TsanEvents.inc"
    CUSAN_MPI_EVENTS
    table.print(std::cout);
#endif
#undef cusan_stat_handle
#undef CUSAN_MPI_EVENTS
  }

 private:
  MPIRuntime() = default;
};
}  // namespace cusan::mpi::runtime

using namespace cusan::mpi::runtime;

_EXTERN_C_ int PMPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);

_EXTERN_C_ int MPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
  int _wrap_py_return_val = PMPI_Send(buf, count, datatype, dest, tag, comm);

  int type_size;
  MPI_Type_size(datatype, &type_size);
  TsanMemoryReadPC(buf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));

  auto& rec = MPIRuntime::get().mpi_recorder;
  rec.inc_Send();
  rec.inc_TsanMemoryRead();

  return _wrap_py_return_val;
}

_EXTERN_C_ int PMPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
                         MPI_Status* status);
_EXTERN_C_ int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
                        MPI_Status* status) {
  int _wrap_py_return_val = PMPI_Recv(buf, count, datatype, source, tag, comm, status);

  int type_size;
  MPI_Type_size(datatype, &type_size);
  TsanMemoryWritePC(buf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));

  auto& rec = MPIRuntime::get().mpi_recorder;
  rec.inc_Recv();
  rec.inc_TsanMemoryWrite();

  return _wrap_py_return_val;
}

thread_local void* async_fiber = nullptr;

_EXTERN_C_ int PMPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
                          MPI_Request* request);
_EXTERN_C_ int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
                         MPI_Request* request) {
  int _wrap_py_return_val = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);

  int type_size;
  MPI_Type_size(datatype, &type_size);
  auto& rec = MPIRuntime::get().mpi_recorder;
  {
    if (!async_fiber) {
      async_fiber = TsanCreateFiber(0);
      rec.inc_TsanCreateFiber();
    }
    void* old_fiber = TsanGetCurrentFiber();
    TsanSwitchToFiber(async_fiber, 0);
    TsanMemoryWritePC(buf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));
    rec.inc_TsanSwitchToFiber();
    rec.inc_TsanMemoryWrite();
    if (request) {
      TsanHappensBefore(request);
      rec.inc_TsanHappensBefore();
    }
    TsanSwitchToFiber(old_fiber, 1);
    rec.inc_TsanSwitchToFiber();
  }

  rec.inc_Irecv();

  return _wrap_py_return_val;
}

_EXTERN_C_ int PMPI_Isend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
                          MPI_Request* request);
_EXTERN_C_ int MPI_Isend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
                         MPI_Request* request) {
  int _wrap_py_return_val = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);

  int type_size;
  MPI_Type_size(datatype, &type_size);

  auto& rec = MPIRuntime::get().mpi_recorder;
  {
    if (!async_fiber) {
      async_fiber = TsanCreateFiber(0);
      rec.inc_TsanCreateFiber();
    }
    void* old_fiber = TsanGetCurrentFiber();
    TsanSwitchToFiber(async_fiber, 0);
    rec.inc_TsanSwitchToFiber();
    TsanMemoryReadPC(buf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));
    rec.inc_TsanMemoryRead();
    if (request) {
      TsanHappensBefore(request);
      rec.inc_TsanHappensBefore();
    }
    TsanSwitchToFiber(old_fiber, 1);
    rec.inc_TsanSwitchToFiber();
  }

  rec.inc_Isend();

  return _wrap_py_return_val;
}

_EXTERN_C_ int PMPI_Wait(MPI_Request* request, MPI_Status* status);
_EXTERN_C_ int MPI_Wait(MPI_Request* request, MPI_Status* status) {
  int _wrap_py_return_val = PMPI_Wait(request, status);
  auto& rec               = MPIRuntime::get().mpi_recorder;
  if (request) {
    TsanHappensAfter(request);
    rec.inc_TsanHappensAfter();
  }
  rec.inc_Wait();
  return _wrap_py_return_val;
}

_EXTERN_C_ int PMPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status* array_of_statuses);
_EXTERN_C_ int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status* array_of_statuses) {
  int _wrap_py_return_val = PMPI_Waitall(count, array_of_requests, array_of_statuses);
  auto& rec               = MPIRuntime::get().mpi_recorder;
  for (int i = 0; i < count; ++i) {
    if (&(array_of_requests[i])) {
      TsanHappensAfter(&(array_of_requests[i]));
      rec.inc_TsanHappensAfter();
    }
  }
  rec.inc_Waitall();

  return _wrap_py_return_val;
}

_EXTERN_C_ int PMPI_Sendrecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                             void* recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
                             MPI_Comm comm, MPI_Status* status);
_EXTERN_C_ int MPI_Sendrecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                            void* recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
                            MPI_Status* status) {
  int _wrap_py_return_val = PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype,
                                          source, recvtag, comm, status);

  auto& rec = MPIRuntime::get().mpi_recorder;
  {
    int send_type_size;
    MPI_Type_size(sendtype, &send_type_size);

    TsanMemoryReadPC(sendbuf, static_cast<uptr>(sendcount) * send_type_size, __builtin_return_address(0));
    rec.inc_TsanMemoryRead();

    int recv_type_size;
    MPI_Type_size(recvtype, &recv_type_size);
    TsanMemoryWritePC(recvbuf, static_cast<uptr>(recvcount) * recv_type_size, __builtin_return_address(0));
    rec.inc_TsanMemoryWrite();
  }
  rec.inc_SendRecv();
  return _wrap_py_return_val;
}

_EXTERN_C_ int PMPI_Barrier(MPI_Comm comm);
_EXTERN_C_ int MPI_Barrier(MPI_Comm comm) {
  int _wrap_py_return_val = PMPI_Barrier(comm);
  auto& rec               = MPIRuntime::get().mpi_recorder;
  if (!async_fiber) {
    async_fiber = TsanCreateFiber(0);
    rec.inc_TsanCreateFiber();
  }
  void* old_fiber = TsanGetCurrentFiber();
  TsanSwitchToFiber(async_fiber, 0);
  TsanSwitchToFiber(old_fiber, 0);
  rec.inc_TsanSwitchToFiber();
  rec.inc_TsanSwitchToFiber();
  rec.inc_Barrier();
  return _wrap_py_return_val;
}

_EXTERN_C_ int PMPI_Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root,
                           MPI_Comm comm);
_EXTERN_C_ int MPI_Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root,
                          MPI_Comm comm) {
  int _wrap_py_return_val = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
  auto& rec               = MPIRuntime::get().mpi_recorder;
  {
    int type_size;
    MPI_Type_size(datatype, &type_size);

    TsanMemoryReadPC(sendbuf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));
    TsanMemoryWritePC(recvbuf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));
    rec.inc_TsanMemoryRead();
    rec.inc_TsanMemoryWrite();
  }
  rec.inc_Reduce();
  return _wrap_py_return_val;
}

_EXTERN_C_ int PMPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                              MPI_Comm comm);
_EXTERN_C_ int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                             MPI_Comm comm) {
  int _wrap_py_return_val = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  auto& rec               = MPIRuntime::get().mpi_recorder;
  {
    int type_size;
    MPI_Type_size(datatype, &type_size);

    TsanMemoryReadPC(sendbuf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));
    TsanMemoryWritePC(recvbuf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));
    rec.inc_TsanMemoryRead();
    rec.inc_TsanMemoryWrite();
  }
  rec.inc_AllReduce();
  return _wrap_py_return_val;
}