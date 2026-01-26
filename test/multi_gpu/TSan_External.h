/* Part of the MUST Project, under BSD-3-Clause License
 * See https://hpc.rwth-aachen.de/must/LICENSE for license information.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file TSan_External.h
 * Functions exported from ThreadSaniziter that can be used for dynamic annotations.
 * https://github.com/llvm/llvm-project/blob/main/compiler-rt/lib/tsan/rtl/tsan_interface.cpp
 * https://github.com/llvm/llvm-project/blob/main/compiler-rt/lib/tsan/rtl/tsan_interface_ann.cpp
 */

#ifndef TSAN_EXTERNAL_H
#define TSAN_EXTERNAL_H

typedef unsigned long uptr;

typedef enum { mo_relaxed, mo_consume, mo_acquire, mo_release, mo_acq_rel, mo_seq_cst } morder;

typedef unsigned char a8;
typedef unsigned short a16;
typedef unsigned int a32;
typedef unsigned long long a64;

#ifdef MUST_DEBUG
// Print an error message *once* if an annotation function is used that is not overwritten by the
// TSan runtime
#define FALLBACK_PRINT(func_name)                                                  \
  {                                                                                \
    static bool once = false;                                                      \
    if (!once) {                                                                   \
      printf(                                                                      \
          "[MUST-ERROR] %s fallback called, check your TSan runtime and the call " \
          "signature\n",                                                           \
          func_name);                                                              \
      once = true;                                                                 \
    }                                                                              \
  }
#else
#define FALLBACK_PRINT(func_name)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ThreadSanitizer defines the following functions that can be used in MUST for dynamic annotations.
void __attribute__((weak)) AnnotateHappensAfter(const char* file, int line, const volatile void* cv) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) AnnotateHappensBefore(const char* file, int line, const volatile void* cv) {
  FALLBACK_PRINT(__func__);
}

void __attribute__((weak)) AnnotateNewMemory(const char* file, int line, const volatile void* cv, uptr size) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) AnnotateMemoryRead(const char* file, int line, const volatile void* cv, uptr size) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) AnnotateMemoryWrite(const char* file, int line, const volatile void* cv, uptr size) {
  FALLBACK_PRINT(__func__);
}

void __attribute__((weak)) AnnotateIgnoreReadsBegin(const char* file, int line) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) AnnotateIgnoreReadsEnd(const char* file, int line) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) AnnotateIgnoreWritesBegin(const char* file, int line) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) AnnotateIgnoreWritesEnd(const char* file, int line) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) AnnotateIgnoreSyncBegin(const char* file, int line) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) AnnotateIgnoreSyncEnd(const char* file, int line) {
  FALLBACK_PRINT(__func__);
}

void __attribute__((weak)) AnnotateRWLockCreate(const char* file, int line, const volatile void* cv) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) AnnotateRWLockDestroy(const char* file, int line, const volatile void* cv) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) AnnotateRWLockAcquired(const char* file, int line, const volatile void* cv,
                                                  unsigned long long is_w) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) AnnotateRWLockReleased(const char* file, int line, const volatile void* cv,
                                                  unsigned long long is_w) {
  FALLBACK_PRINT(__func__);
}

void __attribute__((weak)) __tsan_read_range(void* addr, uptr size) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) __tsan_write_range(void* addr, uptr size) {
  FALLBACK_PRINT(__func__);
}

void __attribute__((weak)) __tsan_read1_pc(void* addr, void* pc) {
}
void __attribute__((weak)) __tsan_write1_pc(void* addr, void* pc) {
}
void __attribute__((weak)) __tsan_read2_pc(void* addr, void* pc) {
}
void __attribute__((weak)) __tsan_write2_pc(void* addr, void* pc) {
}
void __attribute__((weak)) __tsan_read4_pc(void* addr, void* pc) {
}
void __attribute__((weak)) __tsan_write4_pc(void* addr, void* pc) {
}
void __attribute__((weak)) __tsan_read8_pc(void* addr, void* pc) {
}
void __attribute__((weak)) __tsan_write8_pc(void* addr, void* pc) {
}

#define annotateHelper(rw, s)                                            \
  if (len >= s) {                                                        \
    printf("annotateHelper(%s, %i, %p, %li)\n", #rw, s, addr, (uptr)pc); \
    len -= s;                                                            \
    __tsan_##rw##s##_pc(addr, pc);                                       \
    addr += s;                                                           \
    size -= s;                                                           \
  }

void __attribute__((weak)) __tsan_read_range_pc(void* a, uptr size, void* pc) {
  char* addr = (char*)a;
  uptr len   = ((uptr)addr) % 8;
  if (size < len)
    len = size;
  annotateHelper(read, 4) annotateHelper(read, 2) annotateHelper(read, 1) for (; size > 7; size -= 8) {
    __tsan_read8_pc(addr, pc);
    addr += 8;
  }
  len = size;
  annotateHelper(read, 4) annotateHelper(read, 2) annotateHelper(read, 1)
}

void __attribute__((weak)) __tsan_write_range_pc(void* a, uptr size, void* pc) {
  char* addr = (char*)a;
  uptr len   = ((uptr)addr) % 8;
  // printf("__tsan_write_range_pc(%p, %li, %li), %li\n", addr, size, pc, len);
  if (size < len)
    len = size;
  annotateHelper(write, 4) annotateHelper(write, 2) annotateHelper(write, 1) for (; size > 7; size -= 8) {
    // printf("__tsan_write8_pc(%p, %li)\n", addr, pc);
    __tsan_write8_pc(addr, pc);
    addr += 8;
  }
  len = size;
  annotateHelper(write, 4) annotateHelper(write, 2) annotateHelper(write, 1)
}

void __attribute__((weak)) __tsan_func_entry(void* call_pc) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) __tsan_func_exit() {
  FALLBACK_PRINT(__func__);
}

a8 __attribute__((weak)) __tsan_atomic8_load(const volatile a8* a, morder mo) {
  FALLBACK_PRINT(__func__);
  return 0;
}
a16 __attribute__((weak)) __tsan_atomic16_load(const volatile a16* a, morder mo) {
  FALLBACK_PRINT(__func__);
  return 0;
}
a32 __attribute__((weak)) __tsan_atomic32_load(const volatile a32* a, morder mo) {
  FALLBACK_PRINT(__func__);
  return 0;
}
a64 __attribute__((weak)) __tsan_atomic64_load(const volatile a64* a, morder mo) {
  FALLBACK_PRINT(__func__);
  return 0;
}
void __attribute__((weak)) __tsan_atomic8_store(volatile a8* a, a8 v, morder mo) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) __tsan_atomic16_store(volatile a16* a, a16 v, morder mo) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) __tsan_atomic32_store(volatile a32* a, a32 v, morder mo) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) __tsan_atomic64_store(volatile a64* a, a64 v, morder mo) {
  FALLBACK_PRINT(__func__);
}

// TLC extension
void __attribute__((weak)) AnnotateInitTLC(const char* file, int line, const volatile void* cv) {
  FALLBACK_PRINT(__func__);
  AnnotateHappensBefore(file, line, cv);
}
void __attribute__((weak)) AnnotateStartTLC(const char* file, int line, const volatile void* cv) {
  FALLBACK_PRINT(__func__);
  AnnotateHappensAfter(file, line, cv);
}

// Fibers
void __attribute__((weak)) * __tsan_get_current_fiber() {
  FALLBACK_PRINT(__func__);
  return nullptr;
}
void __attribute__((weak)) * __tsan_create_fiber(unsigned flags) {
  FALLBACK_PRINT(__func__);
  return nullptr;
}
void __attribute__((weak)) __tsan_destroy_fiber(void* fiber) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) __tsan_switch_to_fiber(void* fiber, unsigned flags) {
  FALLBACK_PRINT(__func__);
}
void __attribute__((weak)) __tsan_set_fiber_name(void* fiber, const char* name) {
  FALLBACK_PRINT(__func__);
}

#ifdef __cplusplus
}
#endif

#define TsanHappensBefore(cv) AnnotateHappensBefore(__FILE__, __LINE__, cv)
#define TsanHappensAfter(cv) AnnotateHappensAfter(__FILE__, __LINE__, cv)

#define TsanIgnoreWritesBegin() AnnotateIgnoreWritesBegin(__FILE__, __LINE__)
#define TsanIgnoreWritesEnd() AnnotateIgnoreWritesEnd(__FILE__, __LINE__)
#define TsanIgnoreReadsBegin() AnnotateIgnoreReadsBegin(__FILE__, __LINE__)
#define TsanIgnoreReadsEnd() AnnotateIgnoreReadsEnd(__FILE__, __LINE__)
#define TsanIgnoreSyncBegin() AnnotateIgnoreSyncBegin(__FILE__, __LINE__)
#define TsanIgnoreSyncEnd() AnnotateIgnoreSyncEnd(__FILE__, __LINE__)

#define TsanInitTLC(cv) AnnotateInitTLC(__FILE__, __LINE__, cv)
#define TsanStartTLC(cv) AnnotateStartTLC(__FILE__, __LINE__, cv)

#define TsanCreateFiber(flags) __tsan_create_fiber(flags)
#define TsanDestroyFiber(fiber) __tsan_destroy_fiber(fiber)
#define TsanSwitchToFiber(fiber, flags) __tsan_switch_to_fiber(fiber, flags)
#define TsanGetCurrentFiber() __tsan_get_current_fiber()
#define TsanSetFiberName(fiber, name) __tsan_set_fiber_name(fiber, name)

#define TsanMemoryRead(addr, size) __tsan_read_range(addr, size)
#define TsanMemoryWrite(addr, size) __tsan_write_range(addr, size)
#define TsanMemoryReadPC(addr, size, pc) __tsan_read_range_pc(addr, size, pc)
#define TsanMemoryWritePC(addr, size, pc) __tsan_write_range_pc(addr, size, pc)

#define TsanFuncEntry(pc) __tsan_func_entry(pc)
#define TsanFuncExit() __tsan_func_exit()

#define TsanPCMemoryRead(pc, addr, size) AnnotatePCMemoryRead(pc, __FILE__, __LINE__, addr, size)
#define TsanPCMemoryWrite(pc, addr, size) AnnotatePCMemoryWrite(pc, __FILE__, __LINE__, addr, size)

#define TsanNewMemory(addr, size) AnnotateNewMemory(__FILE__, __LINE__, addr, size)
#define TsanFreeMemory(addr, size) AnnotateNewMemory(__FILE__, __LINE__, addr, size)

#define TsanRWLockCreate(cv) AnnotateRWLockCreate(__FILE__, __LINE__, cv)
#define TsanRWLockDestroy(cv) AnnotateRWLockDestroy(__FILE__, __LINE__, cv)
#define TsanRWLockAcquired(cv, is_w) AnnotateRWLockAcquired(__FILE__, __LINE__, cv, is_w)
#define TsanRWLockReleased(cv, is_w) AnnotateRWLockReleased(__FILE__, __LINE__, cv, is_w)

#define TsanAtomic8Load(a8, mo) __tsan_atomic8_load(a8, mo)
#define TsanAtomic8Store(a8, v, mo) __tsan_atomic8_store(a8, v, mo)
#define TsanAtomic16Load(a16, mo) __tsan_atomic16_load(a16, mo)
#define TsanAtomic16Store(a16, v, mo) __tsan_atomic16_store(a16, v, mo)
#define TsanAtomic32Load(a32, mo) __tsan_atomic32_load(a32, mo)
#define TsanAtomic32Store(a32, v, mo) __tsan_atomic32_store(a32, v, mo)
#define TsanAtomic64Load(a64, mo) __tsan_atomic64_load(a64, mo)
#define TsanAtomic64Store(a64, v, mo) __tsan_atomic64_store(a64, v, mo)

#endif /*TSAN_EXTERNAL_H*/