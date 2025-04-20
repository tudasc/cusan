include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(FeatureSummary)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE STRING "" FORCE)
include(FetchContent)

find_package(LLVM CONFIG HINTS "${LLVM_DIR}")
if(NOT LLVM_FOUND)
  message(STATUS "LLVM not found at: ${LLVM_DIR}.")
  find_package(LLVM REQUIRED CONFIG)
endif()

set_package_properties(LLVM PROPERTIES
  URL https://llvm.org/
  TYPE REQUIRED
  PURPOSE
  "LLVM framework installation required to compile (and apply) project cusan."
)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}"
  PROJECT_IS_TOP_LEVEL
)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)

option(CUSAN_TEST_CONFIGURE_IDE "Add targets for tests to help the IDE with completion etc." ON)
mark_as_advanced(CUSAN_TEST_CONFIGURE_IDE)
option(CUSAN_CONFIG_DIR_IS_SHARE "Install to \"share/cmake/\" instead of \"lib/cmake/\"" OFF)
mark_as_advanced(CUSAN_CONFIG_DIR_IS_SHARE)

set(CUSAN_LOG_LEVEL_RT 0 CACHE STRING "Granularity of runtime logger. 3 is most verbose, 0 is least.")
set(CUSAN_LOG_LEVEL_PASS 3  CACHE STRING "Granularity of transform pass logger. 3 is most verbose, 0 is least.")

option(CUSAN_TYPEART "Use external typeart to track allocations" OFF)
option(CUSAN_FIBERPOOL "Use external fiber pool to manage ThreadSanitizer fibers" OFF)
option(CUSAN_SOFTCOUNTER "Print runtime counters" OFF)
option(CUSAN_SYNC_DETAIL_LEVEL "Enable implicit sync analysis of memcpy/memset" ON)

option(CUSAN_DEVICE_SYNC_CALLBACKS "Enable runtime callbacks after sync calls" ON)

option(CUSAN_TEST_WORKAROUNDS "Enable workarounds for MPI + TSan regarding runtime tests" ON)
mark_as_advanced(CUSAN_TEST_WORKAROUNDS)


if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CUSAN_LOG_LEVEL_RT 0 CACHE STRING "" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(warning_guard "")
if(NOT PROJECT_IS_TOP_LEVEL)
  option(
      CUSAN_INCLUDES_WITH_SYSTEM
      "Use SYSTEM modifier for cusan includes to disable warnings."
      ON
  )
  mark_as_advanced(CUSAN_INCLUDES_WITH_SYSTEM)

  if(CUSAN_INCLUDES_WITH_SYSTEM)
    set(warning_guard SYSTEM)
  endif()
endif()

include(modules/cusan-llvm)
include(modules/cusan-format)
include(modules/cusan-target-util)

cusan_find_llvm_progs(CUSAN_CLANG_EXEC "clang-${LLVM_VERSION_MAJOR};clang" DEFAULT_EXE "clang")
cusan_find_llvm_progs(CUSAN_CLANGCXX_EXEC "clang++-${LLVM_VERSION_MAJOR};clang++" DEFAULT_EXE "clang++")
cusan_find_llvm_progs(CUSAN_LLC_EXEC "llc-${LLVM_VERSION_MAJOR};llc" DEFAULT_EXE "llc")
cusan_find_llvm_progs(CUSAN_OPT_EXEC "opt-${LLVM_VERSION_MAJOR};opt" DEFAULT_EXE "opt")

if(PROJECT_IS_TOP_LEVEL)
  if(NOT CMAKE_BUILD_TYPE)
    # set default build type
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
    message(STATUS "Building as debug (default)")
  endif()

  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # set default install path
    set(CMAKE_INSTALL_PREFIX
        "${cusan_SOURCE_DIR}/install/cusan"
        CACHE PATH "Default install path" FORCE
    )
    message(STATUS "Installing to (default): ${CMAKE_INSTALL_PREFIX}")
  endif()

    # CUSAN_DEBUG_POSTFIX is only used for Config
    if(CMAKE_DEBUG_POSTFIX)
      set(CUSAN_DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    else()
      set(CUSAN_DEBUG_POSTFIX "-d")
    endif()

  if(NOT CMAKE_DEBUG_POSTFIX AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_DEBUG_POSTFIX ${CUSAN_DEBUG_POSTFIX})
  endif()
else()
  set(CUSAN_DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
endif()

include(GNUInstallDirs)

set(CUSAN_PREFIX ${PROJECT_NAME})
set(TARGETS_EXPORT_NAME ${CUSAN_PREFIX}Targets)

if(CUSAN_CONFIG_DIR_IS_SHARE)
  set(CUSAN_INSTALL_CONFIGDIR ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME})
else()
  set(CUSAN_INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
endif()
