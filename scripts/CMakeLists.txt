function(configure_cusan_script input output)
  cmake_parse_arguments(
    ARG "" "" "INSTALL_MODE;COMPILER;WITH_FILTER;APPLY_MODE" ${ARGN}
  )

  set(CUSAN_BINARY_DIR ${cusan_BINARY_DIR})

  set(TYPEART_PROJECT_DIR ${typeart_SOURCE_DIR})
  set(TYPEART_BIN_DIR ${typeart_BINARY_DIR})
  set(TYPEART_SCRIPT_DIR ${TYPEART_PROJECT_DIR}/scripts)

  if(ARG_INSTALL_MODE)
    set(CUSAN_RT_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set(CUSAN_INCLUDE_DIRS "-I${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
    set(CUSAN_RELOCATABLE 1)

    set(TYPEART_INCLUDE_DIRS
        "-I${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
    )
    set(TYPEART_RT_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set(TYPEART_PASS_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set(TYPEART_ANALYSIS_PASS_DIR
        ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
    )
    set(TYPEART_BINARY_DIR -I${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
  else()
    set(CUSAN_INCLUDE_DIRS "-I${PROJECT_SOURCE_DIR}/lib/runtime")
    set(CUSAN_RELOCATABLE 0)

    set(TYPEART_INCLUDE_DIRS
        "-I${TYPEART_PROJECT_DIR}/lib/typelib -I${TYPEART_PROJECT_DIR}/lib/runtime -I${TYPEART_PROJECT_DIR}/lib/passes/typegen"
    )
    if(LIBRARY_OUTPUT_PATH)   
      set(TYPEART_RT_DIR ${LIBRARY_OUTPUT_PATH})
      set(TYPEART_PASS_DIR ${LIBRARY_OUTPUT_PATH})
      set(TYPEART_ANALYSIS_PASS_DIR ${LIBRARY_OUTPUT_PATH})

      set(CUSAN_RT_DIR ${LIBRARY_OUTPUT_PATH})
      set(CUSAN_PASS_DIR ${LIBRARY_OUTPUT_PATH})
    else()
      set(TYPEART_RT_DIR ${TYPEART_BIN_DIR}/lib/runtime)
      set(TYPEART_PASS_DIR ${TYPEART_BIN_DIR}/lib/passes)
      set(TYPEART_ANALYSIS_PASS_DIR ${TYPEART_BIN_DIR}/analysis)

      set(CUSAN_RT_DIR ${CMAKE_BINARY_DIR}/lib/runtime)
      set(CUSAN_PASS_DIR ${CMAKE_BINARY_DIR}/lib/pass)
    endif()

    if(EXECUTABLE_OUTPUT_PATH)
        set(TYPEART_BINARY_DIR ${EXECUTABLE_OUTPUT_PATH})
        set(CUSAN_BINARY_DIR ${EXECUTABLE_OUTPUT_PATH})
    else()
        set(TYPEART_BINARY_DIR ${TYPEART_BIN_DIR})
        set(CUSAN_BINARY_DIR ${CUSAN_BINARY_DIR})
    endif()
  endif()

  if(ARG_COMPILER)
    set(CUSAN_COMPILER ${ARG_COMPILER})
  endif()

  set(CUSAN_OPT "${CUSAN_OPT_EXEC}")
  if(${LLVM_VERSION_MAJOR} VERSION_GREATER_EQUAL "13")
    set(CUSAN_OPT "${CUSAN_OPT} -enable-new-pm=0")
  endif()

  if(ARG_APPLY_MODE)
    set(TYPEART_RUN_SCRIPT 0)
    set(CUSAN_RUN_SCRIPT 0)
  else()
    set(TYPEART_RUN_SCRIPT 1)
    set(CUSAN_RUN_SCRIPT 1)
  endif()

  if(ARG_WITH_FILTER)
    set(TYPEART_CALLFILTER "--typeart-filter")
  endif()

  set(CUSAN_SAN_FLAG "-fsanitize=thread")

  cusan_target_generate_file(${input} ${output})
endfunction()

function(typeart_find_mpi_vendor_helper symbol ret_value)
  find_package(MPI)

  if(NOT MPI_FOUND)
    set(${ret_value} false)
    return()
  endif()

  set(MPI_TEST_CODE
   "#include <mpi.h>
    int main(void) {
      #if !defined(${symbol})
        an_illegal_stmt
      #endif
      return 0;
    }")

  set(CMAKE_REQUIRED_INCLUDES ${MPI_C_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_FLAGS "${MPI_C_COMPILE_FLAGS}")
  set(CMAKE_REQUIRED_DEFINITIONS "")
  set(CMAKE_REQUIRED_LIBRARIES "${MPI_C_LIBRARIES}")

  check_c_source_compiles("${MPI_TEST_CODE}" ${ret_value})
endfunction()

function(cusan_find_mpi_vendor_helper symbol ret_value)
  find_package(MPI)

  if(NOT MPI_FOUND)
    set(${ret_value} false)
    return()
  endif()

  set(MPI_TEST_CODE
   "#include <mpi.h>
    int main(void) {
      #if !defined(${symbol})
        an_illegal_stmt
      #endif
      return 0;
    }")

  set(CMAKE_REQUIRED_INCLUDES ${MPI_C_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_FLAGS "${MPI_C_COMPILE_FLAGS}")
  set(CMAKE_REQUIRED_DEFINITIONS "")
  set(CMAKE_REQUIRED_LIBRARIES "${MPI_C_LIBRARIES}")

  check_c_source_compiles("${MPI_TEST_CODE}" ${ret_value})
endfunction()

function(set_cusan_mpi_compiler)
  # Note on mpich & Intel: Intel may also define \"MPICH_NAME\"
  # in mpi.h, so check MPICH last
  list(APPEND l_vendor OPEN_MPI INTEL_MPI MPICH)
  list(APPEND l_symbol OPEN_MPI I_MPI_VERSION MPICH_NAME)
  list(APPEND l_env_c OMPI_CC I_MPI_CC MPICH_CC)
  list(APPEND l_env_cxx OMPI_CXX I_MPI_CXX MPICH_CXX)

  foreach(vendor symbol env_c env_cxx IN ZIP_LISTS l_vendor l_symbol l_env_c l_env_cxx)
    cusan_find_mpi_vendor_helper(${symbol} CUSAN_HAVE_${vendor})
    if(CUSAN_HAVE_${vendor})
      if(MPI_C_FOUND)
        set(CUSAN_MPICC
            "env ${env_c}=${CUSAN_CLANG_EXEC} ${MPI_C_COMPILER}"
            CACHE STRING "CuSan MPICC compiler command for scripts"
          )
        mark_as_advanced(CUSAN_MPICC)
      endif()
      if(MPI_CXX_FOUND)
        set(CUSAN_MPICXX
            "env ${env_cxx}=${CUSAN_CLANGCXX_EXEC} ${MPI_CXX_COMPILER}"
            CACHE STRING "CuSan MPICXX compiler command for scripts"
        )
        mark_as_advanced(CUSAN_MPICXX)
      endif()
      break()
    endif()
  endforeach()
endfunction()

if(MPI_FOUND)
  set_cusan_mpi_compiler()
endif()

if(CUSAN_TYPEART)
  set(CUSAN_WRAPPER cusan-wrapper.in)
else()
  set(CUSAN_WRAPPER cusan-wrapper-no-typeart.in)
endif()

find_package(MPI)
if(MPI_C_FOUND)
  configure_cusan_script(
    ${CUSAN_WRAPPER} cusan-mpicc${CMAKE_DEBUG_POSTFIX}
    INSTALL_MODE ON
    WITH_FILTER ON
    COMPILER "${CUSAN_MPICC}"
  )
  configure_cusan_script(
    ${CUSAN_WRAPPER} cusan-mpicc-test
    WITH_FILTER ON
    COMPILER "${CUSAN_MPICC}"
  )
endif()

if(MPI_CXX_FOUND)
  configure_cusan_script(
    ${CUSAN_WRAPPER} cusan-mpic++${CMAKE_DEBUG_POSTFIX}
    INSTALL_MODE ON
    WITH_FILTER ON
    COMPILER "${CUSAN_MPICXX}"
  )
  configure_cusan_script(
    ${CUSAN_WRAPPER} cusan-mpic++-test
    WITH_FILTER ON
    COMPILER "${CUSAN_MPICXX}"
  )
endif()

configure_cusan_script(
  ${CUSAN_WRAPPER} cusan-clang${CMAKE_DEBUG_POSTFIX}
  INSTALL_MODE ON
  COMPILER ${CUSAN_CLANG_EXEC}
)
configure_cusan_script(
  ${CUSAN_WRAPPER} cusan-clang++${CMAKE_DEBUG_POSTFIX}
  INSTALL_MODE ON
  COMPILER ${CUSAN_CLANGCXX_EXEC}
)

configure_cusan_script(
  ${CUSAN_WRAPPER} cusan-clang-test
  COMPILER ${CUSAN_CLANG_EXEC}
)
configure_cusan_script(
  ${CUSAN_WRAPPER} cusan-clang++-test
  COMPILER ${CUSAN_CLANGCXX_EXEC}
)

#configure_cusan_script(cusan-tmpl.sh.in run.sh)
if(CUSAN_TYPEART)
  configure_cusan_script(
    cusan-tmpl.sh.in apply.sh
    APPLY_MODE ON
  )
endif()

configure_cusan_script(${CUSAN_WRAPPER} cusan-clang.sh)

install(
  PROGRAMS
    ${CMAKE_CURRENT_BINARY_DIR}/cusan-clang++${CMAKE_DEBUG_POSTFIX}
    ${CMAKE_CURRENT_BINARY_DIR}/cusan-clang${CMAKE_DEBUG_POSTFIX}
    $<$<BOOL:${MPI_C_FOUND}>:${CMAKE_CURRENT_BINARY_DIR}/cusan-mpicc${CMAKE_DEBUG_POSTFIX}>
    $<$<BOOL:${MPI_CXX_FOUND}>:${CMAKE_CURRENT_BINARY_DIR}/cusan-mpic++${CMAKE_DEBUG_POSTFIX}>
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
