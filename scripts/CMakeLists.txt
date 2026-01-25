function(configure_cusan_script input output)
  cmake_parse_arguments(
    ARG "" "" "INSTALL_MODE;COMPILER;WITH_FILTER;APPLY_MODE;WRAPPER_MODE;EXTRA_LDFLAGS" ${ARGN}
  )

  set(CUSAN_WRAPPER_MODE ${ARG_WRAPPER_MODE})
  set(CUSAN_EXTRA_LDFLAGS ${ARG_EXTRA_LDFLAGS})

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
    if(CUSAN_TYPEART)
      set(CUSAN_OPT "${CUSAN_OPT} -enable-new-pm=0")
    else()
      set(CUSAN_NEW_PM_REQUIRED 1)
    endif()
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

  if(CUSAN_TYPEART)
    set(CUSAN_WITH_TYPEART 1)
  else()
  set(CUSAN_WITH_TYPEART 0)
  endif()

  cusan_target_generate_file(${input} ${output})
endfunction()

function(configure_cusan_wrapper name compiler)
  cmake_parse_arguments(ARG "WITH_FILTER" "" "" ${ARGN})
  set(extra_args "")
  if(ARG_WITH_FILTER)
    set(extra_args WITH_FILTER ON)
  endif()

  # Configure for installation
  configure_cusan_script(
    ${CUSAN_WRAPPER} ${name}${CMAKE_DEBUG_POSTFIX}
    INSTALL_MODE ON
    COMPILER "${compiler}"
    WRAPPER_MODE "${CUSAN_WRAPPER_MODE_AUTO}"
    EXTRA_LDFLAGS "${CUSAN_EXTRA_LDFLAGS_AUTO}"
    ${extra_args}
  )

  # Configure for tests
  configure_cusan_script(
    ${CUSAN_WRAPPER} ${name}-test
    COMPILER "${compiler}"
    WRAPPER_MODE "${CUSAN_WRAPPER_MODE_AUTO}"
    EXTRA_LDFLAGS "${CUSAN_EXTRA_LDFLAGS_AUTO}"
    ${extra_args}
  )

  # Build the installation list
  set(CUSAN_SCRIPTS_TO_INSTALL
      ${CUSAN_SCRIPTS_TO_INSTALL}
      ${CMAKE_CURRENT_BINARY_DIR}/${name}${CMAKE_DEBUG_POSTFIX}
      PARENT_SCOPE
  )
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

  if(CUSAN_HIP)
    set(m_base_c ${CUSAN_HIPCC_EXEC})
    set(m_base_cxx ${CUSAN_HIPCC_EXEC})
  else()
    set(m_base_c ${CUSAN_CLANG_EXEC})
    set(m_base_cxx ${CUSAN_CLANGCXX_EXEC})
  endif()

  foreach(vendor symbol env_c env_cxx IN ZIP_LISTS l_vendor l_symbol l_env_c l_env_cxx)
    cusan_find_mpi_vendor_helper(${symbol} CUSAN_HAVE_${vendor})
    if(CUSAN_HAVE_${vendor})
      if(MPI_C_FOUND)
        set(CUSAN_MPICC
            "env ${env_c}=${m_base_c} ${MPI_C_COMPILER}"
            CACHE STRING "CuSan MPICC compiler command for scripts" FORCE
          )
        mark_as_advanced(CUSAN_MPICC)
      endif()
      if(MPI_CXX_FOUND)
        set(CUSAN_MPICXX
            "env ${env_cxx}=${m_base_cxx} ${MPI_CXX_COMPILER}"
            CACHE STRING "CuSan MPICXX compiler command for scripts" FORCE
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
  set(CUSAN_WRAPPER cusan-wrapperv2.in)
endif()

set(CUSAN_WRAPPER_MODE_AUTO "CUDA")
if(CUDAToolkit_LIBRARY_DIR)
  set(CUSAN_EXTRA_LDFLAGS_AUTO "-L${CUDAToolkit_LIBRARY_DIR}/ -lcudart -lcudart_static -lcudadevrt")
else()
  set(CUSAN_EXTRA_LDFLAGS_AUTO "-lcudart -lcudart_static -lcudadevrt")
endif()
if(CUSAN_HIP)
  set(CUSAN_WRAPPER_MODE_AUTO "HIP")
  set(CUSAN_EXTRA_LDFLAGS_AUTO "")
endif()

if(NOT CUSAN_TYPEART)
  add_library(typeart_runtime_mock SHARED IMPORTED)
  add_library(typeart::Runtime ALIAS typeart_runtime_mock)
  set_target_properties(
    typeart_runtime_mock
    PROPERTIES
    OUTPUT_NAME "${PROJECT_NAME}RuntimePlaceholder"
    IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/lib/runtime"
  )
  add_library(typeart_transform_mock SHARED IMPORTED)
  add_library(typeart::TransformPass ALIAS typeart_transform_mock)
  set_target_properties(
    typeart_transform_mock
    PROPERTIES
    OUTPUT_NAME "${PROJECT_NAME}TransformPassPlaceholder"
    IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/lib/runtime"
  )
endif()

# Will be appended in configure_cusan_wrapper calls:
set(CUSAN_SCRIPTS_TO_INSTALL "")

if(MPI_C_FOUND)
  configure_cusan_wrapper(cusan-mpicc "${CUSAN_MPICC}" WITH_FILTER)
endif()

if(MPI_CXX_FOUND)
  configure_cusan_wrapper(cusan-mpic++ "${CUSAN_MPICXX}" WITH_FILTER)
endif()

if(CUSAN_HIP)
  configure_cusan_wrapper(cusan-hipcc "${CUSAN_HIPCC_EXEC}")
else()
  configure_cusan_wrapper(cusan-clang "${CUSAN_CLANG_EXEC}")
  configure_cusan_wrapper(cusan-clang++ "${CUSAN_CLANGCXX_EXEC}")
endif()

install(
  PROGRAMS ${CUSAN_SCRIPTS_TO_INSTALL}
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
