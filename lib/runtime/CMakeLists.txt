find_package(MPI)
add_library(cusan_mpi_interceptor SHARED MPIInterception.cpp)
target_link_libraries(cusan_mpi_interceptor
  PRIVATE MPI::MPI_CXX
)
set_target_properties(
  cusan_mpi_interceptor
  PROPERTIES
  OUTPUT_NAME "CusanMPIInterceptor"
  EXPORT_NAME "MPIInterceptor"
)
add_library(cusan::MPI_Interceptor ALIAS cusan_mpi_interceptor)
target_compile_features(cusan_mpi_interceptor PUBLIC cxx_std_17)
target_compile_definitions(
  cusan_mpi_interceptor
  PRIVATE CUSAN_LOG_LEVEL=${CUSAN_LOG_LEVEL_RT}
  LLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}
   $<$<BOOL:${CUSAN_FIBERPOOL}>:CUSAN_FIBERPOOL=1>
   $<$<BOOL:${CUSAN_TYPEART}>:CUSAN_TYPEART=1>
   $<$<BOOL:${CUSAN_SOFTCOUNTER}>:CUSAN_SOFTCOUNTER>
)

target_include_directories(cusan_mpi_interceptor ${warning_guard}
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

target_include_directories(cusan_mpi_interceptor
  SYSTEM
  PRIVATE
  ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(cusan_mpi_interceptor PRIVATE LLVMSupport)

if(CUSAN_FIBERPOOL)
  target_link_libraries(cusan_mpi_interceptor PUBLIC cusan::fiberpool)
  set_target_properties(cusan_mpi_interceptor PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()

set(CONFIG_NAME cusanMPIInterceptor)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(
  TARGETS cusan_mpi_interceptor
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  EXPORT ${TARGETS_EXPORT_NAME}
  NAMESPACE cusan::
  DESTINATION ${CUSAN_INSTALL_CONFIGDIR}
)

export(
  EXPORT ${TARGETS_EXPORT_NAME}
  FILE ${CMAKE_BINARY_DIR}/${TARGETS_EXPORT_NAME}.cmake
  NAMESPACE cusan::
)

add_library(cusan_Runtime SHARED CusanRuntime.cpp CusanRuntime_cudaSpecific.cpp)
set_target_properties(
  cusan_Runtime
  PROPERTIES
  OUTPUT_NAME "CusanRuntime"
  EXPORT_NAME "Runtime"
)
add_library(cusan::Runtime ALIAS cusan_Runtime)

target_compile_features(cusan_Runtime PUBLIC cxx_std_17)

# latter for cudaSpecific.cpp:
target_link_libraries(cusan_Runtime PRIVATE CUDA::cudart)

if(CUSAN_TYPEART)
  target_link_libraries(cusan_Runtime PRIVATE typeart::Runtime)
endif()
target_link_libraries(cusan_Runtime PRIVATE LLVMSupport)


if(CUSAN_FIBERPOOL)
  target_link_libraries(cusan_Runtime PUBLIC cusan::fiberpool)
  set_target_properties(cusan_Runtime PROPERTIES
      BUILD_WITH_INSTALL_RPATH FALSE
      LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()


target_include_directories(cusan_Runtime
  SYSTEM
  PRIVATE
  ${LLVM_INCLUDE_DIRS}
)

target_include_directories(cusan_Runtime ${warning_guard}
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

cusan_target_define_file_basename(cusan_Runtime)

target_compile_definitions(
  cusan_Runtime
  PRIVATE CUSAN_LOG_LEVEL=${CUSAN_LOG_LEVEL_RT}
   LLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}
   $<$<BOOL:${CUSAN_SYNC_DETAIL_LEVEL}>:CUSAN_SYNC_DETAIL_LEVEL=1>
   $<$<BOOL:${CUSAN_FIBERPOOL}>:CUSAN_FIBERPOOL=1>
   $<$<BOOL:${CUSAN_TYPEART}>:CUSAN_TYPEART=1>
   $<$<BOOL:${CUSAN_SOFTCOUNTER}>:CUSAN_SOFTCOUNTER>
)

set(CONFIG_NAME cusanRuntime)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(
  TARGETS cusan_Runtime
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(CUSAN_FIBERPOOL)
install(
  TARGETS cusan_fiberpool
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
endif()

install(
  EXPORT ${TARGETS_EXPORT_NAME}
  NAMESPACE cusan::
  DESTINATION ${CUSAN_INSTALL_CONFIGDIR}
)

export(
  EXPORT ${TARGETS_EXPORT_NAME}
  FILE ${CMAKE_BINARY_DIR}/${TARGETS_EXPORT_NAME}.cmake
  NAMESPACE cusan::
)
