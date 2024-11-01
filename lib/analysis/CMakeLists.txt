add_library(cusan_Analysis STATIC KernelModel.cpp KernelAnalysis.cpp ModelIO.cpp)
set_target_properties(
        cusan_Analysis
        PROPERTIES
        OUTPUT_NAME "CusanAnalysis"
        EXPORT_NAME "Analysis"
)
add_library(cusan::Analysis ALIAS cusan_Analysis)

target_compile_features(cusan_Analysis PUBLIC cxx_std_17)

target_include_directories(cusan_Analysis
        SYSTEM
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
)

target_include_directories(cusan_Analysis ${warning_guard}
        PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

cusan_target_define_file_basename(cusan_Analysis)

target_compile_definitions(cusan_Analysis PRIVATE "LLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}")

set(CONFIG_NAME cusanAnalysis)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(
        TARGETS cusan_Analysis
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
