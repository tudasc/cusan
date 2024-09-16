set(PASS_SOURCES
  CusanPass.cpp
  AnalysisTransform.cpp
)

cusan_llvm_module(cusan_TransformPass
  "${PASS_SOURCES}"
  LINK_LIBS cusan::Analysis
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/lib
)
mark_as_advanced(LLVM_CUSAN_TRANSFORMPASS_LINK_INTO_TOOLS)

set_target_properties(
  cusan_TransformPass
  PROPERTIES
  OUTPUT_NAME "${PROJECT_NAME}TransformPass"
  EXPORT_NAME "TransformPass"
)

add_library(cusan::TransformPass ALIAS cusan_TransformPass)

target_compile_features(cusan_TransformPass PUBLIC cxx_std_17)

target_include_directories(cusan_TransformPass ${warning_guard}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
          ${PROJECT_SOURCE_DIR}/lib/
)

set(CONFIG_NAME cusanTransformPass)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(
  TARGETS cusan_TransformPass
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
