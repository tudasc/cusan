cmake_minimum_required(VERSION 3.20)

project(
    cusan
    VERSION 0.3.0
    DESCRIPTION "A data race detector for CUDA based on ThreadSanitizer"
    HOMEPAGE_URL "https://github.com/tudasc/cusan"
    LANGUAGES CXX C
)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(
  APPEND
  CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)

include(cusanToolchain)

cusan_target_format(
  format-cusan-sources "Formats project source files"
  TARGETS lib/pass/*.cpp
          lib/pass/*.h
)

add_subdirectory(externals)
add_subdirectory(lib)
add_subdirectory(scripts)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/cusanConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMinorVersion
)

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(LIBRARY_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(BINARY_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
set(CMAKE_INSTALL_DIR ${CUSAN_INSTALL_CONFIGDIR})

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/cusanConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cusanConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_DIR}
  PATH_VARS INCLUDE_INSTALL_DIR
            LIBRARY_INSTALL_DIR
            BINARY_INSTALL_DIR
            CMAKE_INSTALL_DIR
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cusanConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/cusanConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_DIR}
)

if(PROJECT_IS_TOP_LEVEL)
    feature_summary(WHAT ENABLED_FEATURES PACKAGES_FOUND PACKAGES_NOT_FOUND
      DESCRIPTION "cusan ${PROJECT_VERSION} package and feature info:"
      INCLUDE_QUIET_PACKAGES
      QUIET_ON_EMPTY
    )
endif()

if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
  add_subdirectory(test)
endif()
