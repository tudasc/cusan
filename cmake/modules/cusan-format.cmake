function(cusan_target_format target comment)
  macro(filter_dir dir_name_)
    foreach (source_file ${ALL_CXX_FILES})
      string(FIND ${source_file} ${dir_name_} EXCLUDE_FOUND)
      if (NOT ${EXCLUDE_FOUND} EQUAL -1)
        list(REMOVE_ITEM ALL_CXX_FILES ${source_file})
      endif()
    endforeach()
  endmacro()

  cmake_parse_arguments(ARG "" "" "TARGETS;EXCLUDES;OTHER" ${ARGN})

  file(GLOB_RECURSE
    ALL_CXX_FILES
    CONFIGURE_DEPENDS
    ${ARG_TARGETS}
  )

  foreach(exclude ${ARG_EXCLUDES})
    filter_dir(${exclude})
  endforeach()

  find_program(CLANG_FORMAT_COMMAND
               NAMES clang-format-${LLVM_VERSION_MAJOR} clang-format)
  mark_as_advanced(CLANG_FORMAT_COMMAND)
  if(CLANG_FORMAT_COMMAND)
    add_custom_target(${target}
      COMMAND ${CLANG_FORMAT_COMMAND} -i -style=file ${ARG_OTHER} ${ARG_UNPARSED_ARGUMENTS}
              ${ALL_CXX_FILES}
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      COMMENT "${comment}"
      USES_TERMINAL
    )
  else()
    message(WARNING "Could not find clang-format executable.")
    add_custom_target(${target}
      COMMAND ${CMAKE_COMMAND} -E echo "${target} does nothing, no clang-format found.")
  endif()
endfunction()

