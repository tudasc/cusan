function(cusan_llvm_module name sources)
  # TODO default of include_dirs is private
  cmake_parse_arguments(ARG "" "" "INCLUDE_DIRS;DEPENDS;LINK_LIBS" ${ARGN})

  add_llvm_pass_plugin(${name}
    ${sources}
    LINK_LIBS LLVMDemangle ${ARG_LINK_LIBS}
    DEPENDS ${ARG_DEPENDS}
  )

  target_compile_definitions(${name}
    PRIVATE
    LLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}
  )

  target_include_directories(${name}
    SYSTEM
    PRIVATE
    ${LLVM_INCLUDE_DIRS}
  )

  if(ARG_INCLUDE_DIRS)
    target_include_directories(${name} ${warning_guard}
      PRIVATE
      ${ARG_INCLUDE_DIRS}
    )
  endif()

  cusan_target_define_file_basename(${name})

  target_compile_definitions(${name}
    PRIVATE
    ${LLVM_DEFINITIONS}
  )
endfunction()

function(cusan_find_llvm_progs target names)
  cmake_parse_arguments(ARG "ABORT_IF_MISSING;SHOW_VAR" "DEFAULT_EXE" "HINTS" ${ARGN})
  set(TARGET_TMP ${target})

  find_program(
    ${target}
    NAMES ${names}
    PATHS ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH
  )
  if(NOT ${target})
    find_program(
      ${target}
      NAMES ${names}
      HINTS ${ARG_HINTS}
    )
  endif()

  if(NOT ${target})
    set(target_missing_message "")
    if(ARG_DEFAULT_EXE)
      unset(${target} CACHE)
      set(${target}
          ${ARG_DEFAULT_EXE}
          CACHE
          STRING
          "Default value for ${TARGET_TMP}."
      )
      set(target_missing_message "Using default: ${ARG_DEFAULT_EXE}")
    endif()

    set(message_status STATUS)
    if(ARG_ABORT_IF_MISSING AND NOT ARG_DEFAULT_EXE)
      set(message_status SEND_ERROR)
    endif()
    message(${message_status}
      "Did not find LLVM program " "${names}"
      " in ${LLVM_TOOLS_BINARY_DIR}, in system path or hints " "\"${ARG_HINTS}\"" ". "
      ${target_missing_message}
    )
  endif()

  if(ARG_SHOW_VAR)
    mark_as_advanced(CLEAR ${target})
  else()
    mark_as_advanced(${target})
  endif()
endfunction()
