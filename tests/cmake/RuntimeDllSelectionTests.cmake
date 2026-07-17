cmake_minimum_required(VERSION 3.24)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
include("${REPO_ROOT}/cmake/AilaRuntimeDllSelection.cmake" OPTIONAL)

set(FIXTURE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/runtime-dll-selection-fixture")
set(SHARED_DIR "${FIXTURE_ROOT}/shared")
set(COMPILER_DIR "${FIXTURE_ROOT}/compiler")
file(REMOVE_RECURSE "${FIXTURE_ROOT}")
file(MAKE_DIRECTORY "${SHARED_DIR}" "${COMPILER_DIR}")

file(WRITE "${SHARED_DIR}/UMF.dll" "selected shared runtime")
file(WRITE "${SHARED_DIR}/libhwloc-15.dll" "selected shared hwloc")
file(WRITE "${COMPILER_DIR}/umf.dll" "conflicting compiler runtime")
file(WRITE "${COMPILER_DIR}/compiler-only.dll" "compiler-only runtime")

aila_select_runtime_dlls(
    OUTPUT_FILES selected_files
    DIRECTORIES "${SHARED_DIR}" "${COMPILER_DIR}"
)

list(LENGTH selected_files selected_count)
if(NOT selected_count EQUAL 3)
    message(FATAL_ERROR "Expected 3 unique runtime DLLs, found ${selected_count}: ${selected_files}")
endif()

set(selected_names)
set(selected_umf "")
set(selected_compiler_only "")
set(selected_hwloc "")
foreach(selected_file IN LISTS selected_files)
    get_filename_component(selected_name "${selected_file}" NAME)
    string(TOLOWER "${selected_name}" selected_key)
    list(FIND selected_names "${selected_key}" duplicate_index)
    if(NOT duplicate_index EQUAL -1)
        message(FATAL_ERROR "Duplicate case-insensitive runtime basename selected: ${selected_name}")
    endif()
    list(APPEND selected_names "${selected_key}")

    if(selected_key STREQUAL "umf.dll")
        set(selected_umf "${selected_file}")
    elseif(selected_key STREQUAL "compiler-only.dll")
        set(selected_compiler_only "${selected_file}")
    elseif(selected_key STREQUAL "libhwloc-15.dll")
        set(selected_hwloc "${selected_file}")
    endif()
endforeach()

if(NOT selected_umf STREQUAL "${SHARED_DIR}/UMF.dll")
    message(FATAL_ERROR "Shared runtime did not win UMF.dll precedence: ${selected_umf}")
endif()
if(NOT selected_compiler_only STREQUAL "${COMPILER_DIR}/compiler-only.dll")
    message(FATAL_ERROR "Compiler-only runtime was not preserved: ${selected_compiler_only}")
endif()
if(NOT selected_hwloc STREQUAL "${SHARED_DIR}/libhwloc-15.dll")
    message(FATAL_ERROR "Shared libhwloc runtime was not preserved: ${selected_hwloc}")
endif()

file(REMOVE_RECURSE "${FIXTURE_ROOT}")
message(STATUS "RuntimeDllSelectionTests PASS")
