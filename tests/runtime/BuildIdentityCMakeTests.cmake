cmake_minimum_required(VERSION 3.24)

foreach(required_variable IN ITEMS SOURCE_DIR BINARY_DIR GIT_EXECUTABLE WORKER_FILE)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

function(git_output output_variable)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "git ${ARGN} failed: ${error}")
    endif()
    set(${output_variable} "${output}" PARENT_SCOPE)
endfunction()

function(normalize_path output_variable input_path)
    cmake_path(SET normalized_path NORMALIZE "${input_path}")
    if(NOT IS_ABSOLUTE "${normalized_path}")
        cmake_path(
            ABSOLUTE_PATH normalized_path
            BASE_DIRECTORY "${SOURCE_DIR}"
            NORMALIZE
        )
    endif()
    set(${output_variable} "${normalized_path}" PARENT_SCOPE)
endfunction()

function(require_ninja_dependency metadata_path description)
    if(NOT EXISTS "${metadata_path}")
        return()
    endif()
    string(FIND "${rerun_cmake_rule}" "${metadata_path}" dependency_index)
    if(dependency_index EQUAL -1)
        message(FATAL_ERROR
            "build.ninja RERUN_CMAKE dependencies omit ${description}: ${metadata_path}")
    endif()
endfunction()

git_output(current_revision rev-parse HEAD)
git_output(raw_head_path rev-parse --git-path HEAD)
normalize_path(head_path "${raw_head_path}")

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" symbolic-ref -q HEAD
    RESULT_VARIABLE symbolic_ref_result
    OUTPUT_VARIABLE symbolic_ref
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(symbolic_ref_result EQUAL 0)
    git_output(raw_ref_path rev-parse --git-path "${symbolic_ref}")
    normalize_path(ref_path "${raw_ref_path}")
elseif(NOT symbolic_ref_result EQUAL 1)
    message(FATAL_ERROR "git symbolic-ref -q HEAD failed with ${symbolic_ref_result}")
endif()

git_output(raw_packed_refs_path rev-parse --git-path packed-refs)
normalize_path(packed_refs_path "${raw_packed_refs_path}")

set(build_ninja "${BINARY_DIR}/build.ninja")
if(NOT EXISTS "${build_ninja}")
    message(FATAL_ERROR "Ninja manifest does not exist: ${build_ninja}")
endif()
file(READ "${build_ninja}" normalized_ninja)
string(REPLACE "$:" ":" normalized_ninja "${normalized_ninja}")
string(REPLACE "$ " " " normalized_ninja "${normalized_ninja}")
string(REPLACE "\\" "/" normalized_ninja "${normalized_ninja}")
string(REGEX MATCH "build build\\.ninja[^\r\n]*" rerun_cmake_rule "${normalized_ninja}")
if(rerun_cmake_rule STREQUAL "")
    message(FATAL_ERROR "build.ninja does not contain a RERUN_CMAKE manifest rule")
endif()

require_ninja_dependency("${head_path}" "active worktree HEAD")
if(symbolic_ref_result EQUAL 0)
    require_ninja_dependency("${ref_path}" "active symbolic ref")
endif()
require_ninja_dependency("${packed_refs_path}" "packed refs")

if(NOT EXISTS "${WORKER_FILE}")
    message(FATAL_ERROR "worker executable does not exist: ${WORKER_FILE}")
endif()
set(expected_build_id "aila-0.1.7-abi1-${current_revision}")
file(STRINGS "${WORKER_FILE}" matching_build_ids REGEX "${expected_build_id}")
if(NOT matching_build_ids)
    message(FATAL_ERROR
        "compiled AILA_BUILD_ID does not match current Git revision: ${expected_build_id}")
endif()

message(STATUS "Aila build identity dependencies and compiled revision are current")
