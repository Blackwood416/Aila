include_guard(GLOBAL)

function(aila_configure_sycl_pch)
    if(NOT (AILA_ENABLE_SYCL_PCH AND WIN32 AND MSVC AND CMAKE_GENERATOR MATCHES "Ninja"))
        return()
    endif()

    set(AILA_SYCL_PCH_DIR "${CMAKE_CURRENT_BINARY_DIR}/sycl_pch")
    file(MAKE_DIRECTORY "${AILA_SYCL_PCH_DIR}")

    set(AILA_SYCL_PCH_HEADER "${AILA_SYCL_PCH_DIR}/aila_sycl_pch.hpp")
    set(AILA_SYCL_PCH_FILE "${AILA_SYCL_PCH_DIR}/aila_sycl_pch.pch")

    set(AILA_SYCL_PCH_CONTENT [=[
#pragma once
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <sycl/sycl.hpp>
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
]=])
    if(EXISTS "${AILA_SYCL_PCH_HEADER}")
        file(READ "${AILA_SYCL_PCH_HEADER}" AILA_EXISTING_PCH_CONTENT)
    else()
        set(AILA_EXISTING_PCH_CONTENT "")
    endif()
    if(NOT AILA_EXISTING_PCH_CONTENT STREQUAL AILA_SYCL_PCH_CONTENT)
        file(WRITE "${AILA_SYCL_PCH_HEADER}" "${AILA_SYCL_PCH_CONTENT}")
    endif()

    get_target_property(AILA_DNNL_INCLUDE_DIRS DNNL::dnnl INTERFACE_INCLUDE_DIRECTORIES)
    set(AILA_PCH_EXTERNAL_INCLUDES)
    foreach(inc_dir IN LISTS AILA_DNNL_INCLUDE_DIRS)
        if(inc_dir AND NOT inc_dir MATCHES "^\\$<")
            list(APPEND AILA_PCH_EXTERNAL_INCLUDES -external:I "${inc_dir}")
        endif()
    endforeach()

    set(AILA_PCH_DEFINES
        -DAILA_STATIC_LIB
        -DNOMINMAX
        -DWIN32_LEAN_AND_MEAN
        -D__TBB_NO_IMPLICIT_LINKAGE=1
    )
    if(TARGET AilaLevelZero)
        list(APPEND AILA_PCH_DEFINES -DAILA_HAS_LEVEL_ZERO=1)
    endif()

    string(TOUPPER "${CMAKE_BUILD_TYPE}" AILA_BUILD_TYPE_UPPER)
    set(AILA_PCH_CFG_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${AILA_BUILD_TYPE_UPPER}}")
    separate_arguments(AILA_PCH_CFG_FLAGS_LIST WINDOWS_COMMAND "${AILA_PCH_CFG_FLAGS}")

    set(AILA_PCH_CRT_FLAG "-MD")
    if(AILA_BUILD_TYPE_UPPER STREQUAL "DEBUG")
        set(AILA_PCH_CRT_FLAG "-MDd")
    endif()

    add_custom_command(
        OUTPUT "${AILA_SYCL_PCH_FILE}"
        COMMAND "${CMAKE_CXX_COMPILER}"
            /nologo
            ${AILA_PCH_DEFINES}
            ${AILA_PCH_EXTERNAL_INCLUDES}
            -external:W0
            ${AILA_PCH_CFG_FLAGS_LIST}
            ${AILA_PCH_CRT_FLAG}
            -fsycl
            -fsycl-device-code-split=per_kernel
            -fsycl-default-sub-group-size=16
            "${AILA_SYCL_PCH_HEADER}"
            -clang:-o "-clang:${AILA_SYCL_PCH_FILE}"
        DEPENDS "${AILA_SYCL_PCH_HEADER}"
        COMMENT "Building bundled SYCL precompiled header ${AILA_SYCL_PCH_FILE}"
        VERBATIM
    )
    add_custom_target(AilaSyclPch DEPENDS "${AILA_SYCL_PCH_FILE}")

    set(AILA_SYCL_PCH_DIR "${AILA_SYCL_PCH_DIR}" PARENT_SCOPE)
    set(AILA_SYCL_PCH_FILE "${AILA_SYCL_PCH_FILE}" PARENT_SCOPE)
endfunction()

function(aila_enable_sycl_pch TARGET_NAME)
    if(NOT TARGET AilaSyclPch OR NOT AILA_SYCL_PCH_FILE)
        return()
    endif()

    cmake_parse_arguments(PARSE_ARGV 1 AILA_PCH "" "" "SOURCES")
    if(NOT AILA_PCH_SOURCES)
        get_target_property(AILA_PCH_SOURCES "${TARGET_NAME}" SOURCES)
    endif()

    add_dependencies("${TARGET_NAME}" AilaSyclPch)

    foreach(src IN LISTS AILA_PCH_SOURCES)
        if(NOT src MATCHES "\\.cpp$")
            continue()
        endif()
        string(REGEX REPLACE "[^A-Za-z0-9_]" "_" src_id "${TARGET_NAME}_${src}")
        set(int_hdr "${AILA_SYCL_PCH_DIR}/${src_id}_int.h")
        set(wrap_hdr "${AILA_SYCL_PCH_DIR}/${src_id}_wrap.h")
        file(TO_NATIVE_PATH "${int_hdr}" int_hdr_native)
        file(TO_NATIVE_PATH "${wrap_hdr}" wrap_hdr_native)
        string(REPLACE "\\" "\\\\" int_hdr_escaped "${int_hdr_native}")

        set(wrap_content "#pragma clang system_header\n#ifndef __SYCL_DEVICE_ONLY__\n#include \"${int_hdr_escaped}\"\n#endif\n")
        if(EXISTS "${wrap_hdr}")
            file(READ "${wrap_hdr}" existing_wrap)
        else()
            set(existing_wrap "")
        endif()
        if(NOT existing_wrap STREQUAL wrap_content)
            file(WRITE "${wrap_hdr}" "${wrap_content}")
        endif()

        set_property(SOURCE "${src}" APPEND PROPERTY OBJECT_DEPENDS "${AILA_SYCL_PCH_FILE}")
        set_property(SOURCE "${src}" APPEND PROPERTY COMPILE_OPTIONS
            "-clang:-include-pch"
            "-clang:${AILA_SYCL_PCH_FILE}"
            "-Xclang" "-fsycl-int-header=${int_hdr_native}"
            "-Xclang" "-include" "-Xclang" "${wrap_hdr_native}"
            "-Xclang" "-dependency-filter" "-Xclang" "${wrap_hdr_native}"
        )
    endforeach()
endfunction()
