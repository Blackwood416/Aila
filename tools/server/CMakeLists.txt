cmake_minimum_required(VERSION 3.24)

# Keep this a separate host-only project: the repository's top-level project
# selects the oneAPI compiler and owns the independently built Worker.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" CACHE STRING "Server CRT linkage")
project(AilaServer LANGUAGES CXX)

if(NOT WIN32 OR NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(FATAL_ERROR "AilaServer requires ordinary Windows MSVC (cl.exe).")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "AilaServer requires the x64 MSVC toolchain.")
endif()
if(NOT CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreadedDLL")
    message(FATAL_ERROR "The pinned server dependencies require /MD.")
endif()

function(aila_detect_msvc_include_prefix)
    # Some MSVC installations emit UTF-8 even when CMake's ENCODING AUTO
    # selects the ANSI code page. Ninja compares raw bytes, so do not decode
    # the compiler's localized /showIncludes prefix or assume VSLANG works.
    set(probe_dir "${CMAKE_BINARY_DIR}/CMakeFiles/AilaShowIncludes")
    file(MAKE_DIRECTORY "${probe_dir}")
    file(WRITE "${probe_dir}/aila_server_include_probe.hpp" "#pragma once\n")
    file(WRITE "${probe_dir}/aila_server_include_probe.cpp"
        "#include \"aila_server_include_probe.hpp\"\nint aila_server_include_probe() { return 0; }\n")
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" /nologo /TP /c /std:c++17 /MD /utf-8
            /showIncludes /Foaila_server_include_probe.obj aila_server_include_probe.cpp
        WORKING_DIRECTORY "${probe_dir}"
        OUTPUT_VARIABLE probe_stdout
        ERROR_VARIABLE probe_stderr
        RESULT_VARIABLE probe_result
        TIMEOUT 20
        ENCODING NONE)
    file(WRITE "${probe_dir}/stdout.bin" "${probe_stdout}")
    file(WRITE "${probe_dir}/stderr.bin" "${probe_stderr}")
    if(NOT probe_result STREQUAL "0")
        message(FATAL_ERROR "MSVC /showIncludes probe failed (${probe_result}); see ${probe_dir}.")
    endif()
    # Match only our known header, with a drive-qualified or UNC path. Keep
    # the exact prefix whitespace, including the first include-depth indent.
    if(NOT "${probe_stdout}" MATCHES "(^|\n)([^\r\n]*[ \t])(([A-Za-z]:[/\\\\]|\\\\\\\\)[^\r\n]*aila_server_include_probe\\.hpp)\r?(\n|$)")
        message(FATAL_ERROR "Cannot identify MSVC /showIncludes output; see ${probe_dir}/stdout.bin.")
    endif()
    set(probe_prefix "${CMAKE_MATCH_2}")
    string(SHA256 probe_hash "${probe_prefix}")
    file(WRITE "${probe_dir}/prefix.bin" "${probe_prefix}")
    file(WRITE "${probe_dir}/prefix.sha256" "${probe_hash}\n")
    set(CMAKE_CL_SHOWINCLUDES_PREFIX "${probe_prefix}" PARENT_SCOPE)
    set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "${probe_prefix}" PARENT_SCOPE)
    set(AILA_SERVER_MSVC_DEPS_PREFIX_SHA256 "${probe_hash}" PARENT_SCOPE)
    message(STATUS "MSVC /showIncludes raw prefix SHA256: ${probe_hash}")
endfunction()

if(CMAKE_GENERATOR MATCHES "^Ninja")
    aila_detect_msvc_include_prefix()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "Server configurations" FORCE)
elseif(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Server configuration" FORCE)
elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR "The pinned server dependencies support Release builds only.")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

get_filename_component(AILA_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
file(READ "${AILA_REPO_ROOT}/VERSION.txt" AILA_VERSION_RAW)
string(STRIP "${AILA_VERSION_RAW}" AILA_PUBLIC_API_VERSION)
if(NOT AILA_PUBLIC_API_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "Invalid VERSION.txt file contents: '${AILA_PUBLIC_API_VERSION}'")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${AILA_REPO_ROOT}/VERSION.txt")
string(REPLACE "." "\\." AILA_VERSION_REGEX "${AILA_PUBLIC_API_VERSION}")

set(AILA_WORKER_BUILD_ID "" CACHE STRING "Exact build identity of the external Worker")
if(NOT AILA_WORKER_BUILD_ID MATCHES "^aila-${AILA_VERSION_REGEX}-abi1-([0-9a-f]+)$")
    message(FATAL_ERROR "AILA_WORKER_BUILD_ID must match aila-${AILA_PUBLIC_API_VERSION}-abi1-<40-hex-commit>.")
endif()
string(LENGTH "${CMAKE_MATCH_1}" AILA_WORKER_COMMIT_LENGTH)
if(NOT AILA_WORKER_COMMIT_LENGTH EQUAL 40)
    message(FATAL_ERROR "AILA_WORKER_BUILD_ID must contain the full 40-character Worker commit.")
endif()

set(AILA_FAKE_WORKER_PATH "" CACHE FILEPATH "External AilaFakeInferenceWorker.exe for pipe tests")
if(NOT IS_ABSOLUTE "${AILA_FAKE_WORKER_PATH}" OR
   NOT EXISTS "${AILA_FAKE_WORKER_PATH}" OR IS_DIRECTORY "${AILA_FAKE_WORKER_PATH}")
    message(FATAL_ERROR "AILA_FAKE_WORKER_PATH must name an existing absolute Worker executable.")
endif()
if(AILA_FAKE_WORKER_PATH MATCHES "[\";\r\n]")
    message(FATAL_ERROR "AILA_FAKE_WORKER_PATH cannot contain quotes, semicolons or line breaks.")
endif()
file(TO_CMAKE_PATH "${AILA_FAKE_WORKER_PATH}" AILA_FAKE_WORKER_PATH)
find_package(Threads REQUIRED)
find_package(jsoncpp 1.9.6 EXACT CONFIG REQUIRED)
find_package(ZLIB 1.3.1 EXACT REQUIRED)
find_package(Trantor 1.5.24 EXACT CONFIG REQUIRED)
find_package(Drogon 1.9.11 EXACT CONFIG REQUIRED)
# JsonCpp's namespace target is an imported interface, not a static-library
# alias. Check its concrete target as well as the public forwarding interface.
if(NOT TARGET JsonCpp::JsonCpp)
    message(FATAL_ERROR "Missing pinned dependency target: JsonCpp::JsonCpp")
endif()
get_target_property(jsoncpp_link JsonCpp::JsonCpp INTERFACE_LINK_LIBRARIES)
if(NOT jsoncpp_link STREQUAL "jsoncpp_static")
    message(FATAL_ERROR "JsonCpp::JsonCpp must forward to the pinned jsoncpp_static target.")
endif()
foreach(dependency IN ITEMS jsoncpp_static Trantor::Trantor Drogon::Drogon)
    if(NOT TARGET ${dependency})
        message(FATAL_ERROR "Missing pinned dependency target: ${dependency}")
    endif()
    get_target_property(dependency_type ${dependency} TYPE)
    if(NOT dependency_type STREQUAL "STATIC_LIBRARY")
        message(FATAL_ERROR "The server requires static ${dependency} with the /MD runtime.")
    endif()
endforeach()

add_library(AilaServerBuildOptions INTERFACE)
target_compile_features(AilaServerBuildOptions INTERFACE cxx_std_17)
target_compile_definitions(AilaServerBuildOptions INTERFACE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(AilaServerBuildOptions INTERFACE /W4 /utf-8 /EHsc /fp:precise /bigobj)
if(DEFINED AILA_SERVER_MSVC_DEPS_PREFIX_SHA256)
    # Ninja does not include msvc_deps_prefix in its command hash. Changing a
    # bad prefix alone leaves existing zero-dependency objects "up to date".
    # Put the measured prefix in each command's identity to rebuild those
    # objects once, and again if the compiler's output locale ever changes.
    target_compile_definitions(AilaServerBuildOptions INTERFACE
        AILA_SERVER_MSVC_DEPS_PREFIX_SHA256="${AILA_SERVER_MSVC_DEPS_PREFIX_SHA256}")
endif()

add_library(AilaServerSimdjson STATIC "${AILA_REPO_ROOT}/third_party/simdjson/simdjson.cpp")
target_include_directories(AilaServerSimdjson SYSTEM PUBLIC "${AILA_REPO_ROOT}/third_party/simdjson")
target_link_libraries(AilaServerSimdjson PUBLIC AilaServerBuildOptions)

add_library(AilaServerProtocol STATIC
    "${AILA_REPO_ROOT}/src/ipc/IpcProtocol.cpp"
    "${AILA_REPO_ROOT}/src/ipc/InferenceProtocol.cpp")
target_include_directories(AilaServerProtocol PUBLIC
    "${AILA_REPO_ROOT}/include"
    "${AILA_REPO_ROOT}/src")
target_link_libraries(AilaServerProtocol PUBLIC AilaServerSimdjson)

add_library(AilaServerClient STATIC
    "${AILA_REPO_ROOT}/src/runtime/InferenceClient.cpp"
    "${AILA_REPO_ROOT}/src/runtime/InferencePipeTransport.cpp"
    "${AILA_REPO_ROOT}/src/runtime/RuntimeDirectory.cpp"
    "${AILA_REPO_ROOT}/src/runtime/ChildEnvironment.cpp"
    "${AILA_REPO_ROOT}/src/ipc/Win32Pipe.cpp")
target_link_libraries(AilaServerClient PUBLIC AilaServerProtocol Threads::Threads)

add_library(AilaServerChat STATIC
    "${AILA_REPO_ROOT}/src/server/ChatCodec.cpp"
    "${AILA_REPO_ROOT}/src/utils/Tokenizer.cpp"
    "${AILA_REPO_ROOT}/src/profile/Profiling.cpp"
    "${AILA_REPO_ROOT}/src/chat/AssistantOutputParser.cpp"
    "${AILA_REPO_ROOT}/src/chat/BuiltinTemplates.cpp"
    "${AILA_REPO_ROOT}/src/chat/ChatFormatter.cpp"
    "${AILA_REPO_ROOT}/src/chat/ChatJson.cpp"
    "${AILA_REPO_ROOT}/src/chat/ChatStreamJson.cpp"
    "${AILA_REPO_ROOT}/src/chat/ChatSessionState.cpp"
    "${AILA_REPO_ROOT}/src/chat/ChatTemplateEngine.cpp"
    "${AILA_REPO_ROOT}/src/chat/StructuredStreamParser.cpp"
    "${AILA_REPO_ROOT}/src/chat/ToolPolicy.cpp"
    "${AILA_REPO_ROOT}/src/templates/jinja/lexer.cpp"
    "${AILA_REPO_ROOT}/src/templates/jinja/parser.cpp"
    "${AILA_REPO_ROOT}/src/templates/jinja/runtime.cpp"
    "${AILA_REPO_ROOT}/src/templates/jinja/jinja_string.cpp"
    "${AILA_REPO_ROOT}/src/templates/jinja/jinja_unicode.cpp"
    "${AILA_REPO_ROOT}/src/templates/jinja/value.cpp")
target_include_directories(AilaServerChat PRIVATE "${AILA_REPO_ROOT}/src/templates/jinja")
target_link_libraries(AilaServerChat PUBLIC AilaServerClient JsonCpp::JsonCpp)

add_executable(AilaServer
    "${AILA_REPO_ROOT}/src/server/ServerMain.cpp"
    "${AILA_REPO_ROOT}/src/server/HttpServer.cpp")
target_compile_definitions(AilaServer PRIVATE AILA_WORKER_BUILD_ID="${AILA_WORKER_BUILD_ID}")
target_link_libraries(AilaServer PRIVATE AilaServerChat Drogon::Drogon)

enable_testing()
add_executable(AilaServerChannelTests "${AILA_REPO_ROOT}/tests/server/ChannelTests.cpp")
target_link_libraries(AilaServerChannelTests PRIVATE AilaServerBuildOptions Trantor::Trantor)
add_test(NAME AilaServerChannelTests COMMAND AilaServerChannelTests)
set_tests_properties(AilaServerChannelTests PROPERTIES LABELS "cpu;server;transport" TIMEOUT 30)

add_executable(AilaServerChatTests "${AILA_REPO_ROOT}/tests/server/ChatCodecTests.cpp")
target_link_libraries(AilaServerChatTests PRIVATE AilaServerChat)
add_test(NAME AilaServerChatTests COMMAND AilaServerChatTests)
set_tests_properties(AilaServerChatTests PROPERTIES LABELS "cpu;server;chat" TIMEOUT 60)

add_executable(AilaServerClientTests "${AILA_REPO_ROOT}/tests/runtime/InferenceClientTests.cpp")
target_link_libraries(AilaServerClientTests PRIVATE AilaServerClient)
add_test(NAME AilaServerClientTests COMMAND AilaServerClientTests)
set_tests_properties(AilaServerClientTests PROPERTIES LABELS "cpu;server;ipc" TIMEOUT 90)

add_executable(AilaServerPipeTests "${AILA_REPO_ROOT}/tests/runtime/InferencePipeTests.cpp")
target_compile_definitions(AilaServerPipeTests PRIVATE
    AILA_BUILD_ID="${AILA_WORKER_BUILD_ID}"
    AILA_FAKE_INFERENCE_WORKER_PATH="${AILA_FAKE_WORKER_PATH}")
target_link_libraries(AilaServerPipeTests PRIVATE AilaServerClient)
add_test(NAME AilaServerPipeTests COMMAND AilaServerPipeTests)
set_tests_properties(AilaServerPipeTests PROPERTIES LABELS "cpu;server;ipc" TIMEOUT 90)

message(STATUS "AilaServer external Worker identity: ${AILA_WORKER_BUILD_ID}")
message(STATUS "AilaServer fake Worker: ${AILA_FAKE_WORKER_PATH}")
