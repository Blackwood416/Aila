# required cmake version
cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)

if(WIN32)
        set(CMAKE_CXX_COMPILER "icx-cl")
else()
        set(CMAKE_CXX_COMPILER "icpx")
endif()

# Set default build type to Release before project() initializes the toolchain.
if (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release")
    set (CMAKE_BUILD_TYPE "Release" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project (Aila CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/AilaRuntimeDllSelection.cmake")

option(AILA_ALLOW_LEGACY_ONEAPI_BASELINE
    "Allow the temporary oneAPI 2025.3 comparison build" OFF)

set(AILA_PUBLIC_API_VERSION "0.2.0")
set(AILA_PUBLIC_ABI_VERSION "1")
find_package(Git QUIET)

function(aila_register_git_configure_dependency git_path_name)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --git-path "${git_path_name}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE path_result
        OUTPUT_VARIABLE metadata_path
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT path_result EQUAL 0 OR metadata_path STREQUAL "")
        return()
    endif()

    cmake_path(SET metadata_path NORMALIZE "${metadata_path}")
    if(NOT IS_ABSOLUTE "${metadata_path}")
        cmake_path(
            ABSOLUTE_PATH metadata_path
            BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            NORMALIZE
        )
    endif()
    if(EXISTS "${metadata_path}")
        set_property(DIRECTORY APPEND PROPERTY
            CMAKE_CONFIGURE_DEPENDS "${metadata_path}")
    endif()
endfunction()

if(Git_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE AILA_GIT_REVISION_RESULT
        OUTPUT_VARIABLE AILA_GIT_REVISION
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    aila_register_git_configure_dependency(HEAD)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" symbolic-ref -q HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE AILA_GIT_SYMBOLIC_REF_RESULT
        OUTPUT_VARIABLE AILA_GIT_SYMBOLIC_REF
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(AILA_GIT_SYMBOLIC_REF_RESULT EQUAL 0)
        aila_register_git_configure_dependency("${AILA_GIT_SYMBOLIC_REF}")
    endif()
    aila_register_git_configure_dependency(packed-refs)
endif()
string(LENGTH "${AILA_GIT_REVISION}" AILA_GIT_REVISION_LENGTH)
if(NOT Git_FOUND OR
   NOT AILA_GIT_REVISION_RESULT EQUAL 0 OR
   NOT AILA_GIT_REVISION MATCHES "^[0-9a-fA-F]+$" OR
   NOT AILA_GIT_REVISION_LENGTH EQUAL 40)
    set(AILA_GIT_REVISION "git-unavailable")
    message(WARNING
        "Git revision is unavailable; using deterministic AILA_BUILD_ID fallback")
endif()
set(AILA_BUILD_ID
    "aila-${AILA_PUBLIC_API_VERSION}-abi${AILA_PUBLIC_ABI_VERSION}-${AILA_GIT_REVISION}")
add_library(AilaBuildIdentity INTERFACE)
target_compile_definitions(AilaBuildIdentity INTERFACE
    "AILA_BUILD_ID=\"${AILA_BUILD_ID}\"")

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message(FATAL_ERROR "Aila requires the Intel oneAPI DPC++/C++ Compiler")
endif()

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "2026.1" AND
   NOT AILA_ALLOW_LEGACY_ONEAPI_BASELINE)
    message(FATAL_ERROR
        "Aila requires IntelLLVM >= 2026.1; use "
        "-DAILA_ALLOW_LEGACY_ONEAPI_BASELINE=ON only for the controlled 2025.3 comparison build")
endif()

if(MSVC)
    # --- Base flags ---
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math /Qstd=c++17 /EHsc /Qftz")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -std=c++17")
endif()

add_library(AilaSyclOptions INTERFACE)
target_compile_options(AilaSyclOptions INTERFACE
    -fsycl
    -fsycl-device-code-split=per_kernel
    $<$<BOOL:${MSVC}>:-fsycl-default-sub-group-size=16>
)
target_link_options(AilaSyclOptions INTERFACE -fsycl)
# Enable >4GB device allocations on Intel GPUs (passed through to IGC via -Xs)
# NOTE: regresses decode ~10% on Arc A770.  Keep for reference; enable only if
# >4GB buffers are actually needed (e.g. >13B models on 16GB GPUs).
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Xs -ze-intel-greater-than-4GB-buffer-required")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Xs -ze-intel-greater-than-4GB-buffer-required")

# ============================================================
# Source file lists
# ============================================================

set(CHAT_SOURCES
    src/chat/AssistantOutputParser.cpp
    src/chat/BuiltinTemplates.cpp
    src/chat/ChatFormatter.cpp
    src/chat/ChatJson.cpp
    src/chat/ChatStreamJson.cpp
    src/chat/ChatSessionState.cpp
    src/chat/ChatTemplateEngine.cpp
    src/chat/StructuredStreamParser.cpp
    src/chat/ToolPolicy.cpp
)

set(JINJA_SOURCES
    src/templates/jinja/lexer.cpp
    src/templates/jinja/parser.cpp
    src/templates/jinja/runtime.cpp
    src/templates/jinja/jinja_string.cpp
    src/templates/jinja/jinja_unicode.cpp
    src/templates/jinja/value.cpp
)

# Core library sources (everything except main.cpp and CLI)
set(LIB_SOURCES
    ${CHAT_SOURCES}
    src/ops/Linear.cpp
    src/ops/Bnb4BitLinear.cpp
    src/ops/NormOps.cpp
    src/ops/AttentionOps.cpp
    src/ops/ElementwiseOps.cpp
    src/ops/VisionOps.cpp
    src/ops/SamplingOps.cpp
    src/ops/Transpose.cpp
    src/ops/ConvOps.cpp
    src/memory/KVCache.cpp
    src/models/Qwen3DenseBackend.cpp
    src/models/Qwen3Bnb4Backend.cpp
    src/models/Qwen35HybridTextBackend.cpp
    src/models/Qwen35HybridBnb4Backend.cpp
    src/models/Qwen3ASRBackend.cpp
    src/models/Qwen3ASRBnb4Backend.cpp
    src/models/Qwen3ForceAlignerBackend.cpp
    src/models/Qwen3ForceAlignerBnb4Backend.cpp
    src/models/Qwen3TTSBackend.cpp
    src/utils/Tokenizer.cpp
    src/utils/ModelConfig.cpp
    src/utils/ModelSpec.cpp
    src/utils/MemoryMappedFile.cpp
    src/utils/ForceAlignerPostProcess.cpp
    src/utils/SafeTensors.cpp
    src/utils/Base64.cpp
    src/utils/JsonParser.cpp
    ${JINJA_SOURCES}
    src/vision/Qwen35VisionEncoder.cpp
    src/vision/Yolo26Detector.cpp
    src/vision/DetectionRenderer.cpp
    src/audio/Qwen3ASRAudioEncoder.cpp
    src/audio/AudioPreprocessor.cpp
    src/audio/SpeakerEncoder.cpp
    src/audio/GpuSpeakerEncoder.cpp
    src/audio/MimiEncoder.cpp
    src/profile/Device.cpp
    src/profile/Profiling.cpp
    src/api/aila_api.cpp
    src/lora/LoraLoader.cpp
    src/bench/Benchmark.cpp
)

# CLI-only sources
set(CLI_SOURCES
    src/cli/CLI.cpp
)

# Lightweight chat tests
set(CHAT_TEST_SOURCES
    ${CHAT_SOURCES}
    ${JINJA_SOURCES}
    tests/chat/ChatTestMain.cpp
    tests/chat/AssistantOutputParserTests.cpp
    tests/chat/ChatJsonTests.cpp
    tests/chat/ChatStreamJsonTests.cpp
    tests/chat/ChatTemplateEngineTests.cpp
    tests/chat/ChatFormatterTests.cpp
    tests/chat/ThinkingBudgetControllerTests.cpp
    tests/chat/StructuredStreamParserTests.cpp
    tests/chat/ToolPolicyTests.cpp
)

# Third-party
set(SIMDJSON_SOURCES
    third_party/simdjson/simdjson.cpp
)

# ============================================================
# Include directories
# ============================================================
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src/utils)
include_directories(${PROJECT_SOURCE_DIR}/src/core)
include_directories(${PROJECT_SOURCE_DIR}/src/ops)
include_directories(${PROJECT_SOURCE_DIR}/src/memory)
include_directories(${PROJECT_SOURCE_DIR}/src/models)
include_directories(${PROJECT_SOURCE_DIR}/src/audio)
include_directories(${PROJECT_SOURCE_DIR}/src/bench)
include_directories(${PROJECT_SOURCE_DIR}/src/lora)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/third_party/dr_libs)
include_directories(${PROJECT_SOURCE_DIR}/src/templates/jinja)

# ============================================================
# Dependencies
# ============================================================
find_package(dnnl REQUIRED)
add_library(simdjson STATIC ${SIMDJSON_SOURCES})
target_include_directories(simdjson PUBLIC ${PROJECT_SOURCE_DIR}/third_party/simdjson)
if(WIN32)
    get_filename_component(AILA_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(AILA_COMPILER_ROOT
        "${AILA_COMPILER_BIN_DIR}/.." ABSOLUTE)
    add_library(simdjson_proxy STATIC ${SIMDJSON_SOURCES})
    target_compile_options(simdjson_proxy PRIVATE
        -fno-sycl
        -Wno-unused-command-line-argument)
    set_target_properties(simdjson_proxy PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_include_directories(simdjson_proxy PUBLIC
        ${PROJECT_SOURCE_DIR}/third_party/simdjson)
endif()

# ============================================================
# Static Library: AilaLib
# ============================================================
add_library(AilaLib STATIC ${LIB_SOURCES})
target_compile_definitions(AilaLib PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_link_libraries(AilaLib PUBLIC DNNL::dnnl simdjson AilaSyclOptions)
if(WIN32)
    target_link_libraries(AilaLib PUBLIC windowscodecs ole32)
endif()
target_include_directories(AilaLib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# ============================================================
# Shared Library: AilaShared
# ============================================================
if(WIN32)
    add_library(AilaShared SHARED
        src/proxy/AilaProxyApi.cpp
        src/proxy/ProxyEngine.cpp
        src/proxy/ProxyLogging.cpp
        src/runtime/RuntimeDirectory.cpp
        src/runtime/ChildEnvironment.cpp
        src/runtime/WorkerProcess.cpp
        src/ipc/IpcProtocol.cpp
        src/ipc/Win32Pipe.cpp
    )
    target_compile_options(AilaShared PRIVATE
        -fno-sycl
        -Wno-unused-command-line-argument)
    target_link_options(AilaShared PRIVATE -fno-sycl)
    set_target_properties(AilaShared PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_link_directories(AilaShared PRIVATE "${AILA_COMPILER_ROOT}/lib")
    target_link_libraries(AilaShared PRIVATE simdjson_proxy AilaBuildIdentity)
else()
    add_library(AilaShared SHARED ${LIB_SOURCES})
    target_link_libraries(AilaShared PUBLIC DNNL::dnnl simdjson AilaSyclOptions)
endif()
target_compile_definitions(AilaShared PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_BUILDING_DLL)
target_include_directories(AilaShared PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)
set_target_properties(AilaShared PROPERTIES OUTPUT_NAME "AilaShared")

# ============================================================
# Executable: Aila
# ============================================================
add_executable(Aila src/main.cpp ${CLI_SOURCES})
target_compile_definitions(Aila PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_link_libraries(Aila PRIVATE AilaLib)

# ============================================================
# Executable: AilaChatTests
# ============================================================
add_executable(AilaChatTests ${CHAT_TEST_SOURCES})
target_compile_definitions(AilaChatTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaChatTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/chat
)
target_link_libraries(AilaChatTests PRIVATE simdjson)
enable_testing()
add_test(NAME AilaChatTests COMMAND AilaChatTests)
add_test(
    NAME AilaRuntimeDllSelectionCMakeTests
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/RuntimeDllSelectionTests.cmake"
)

add_executable(AilaBnb4BitLinearKernelTests tests/ops/Bnb4BitLinearKernelTests.cpp)
target_compile_definitions(AilaBnb4BitLinearKernelTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaBnb4BitLinearKernelTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaBnb4BitLinearKernelTests PRIVATE
    DNNL::dnnl simdjson AilaSyclOptions)
add_test(NAME AilaBnb4BitLinearKernelTests COMMAND AilaBnb4BitLinearKernelTests)

add_executable(AilaConvOpsKernelTests tests/ops/ConvOpsKernelTests.cpp)
target_compile_definitions(AilaConvOpsKernelTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaConvOpsKernelTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaConvOpsKernelTests PRIVATE
    DNNL::dnnl simdjson AilaSyclOptions)
add_test(NAME AilaConvOpsKernelTests COMMAND AilaConvOpsKernelTests)

add_executable(AilaYolo26InferRefAdapter tests/inferref/Yolo26InferRefAdapter.cpp)
target_compile_definitions(AilaYolo26InferRefAdapter PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaYolo26InferRefAdapter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaYolo26InferRefAdapter PRIVATE AilaLib)

add_executable(AilaYolo26ModelSpecTests
    tests/vision/Yolo26ModelSpecTests.cpp
    src/utils/ModelSpec.cpp
)
target_compile_definitions(AilaYolo26ModelSpecTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaYolo26ModelSpecTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaYolo26ModelSpecTests PRIVATE simdjson)
add_test(NAME AilaYolo26ModelSpecTests COMMAND AilaYolo26ModelSpecTests)

add_executable(AilaEnvUtilsTests tests/utils/EnvUtilsTests.cpp)
target_compile_definitions(AilaEnvUtilsTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaEnvUtilsTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaEnvUtilsTests PRIVATE AilaLib)
add_test(NAME AilaEnvUtilsTests COMMAND AilaEnvUtilsTests)

add_executable(AilaSafeTensorsTests tests/utils/SafeTensorsTests.cpp)
target_compile_definitions(AilaSafeTensorsTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaSafeTensorsTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaSafeTensorsTests PRIVATE AilaLib)
add_test(NAME AilaSafeTensorsTests COMMAND AilaSafeTensorsTests)

add_executable(AilaMimiEncoderTests tests/audio/MimiEncoderTests.cpp)
target_compile_definitions(AilaMimiEncoderTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaMimiEncoderTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaMimiEncoderTests PRIVATE AilaLib simdjson)
add_test(NAME AilaMimiEncoderTests COMMAND AilaMimiEncoderTests WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(AilaLoraLoaderTests tests/lora/LoraLoaderTests.cpp)
target_compile_definitions(AilaLoraLoaderTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaLoraLoaderTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaLoraLoaderTests PRIVATE AilaLib)
add_test(NAME AilaLoraLoaderTests COMMAND AilaLoraLoaderTests)

add_executable(AilaQwen35VisionWeightsTests tests/vision/Qwen35VisionWeightsTests.cpp)
target_compile_definitions(AilaQwen35VisionWeightsTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaQwen35VisionWeightsTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
add_test(NAME AilaQwen35VisionWeightsTests COMMAND AilaQwen35VisionWeightsTests)

add_executable(AilaIpcProtocolTests
    tests/runtime/IpcProtocolTests.cpp
    src/ipc/IpcProtocol.cpp
)
if(WIN32)
    target_sources(AilaIpcProtocolTests PRIVATE src/ipc/Win32Pipe.cpp)
endif()
target_compile_definitions(AilaIpcProtocolTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaIpcProtocolTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaIpcProtocolTests PRIVATE simdjson)
add_test(NAME AilaIpcProtocolTests COMMAND AilaIpcProtocolTests)

if(WIN32)
    get_filename_component(AILA_WORKER_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(AILA_WORKER_COMPILER_ROOT
        "${AILA_WORKER_COMPILER_BIN_DIR}/.." ABSOLUTE)

    add_executable(AilaWorker
        src/worker/WorkerMain.cpp
        src/worker/WorkerDispatcher.cpp
        src/ipc/IpcProtocol.cpp
        src/ipc/Win32Pipe.cpp
    )
    target_compile_definitions(AilaWorker PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        AILA_STATIC_LIB
    )
    target_include_directories(AilaWorker PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_directories(AilaWorker PRIVATE
        "${AILA_WORKER_COMPILER_ROOT}/lib")
    target_link_libraries(AilaWorker PRIVATE AilaLib simdjson AilaBuildIdentity)
    if(CMAKE_GENERATOR MATCHES "Ninja" AND Git_FOUND)
        add_test(
            NAME AilaBuildIdentityCMakeTests
            COMMAND ${CMAKE_COMMAND}
                "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
                "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
                "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
                "-DWORKER_FILE=$<TARGET_FILE:AilaWorker>"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/BuildIdentityCMakeTests.cmake"
        )
    endif()

    add_executable(AilaWorkerDispatcherTests
        tests/runtime/WorkerDispatcherTests.cpp
        src/worker/WorkerDispatcher.cpp
        src/ipc/IpcProtocol.cpp
    )
    target_compile_definitions(AilaWorkerDispatcherTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(AilaWorkerDispatcherTests PRIVATE
        -fno-sycl
        -Wno-unused-command-line-argument)
    target_link_options(AilaWorkerDispatcherTests PRIVATE -fno-sycl)
    target_include_directories(AilaWorkerDispatcherTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_directories(AilaWorkerDispatcherTests PRIVATE
        "${AILA_WORKER_COMPILER_ROOT}/lib")
    target_link_libraries(AilaWorkerDispatcherTests PRIVATE simdjson)
    add_test(NAME AilaWorkerDispatcherTests COMMAND AilaWorkerDispatcherTests)

    add_executable(AilaFakeWorker
        tests/runtime/FakeWorkerMain.cpp
        src/ipc/IpcProtocol.cpp
        src/ipc/Win32Pipe.cpp
    )
    target_compile_definitions(AilaFakeWorker PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(AilaFakeWorker PRIVATE
        -fno-sycl
        -Wno-unused-command-line-argument)
    target_link_options(AilaFakeWorker PRIVATE -fno-sycl)
    target_include_directories(AilaFakeWorker PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    get_filename_component(AILA_FAKE_WORKER_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(AILA_FAKE_WORKER_COMPILER_ROOT
        "${AILA_FAKE_WORKER_COMPILER_BIN_DIR}/.." ABSOLUTE)
    target_link_directories(AilaFakeWorker PRIVATE
        "${AILA_FAKE_WORKER_COMPILER_ROOT}/lib")
    target_link_libraries(AilaFakeWorker PRIVATE simdjson)

    add_executable(AilaRuntimeIsolationTests
        tests/runtime/RuntimeIsolationTests.cpp
        src/runtime/RuntimeDirectory.cpp
        src/runtime/ChildEnvironment.cpp
        src/runtime/WorkerProcess.cpp
        src/ipc/IpcProtocol.cpp
        src/ipc/Win32Pipe.cpp
    )
    target_compile_definitions(AilaRuntimeIsolationTests PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        AILA_FAKE_WORKER_PATH=\"$<TARGET_FILE:AilaFakeWorker>\"
    )
    target_include_directories(AilaRuntimeIsolationTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_directories(AilaRuntimeIsolationTests PRIVATE
        "${AILA_FAKE_WORKER_COMPILER_ROOT}/lib")
    target_link_libraries(AilaRuntimeIsolationTests PRIVATE simdjson)
    add_dependencies(AilaRuntimeIsolationTests AilaFakeWorker)
    add_test(NAME AilaRuntimeIsolationTests COMMAND AilaRuntimeIsolationTests)

    add_executable(AilaProxyAbiTests tests/runtime/ProxyAbiTests.cpp)
    target_compile_definitions(AilaProxyAbiTests PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        AILA_STATIC_LIB
        AILA_PROXY_DLL_PATH="$<TARGET_FILE:AilaShared>"
        AILA_FAKE_WORKER_PATH="$<TARGET_FILE:AilaFakeWorker>"
    )
    target_compile_options(AilaRuntimeIsolationTests PRIVATE
        -fno-sycl
        -Wno-unused-command-line-argument)
    target_link_options(AilaRuntimeIsolationTests PRIVATE -fno-sycl)
    target_compile_options(AilaProxyAbiTests PRIVATE
        -fno-sycl
        -Wno-unused-command-line-argument)
    target_link_options(AilaProxyAbiTests PRIVATE -fno-sycl)
    set_target_properties(AilaProxyAbiTests PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_include_directories(AilaProxyAbiTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(AilaProxyAbiTests PRIVATE AilaBuildIdentity)
    add_dependencies(AilaProxyAbiTests AilaShared AilaFakeWorker)
    add_test(NAME AilaProxyAbiTests COMMAND AilaProxyAbiTests)

    add_executable(AilaProxyLoggingTests
        tests/runtime/ProxyLoggingTests.cpp
        src/proxy/ProxyLogging.cpp)
    target_compile_definitions(AilaProxyLoggingTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
    target_compile_options(AilaProxyLoggingTests PRIVATE
        -fno-sycl
        -Wno-unused-command-line-argument)
    target_link_options(AilaProxyLoggingTests PRIVATE -fno-sycl)
    set_target_properties(AilaProxyLoggingTests PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_include_directories(AilaProxyLoggingTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_directories(AilaProxyLoggingTests PRIVATE
        "${AILA_WORKER_COMPILER_ROOT}/lib")
    add_test(NAME AilaProxyLoggingTests COMMAND AilaProxyLoggingTests)
endif()

# ============================================================
# Post-build: Copy runtime DLL dependencies
# ============================================================
if(WIN32)
    get_target_property(DNNL_DLL_LOCATION DNNL::dnnl IMPORTED_LOCATION_RELEASE)
    if(NOT DNNL_DLL_LOCATION)
        get_target_property(DNNL_DLL_LOCATION DNNL::dnnl IMPORTED_LOCATION)
    endif()

    get_filename_component(AILA_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(AILA_COMPILER_VERSION_ROOT "${AILA_COMPILER_BIN_DIR}/.." ABSOLUTE)
    get_filename_component(AILA_COMPILER_VERSION "${AILA_COMPILER_VERSION_ROOT}" NAME)
    get_filename_component(AILA_COMPILER_COMPONENT_ROOT "${AILA_COMPILER_VERSION_ROOT}/.." ABSOLUTE)
    get_filename_component(AILA_ONEAPI_INSTALL_ROOT "${AILA_COMPILER_COMPONENT_ROOT}/.." ABSOLUTE)
    set(AILA_ONEAPI_SHARED_BIN_DIR "${AILA_ONEAPI_INSTALL_ROOT}/${AILA_COMPILER_VERSION}/bin")
    file(GLOB AILA_ONEAPI_SHARED_HWLOC_DLLS LIST_DIRECTORIES FALSE
        "${AILA_ONEAPI_SHARED_BIN_DIR}/libhwloc-*.dll")
    if(NOT IS_DIRECTORY "${AILA_ONEAPI_SHARED_BIN_DIR}" OR NOT AILA_ONEAPI_SHARED_HWLOC_DLLS)
        message(FATAL_ERROR
            "Resolved oneAPI shared runtime directory is invalid: ${AILA_ONEAPI_SHARED_BIN_DIR}")
    endif()

    find_package(TBB CONFIG REQUIRED)
    get_filename_component(AILA_TBB_ROOT "${TBB_DIR}/../../.." ABSOLUTE)
    set(AILA_TBB_BIN_DIR "${AILA_TBB_ROOT}/bin")

    if(NOT EXISTS "${AILA_TBB_BIN_DIR}/tbb12.dll")
        message(FATAL_ERROR "Resolved TBB runtime directory is invalid: ${AILA_TBB_BIN_DIR}")
    endif()

    set(AILA_RUNTIME_SOURCE_DIRS
        "${AILA_ONEAPI_SHARED_BIN_DIR}"
        "${AILA_COMPILER_BIN_DIR}"
        "${AILA_TBB_BIN_DIR}"
    )
    if(DNNL_DLL_LOCATION)
        get_filename_component(DNNL_DLL_DIR "${DNNL_DLL_LOCATION}" DIRECTORY)
        list(APPEND AILA_RUNTIME_SOURCE_DIRS "${DNNL_DLL_DIR}")
    endif()

    get_filename_component(AILA_LINKER_BIN_DIR "${CMAKE_LINKER}" DIRECTORY)
    if(EXISTS "${AILA_LINKER_BIN_DIR}/msvcp140.dll")
        get_filename_component(AILA_MSVC_DIR "${AILA_LINKER_BIN_DIR}/../../../../../.." ABSOLUTE)
        get_filename_component(AILA_MSVC_TOOLS_VERSION "${AILA_LINKER_BIN_DIR}/../../.." NAME)
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" AILA_MSVC_VERSION_PREFIX "${AILA_MSVC_TOOLS_VERSION}")

        if(AILA_MSVC_VERSION_PREFIX)
            file(GLOB AILA_MSVC_REDIST_DIRS "${AILA_MSVC_DIR}/Redist/MSVC/${AILA_MSVC_VERSION_PREFIX}*/x64/Microsoft.VC*.CRT")
        endif()
        if(NOT AILA_MSVC_REDIST_DIRS)
            file(GLOB AILA_MSVC_REDIST_DIRS "${AILA_MSVC_DIR}/Redist/MSVC/*/x64/Microsoft.VC*.CRT")
        endif()
        if(AILA_MSVC_REDIST_DIRS)
            list(SORT AILA_MSVC_REDIST_DIRS)
            list(REVERSE AILA_MSVC_REDIST_DIRS)
            list(GET AILA_MSVC_REDIST_DIRS 0 AILA_MSVC_RUNTIME_DIR)
        else()
            set(AILA_MSVC_RUNTIME_DIR "${AILA_LINKER_BIN_DIR}")
        endif()
        list(APPEND AILA_RUNTIME_SOURCE_DIRS "${AILA_MSVC_RUNTIME_DIR}")
    endif()
    list(REMOVE_DUPLICATES AILA_RUNTIME_SOURCE_DIRS)

    aila_select_runtime_dlls(
        OUTPUT_FILES AILA_SELECTED_RUNTIME_DLLS
        DIRECTORIES ${AILA_RUNTIME_SOURCE_DIRS}
    )

    set(AILA_RUNTIME_SEARCH_DIRS)
    set(AILA_RUNTIME_OVERLAY_SOURCE_DLLS)
    set(AILA_RUNTIME_LOOKUP_DIR "${CMAKE_CURRENT_BINARY_DIR}/runtime_dependency_lookup")
    foreach(AILA_RUNTIME_SOURCE_DIR IN LISTS AILA_RUNTIME_SOURCE_DIRS)
        file(GLOB AILA_RUNTIME_SOURCE_DLLS LIST_DIRECTORIES FALSE
            "${AILA_RUNTIME_SOURCE_DIR}/*.dll")
        list(SORT AILA_RUNTIME_SOURCE_DLLS CASE INSENSITIVE)
        set(AILA_RUNTIME_SELECTED_FROM_DIR)
        foreach(AILA_RUNTIME_SOURCE_DLL IN LISTS AILA_RUNTIME_SOURCE_DLLS)
            list(FIND AILA_SELECTED_RUNTIME_DLLS "${AILA_RUNTIME_SOURCE_DLL}" AILA_RUNTIME_SELECTED_INDEX)
            if(NOT AILA_RUNTIME_SELECTED_INDEX EQUAL -1)
                list(APPEND AILA_RUNTIME_SELECTED_FROM_DIR "${AILA_RUNTIME_SOURCE_DLL}")
            endif()
        endforeach()

        list(LENGTH AILA_RUNTIME_SOURCE_DLLS AILA_RUNTIME_SOURCE_COUNT)
        list(LENGTH AILA_RUNTIME_SELECTED_FROM_DIR AILA_RUNTIME_SELECTED_COUNT)
        if(AILA_RUNTIME_SOURCE_COUNT GREATER 0 AND
                AILA_RUNTIME_SELECTED_COUNT EQUAL AILA_RUNTIME_SOURCE_COUNT)
            list(APPEND AILA_RUNTIME_SEARCH_DIRS "${AILA_RUNTIME_SOURCE_DIR}")
        elseif(AILA_RUNTIME_SELECTED_COUNT GREATER 0)
            list(APPEND AILA_RUNTIME_OVERLAY_SOURCE_DLLS ${AILA_RUNTIME_SELECTED_FROM_DIR})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES AILA_RUNTIME_SEARCH_DIRS)
    list(REMOVE_DUPLICATES AILA_RUNTIME_OVERLAY_SOURCE_DLLS)

    if(AILA_RUNTIME_OVERLAY_SOURCE_DLLS)
        set(AILA_RUNTIME_LOOKUP_STAMP "${AILA_RUNTIME_LOOKUP_DIR}/.ready")
        set(AILA_PREPARE_RUNTIME_LOOKUP_TEMPLATE [=[
set(source_dlls [==[@AILA_RUNTIME_OVERLAY_SOURCE_DLLS@]==])
set(lookup_dir [==[@AILA_RUNTIME_LOOKUP_DIR@]==])
set(stamp_path [==[@AILA_RUNTIME_LOOKUP_STAMP@]==])
file(REMOVE_RECURSE "${lookup_dir}")
file(MAKE_DIRECTORY "${lookup_dir}")
foreach(source_dll IN LISTS source_dlls)
    if(NOT EXISTS "${source_dll}")
        message(FATAL_ERROR "Selected runtime DLL no longer exists: ${source_dll}")
    endif()
    get_filename_component(source_name "${source_dll}" NAME)
    execute_process(
        COMMAND "@CMAKE_COMMAND@" -E copy_if_different
            "${source_dll}" "${lookup_dir}/${source_name}"
        COMMAND_ERROR_IS_FATAL ANY
    )
endforeach()
file(WRITE "${stamp_path}" "ready\n")
]=])
        string(CONFIGURE "${AILA_PREPARE_RUNTIME_LOOKUP_TEMPLATE}" AILA_PREPARE_RUNTIME_LOOKUP_SCRIPT @ONLY)
        file(GENERATE
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/prepare_runtime_dependency_lookup.cmake"
            CONTENT "${AILA_PREPARE_RUNTIME_LOOKUP_SCRIPT}"
        )
        add_custom_command(
            OUTPUT "${AILA_RUNTIME_LOOKUP_STAMP}"
            COMMAND "${CMAKE_COMMAND}" -P
                "${CMAKE_CURRENT_BINARY_DIR}/prepare_runtime_dependency_lookup.cmake"
            DEPENDS
                "${CMAKE_CURRENT_BINARY_DIR}/prepare_runtime_dependency_lookup.cmake"
                ${AILA_RUNTIME_OVERLAY_SOURCE_DLLS}
            COMMENT "Preparing conflict-free runtime DLL lookup directory"
            VERBATIM
        )
        add_custom_target(AilaRuntimeDependencyLookup
            DEPENDS "${AILA_RUNTIME_LOOKUP_STAMP}")
        list(APPEND AILA_RUNTIME_SEARCH_DIRS "${AILA_RUNTIME_LOOKUP_DIR}")
    endif()

    set(AILA_ONEAPI_RUNTIME_ROOT_DLLS)
    foreach(AILA_SELECTED_RUNTIME_DLL IN LISTS AILA_SELECTED_RUNTIME_DLLS)
        get_filename_component(AILA_SELECTED_RUNTIME_NAME "${AILA_SELECTED_RUNTIME_DLL}" NAME)
        string(TOLOWER "${AILA_SELECTED_RUNTIME_NAME}" AILA_SELECTED_RUNTIME_KEY)
        if(NOT AILA_SELECTED_RUNTIME_KEY MATCHES
                "^(ur_loader|ur_win_proxy_loader|ur_adapter_opencl|ur_adapter_level_zero|ur_adapter_level_zero_v2|umf|xptifw|sycl-jit|opencl|intelocl64|common_clang64|libmmd.*|libhwloc-.*|tbb12|tbbbind|tbbbind_2_0|tbbbind_2_5|tbbmalloc|tbbmalloc_proxy|tcm|concrt140|msvcp140.*|vccorlib140|vcruntime140.*)\\.dll$")
            continue()
        endif()

        get_filename_component(AILA_SELECTED_RUNTIME_DIR "${AILA_SELECTED_RUNTIME_DLL}" DIRECTORY)
        list(FIND AILA_RUNTIME_SEARCH_DIRS "${AILA_SELECTED_RUNTIME_DIR}" AILA_RUNTIME_DIRECT_DIR_INDEX)
        if(AILA_RUNTIME_DIRECT_DIR_INDEX EQUAL -1)
            list(APPEND AILA_ONEAPI_RUNTIME_ROOT_DLLS
                "${AILA_RUNTIME_LOOKUP_DIR}/${AILA_SELECTED_RUNTIME_NAME}")
        else()
            list(APPEND AILA_ONEAPI_RUNTIME_ROOT_DLLS "${AILA_SELECTED_RUNTIME_DLL}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES AILA_ONEAPI_RUNTIME_ROOT_DLLS)

    set(AILA_COPY_RUNTIME_DEPENDENCIES_TEMPLATE [=[
set(target_path "${TARGET_PATH}")
set(target_dir "${TARGET_DIR}")
set(target_kind "${TARGET_KIND}")

set(search_dirs [==[@AILA_RUNTIME_SEARCH_DIRS@]==])
set(extra_root_dlls [==[@AILA_ONEAPI_RUNTIME_ROOT_DLLS@]==])

set(search_dll_names)
foreach(search_dir IN LISTS search_dirs)
    if(NOT IS_DIRECTORY "${search_dir}")
        message(FATAL_ERROR "Runtime dependency search directory does not exist: ${search_dir}")
    endif()
    file(GLOB search_dlls LIST_DIRECTORIES FALSE "${search_dir}/*.dll")
    foreach(search_dll IN LISTS search_dlls)
        get_filename_component(search_name "${search_dll}" NAME)
        string(TOLOWER "${search_name}" search_key)
        list(FIND search_dll_names "${search_key}" duplicate_index)
        if(NOT duplicate_index EQUAL -1)
            message(FATAL_ERROR
                "Runtime dependency search directories contain duplicate basename '${search_name}'.")
        endif()
        list(APPEND search_dll_names "${search_key}")
    endforeach()
endforeach()

set(common_dependency_args
    DIRECTORIES ${search_dirs}
    PRE_EXCLUDE_REGEXES "api-ms-win-.*" "ext-ms-.*"
    POST_EXCLUDE_REGEXES ".*[Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\][Ss][Yy][Ss][Tt][Ee][Mm]32[/\\].*"
)

set(resolved_deps)
set(unresolved_deps)
if(target_kind STREQUAL "EXECUTABLE")
    file(GET_RUNTIME_DEPENDENCIES
        EXECUTABLES "${target_path}"
        RESOLVED_DEPENDENCIES_VAR target_resolved_deps
        UNRESOLVED_DEPENDENCIES_VAR target_unresolved_deps
        ${common_dependency_args}
    )
elseif(target_kind STREQUAL "LIBRARY")
    file(GET_RUNTIME_DEPENDENCIES
        LIBRARIES "${target_path}"
        RESOLVED_DEPENDENCIES_VAR target_resolved_deps
        UNRESOLVED_DEPENDENCIES_VAR target_unresolved_deps
        ${common_dependency_args}
    )
else()
    message(FATAL_ERROR "Unsupported target kind: ${target_kind}")
endif()
list(APPEND resolved_deps ${target_resolved_deps})
list(APPEND unresolved_deps ${target_unresolved_deps})

if(extra_root_dlls)
    file(GET_RUNTIME_DEPENDENCIES
        LIBRARIES ${extra_root_dlls}
        RESOLVED_DEPENDENCIES_VAR extra_resolved_deps
        UNRESOLVED_DEPENDENCIES_VAR extra_unresolved_deps
        ${common_dependency_args}
    )
    list(APPEND resolved_deps ${extra_root_dlls} ${extra_resolved_deps})
    list(APPEND unresolved_deps ${extra_unresolved_deps})
endif()

list(REMOVE_DUPLICATES resolved_deps)

foreach(dep IN LISTS resolved_deps)
    if(EXISTS "${dep}")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${dep}" "${target_dir}"
            COMMAND_ERROR_IS_FATAL ANY
        )
    endif()
endforeach()

if(unresolved_deps)
    list(SORT unresolved_deps)
    list(REMOVE_DUPLICATES unresolved_deps)
    message(FATAL_ERROR "Unresolved runtime dependencies for ${target_path}: ${unresolved_deps}")
endif()
]=])
    string(CONFIGURE "${AILA_COPY_RUNTIME_DEPENDENCIES_TEMPLATE}" AILA_COPY_RUNTIME_DEPENDENCIES_SCRIPT @ONLY)
    file(GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/copy_runtime_dependencies.cmake"
        CONTENT "${AILA_COPY_RUNTIME_DEPENDENCIES_SCRIPT}"
    )

    function(copy_runtime_dependencies target target_kind)
        if(TARGET AilaRuntimeDependencyLookup)
            add_dependencies(${target} AilaRuntimeDependencyLookup)
        endif()
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                "-DTARGET_PATH=$<TARGET_FILE:${target}>"
                "-DTARGET_DIR=$<TARGET_FILE_DIR:${target}>"
                "-DTARGET_KIND=${target_kind}"
                -P "${CMAKE_CURRENT_BINARY_DIR}/copy_runtime_dependencies.cmake"
            COMMENT "Copying runtime DLLs for ${target}"
            VERBATIM
        )
    endfunction()

    copy_runtime_dependencies(Aila EXECUTABLE)

    set(AILA_RELEASE_BIN_DIR "${CMAKE_BINARY_DIR}/Release/bin")
    set(AILA_RELEASE_RUNTIME_DIR "${AILA_RELEASE_BIN_DIR}/aila_runtime")

    add_custom_target(release
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${AILA_RELEASE_BIN_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${AILA_RELEASE_RUNTIME_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:AilaShared>" "${AILA_RELEASE_BIN_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:Aila>" "${AILA_RELEASE_RUNTIME_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:AilaWorker>" "${AILA_RELEASE_RUNTIME_DIR}"
        COMMAND ${CMAKE_COMMAND}
            "-DTARGET_PATH=$<TARGET_FILE:Aila>"
            "-DTARGET_DIR=${AILA_RELEASE_RUNTIME_DIR}"
            "-DTARGET_KIND=EXECUTABLE"
            -P "${CMAKE_CURRENT_BINARY_DIR}/copy_runtime_dependencies.cmake"
        COMMAND ${CMAKE_COMMAND}
            "-DTARGET_PATH=$<TARGET_FILE:AilaWorker>"
            "-DTARGET_DIR=${AILA_RELEASE_RUNTIME_DIR}"
            "-DTARGET_KIND=EXECUTABLE"
            -P "${CMAKE_CURRENT_BINARY_DIR}/copy_runtime_dependencies.cmake"
        COMMENT "Staging release binaries in ${AILA_RELEASE_BIN_DIR}"
        VERBATIM
    )
    add_dependencies(release Aila AilaShared AilaWorker)
endif()
