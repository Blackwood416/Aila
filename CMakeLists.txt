# required cmake version
cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)

if(WIN32)
        set(CMAKE_CXX_COMPILER "icx-cl")
        set(CMAKE_C_COMPILER "icx-cl")
else()
        set(CMAKE_CXX_COMPILER "icpx")
        set(CMAKE_C_COMPILER "icx")
endif()

# Set default build type to Release before project() initializes the toolchain.
if (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release")
    set (CMAKE_BUILD_TYPE "Release" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project (Aila C CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/AilaRuntimeDllSelection.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/AilaSyclPch.cmake")

option(AILA_ENABLE_SYCL_PCH
    "Enable bundled SYCL precompiled headers for IntelLLVM Ninja builds" ON)
if(AILA_ENABLE_SYCL_PCH AND WIN32 AND MSVC AND CMAKE_GENERATOR MATCHES "Ninja")
    set(CMAKE_DEPFILE_FLAGS_CXX "-QMMD -QMT <OBJECT> -QMF <DEP_FILE>")
endif()

option(AILA_ALLOW_LEGACY_ONEAPI_BASELINE
    "Allow the temporary oneAPI 2025.3 comparison build" OFF)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt" AILA_VERSION_RAW)
string(STRIP "${AILA_VERSION_RAW}" AILA_PUBLIC_API_VERSION)
if(NOT AILA_PUBLIC_API_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "Invalid VERSION.txt file contents: '${AILA_PUBLIC_API_VERSION}'")
endif()
set_property(DIRECTORY APPEND PROPERTY
    CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt")
set(AILA_PUBLIC_ABI_VERSION "1")
add_compile_definitions("AILA_VERSION=\"${AILA_PUBLIC_API_VERSION}\"")
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
    src/engine/AsrText.cpp
    src/engine/EngineLoad.cpp
    src/engine/EngineContext.cpp
    src/engine/EngineGeneration.cpp
    src/engine/EngineAudio.cpp
    src/engine/EngineAudioCache.cpp
    src/engine/EngineBenchmark.cpp
    src/engine/EngineDetection.cpp
    src/ops/Linear.cpp
    src/ops/Bnb4BitLinear.cpp
    src/ops/NormOps.cpp
    src/ops/AttentionOps.cpp
    src/ops/ElementwiseOps.cpp
    src/ops/VisionOps.cpp
    src/ops/SamplingOps.cpp
    src/ops/Transpose.cpp
    src/ops/ConvOps.cpp
    src/ops/GatedDeltaNetOps.cpp
    src/ops/GdnChunkOps.cpp
    src/ops/GdnPrefillVariant.cpp
    src/ops/HcDecodeVariant.cpp
    src/ops/HyperConnectionOps.cpp
    src/ops/MoeExpertOps.cpp
    src/ops/MoeDecodeVariant.cpp
    src/ops/MoeGroupedXmx.cpp
    src/ops/MoeExpertXmx.cpp
    src/ops/Qwen4ExpCpuExpertOps.cpp
    src/ops/QsaIndexerOps.cpp
    src/ops/QsaKvStreamOps.cpp
    src/memory/LargePages.cpp
    src/memory/KVCache.cpp
    src/core/ExecutionTrace.cpp
    src/inference/Model.cpp
    src/inference/Session.cpp
    src/inference/Request.cpp
    src/inference/Scheduler.cpp
    src/models/Qwen3DenseBackend.cpp
    src/models/Qwen3Bnb4Backend.cpp
    src/models/Qwen35HybridTextBackend.cpp
    src/models/Qwen35HybridBnb4Backend.cpp
    src/models/Qwen3ASRBackend.cpp
    src/models/Qwen3ASRBnb4Backend.cpp
    src/models/Qwen3ForceAlignerBackend.cpp
    src/models/Qwen3ForceAlignerBnb4Backend.cpp
    src/models/Qwen3TTSBackend.cpp
    src/models/DiffusionGemmaBnb4Backend.cpp
    src/models/Qwen4ExpPle.cpp
    src/models/Qwen4ExpBnb4Backend.cpp
    src/models/Qwen4ExpDecodeFusion.cpp
    src/models/Qwen4ExpMtpDraft.cpp
    src/models/Qwen4ExpMtpRuntime.cpp
    src/models/Qwen4ExpPrefill.cpp
    src/models/Qwen4ExpWholeTokenGraph.cpp
    src/models/Qwen4ExpExpertExecutor.cpp
    src/models/Qwen4ExpExpertReader.cpp
    src/models/Qwen4ExpHostArena.cpp
    src/models/Qwen4ExpAdaptiveCache.cpp
    src/generation/DiffusionGemmaGenerator.cpp
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
    src/vision/Gemma4VisionEncoder.cpp
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

set(SOXR_SOURCES
    third_party/soxr/soxr.c
    third_party/soxr/data-io.c
    third_party/soxr/dbesi0.c
    third_party/soxr/filter.c
    third_party/soxr/cr.c
    third_party/soxr/cr32.c
    third_party/soxr/cr64.c
    third_party/soxr/fft4g32.c
    third_party/soxr/fft4g64.c
    third_party/soxr/vr32.c
)

# ============================================================
# Include directories
# ============================================================
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/soxr)
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

# Optional Level Zero SDK support for device-polled host USM. The Context API
# fails closed at runtime when this is unavailable.
set(AILA_LEVEL_ZERO_ROOT "" CACHE PATH "Level Zero SDK root")
if(WIN32)
    file(GLOB AILA_LEVEL_ZERO_SDK_CANDIDATES LIST_DIRECTORIES true
        "C:/Program Files/LevelZeroSDK/*")
    list(SORT AILA_LEVEL_ZERO_SDK_CANDIDATES COMPARE NATURAL ORDER DESCENDING)
endif()
find_path(AILA_LEVEL_ZERO_INCLUDE_DIR
    NAMES level_zero/ze_api.h
    HINTS
        "${AILA_LEVEL_ZERO_ROOT}"
        "$ENV{LEVEL_ZERO_ROOT}"
        "$ENV{ZE_ROOT}"
        ${AILA_LEVEL_ZERO_SDK_CANDIDATES}
    PATH_SUFFIXES include
)
find_library(AILA_LEVEL_ZERO_LOADER_LIBRARY
    NAMES ze_loader
    HINTS
        "${AILA_LEVEL_ZERO_ROOT}"
        "$ENV{LEVEL_ZERO_ROOT}"
        "$ENV{ZE_ROOT}"
        ${AILA_LEVEL_ZERO_SDK_CANDIDATES}
    PATH_SUFFIXES lib
)
if(AILA_LEVEL_ZERO_INCLUDE_DIR AND AILA_LEVEL_ZERO_LOADER_LIBRARY)
    add_library(AilaLevelZero INTERFACE)
    target_include_directories(AilaLevelZero INTERFACE
        "${AILA_LEVEL_ZERO_INCLUDE_DIR}")
    target_compile_definitions(AilaLevelZero INTERFACE AILA_HAS_LEVEL_ZERO=1)
    target_link_libraries(AilaLevelZero INTERFACE
        "${AILA_LEVEL_ZERO_LOADER_LIBRARY}")
    message(STATUS "Level Zero SDK: ${AILA_LEVEL_ZERO_LOADER_LIBRARY}")
else()
    message(STATUS
        "Level Zero SDK not found; alloc_host_uncached will fail closed")
endif()

# ============================================================
# Dependencies
# ============================================================
find_package(dnnl REQUIRED)
add_library(simdjson STATIC ${SIMDJSON_SOURCES})
target_include_directories(simdjson PUBLIC ${PROJECT_SOURCE_DIR}/third_party/simdjson)

add_library(soxr STATIC ${SOXR_SOURCES})
target_include_directories(soxr PUBLIC ${PROJECT_SOURCE_DIR}/third_party/soxr)
target_compile_definitions(soxr PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX SOXR_LIB=1)
set_target_properties(soxr PROPERTIES LINKER_LANGUAGE CXX)
if(WIN32)
    target_compile_options(soxr PRIVATE
        -fno-sycl
        -Wno-unused-command-line-argument)
endif()

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
target_link_libraries(AilaLib PUBLIC DNNL::dnnl simdjson soxr AilaSyclOptions)
if(TARGET AilaLevelZero)
    target_link_libraries(AilaLib PUBLIC AilaLevelZero)
endif()
if(WIN32)
    target_link_libraries(AilaLib PUBLIC windowscodecs ole32)
endif()
target_include_directories(AilaLib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

aila_configure_sycl_pch()
set(AILA_LIB_SYCL_PCH_SOURCES ${LIB_SOURCES})
list(REMOVE_ITEM AILA_LIB_SYCL_PCH_SOURCES
    ${CHAT_SOURCES}
    ${JINJA_SOURCES}
    src/engine/AsrText.cpp
    src/ops/Qwen4ExpCpuExpertOps.cpp
    src/memory/LargePages.cpp
    src/core/ExecutionTrace.cpp
    src/inference/Session.cpp
    src/inference/Request.cpp
    src/inference/Scheduler.cpp
    src/models/Qwen4ExpExpertReader.cpp
    src/utils/Tokenizer.cpp
    src/utils/ModelConfig.cpp
    src/utils/ModelSpec.cpp
    src/utils/MemoryMappedFile.cpp
    src/utils/ForceAlignerPostProcess.cpp
    src/utils/Base64.cpp
    src/utils/JsonParser.cpp
    src/vision/DetectionRenderer.cpp
    src/audio/AudioPreprocessor.cpp
    src/profile/Profiling.cpp
)
aila_enable_sycl_pch(AilaLib SOURCES ${AILA_LIB_SYCL_PCH_SOURCES})

# ============================================================
# Shared Library: AilaShared
# ============================================================
if(WIN32)
    add_library(AilaShared SHARED
        src/proxy/AilaProxyApi.cpp
        src/proxy/ProxyEngine.cpp
        src/proxy/ProxyLogging.cpp
        src/proxy/LegacyTextSession.cpp
        src/proxy/LegacyInferenceBridge.cpp
        src/utils/Tokenizer.cpp
        src/utils/ModelSpec.cpp
        src/profile/Profiling.cpp
        ${CHAT_SOURCES}
        ${JINJA_SOURCES}
        src/runtime/RuntimeDirectory.cpp
        src/runtime/ChildEnvironment.cpp
        src/runtime/WorkerProcess.cpp
        src/ipc/IpcProtocol.cpp
        src/ipc/Win32Pipe.cpp
    )
    target_compile_options(AilaShared PRIVATE
        -fno-sycl
        -Wno-unused-command-line-argument
        /fp:precise)
    target_link_options(AilaShared PRIVATE -fno-sycl)
    set_target_properties(AilaShared PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_link_directories(AilaShared PRIVATE "${AILA_COMPILER_ROOT}/lib")
    target_link_libraries(AilaShared PRIVATE simdjson_proxy AilaBuildIdentity)
else()
    add_library(AilaShared SHARED ${LIB_SOURCES})
    target_link_libraries(AilaShared PUBLIC DNNL::dnnl simdjson soxr AilaSyclOptions)
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
aila_enable_sycl_pch(Aila SOURCES src/main.cpp ${CLI_SOURCES})

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

# Request validation must retain NaN/Inf checks despite the kernel-oriented
# global fast-math flags. Keep this override local to the host state machine.
if(MSVC)
    set_property(SOURCE src/inference/Request.cpp APPEND PROPERTY COMPILE_OPTIONS /fp:precise)
else()
    set_property(SOURCE src/inference/Request.cpp APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math)
endif()

find_package(Threads REQUIRED)
target_link_libraries(AilaLib PUBLIC Threads::Threads)
target_link_libraries(AilaShared PRIVATE Threads::Threads)
add_executable(AilaRequestStateTests
    tests/inference/RequestStateTests.cpp
    src/inference/Session.cpp
    src/inference/Request.cpp
    src/profile/Profiling.cpp
)
target_include_directories(AilaRequestStateTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(AilaRequestStateTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(AilaRequestStateTests PRIVATE -fno-sycl)
target_link_options(AilaRequestStateTests PRIVATE -fno-sycl)
if(MSVC)
    target_compile_options(AilaRequestStateTests PRIVATE /fp:precise)
else()
    target_compile_options(AilaRequestStateTests PRIVATE -fno-fast-math)
endif()
target_link_libraries(AilaRequestStateTests PRIVATE Threads::Threads)
add_test(NAME AilaRequestStateTests COMMAND AilaRequestStateTests)
set_tests_properties(AilaRequestStateTests PROPERTIES LABELS "cpu;inference" TIMEOUT 30)

add_executable(AilaSchedulerTests
    tests/inference/SchedulerTests.cpp
    src/inference/Session.cpp
    src/inference/Request.cpp
    src/inference/Scheduler.cpp
    src/profile/Profiling.cpp
)
target_include_directories(AilaSchedulerTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(AilaSchedulerTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(AilaSchedulerTests PRIVATE -fno-sycl)
target_link_options(AilaSchedulerTests PRIVATE -fno-sycl)
if(MSVC)
    target_compile_options(AilaSchedulerTests PRIVATE /fp:precise)
else()
    target_compile_options(AilaSchedulerTests PRIVATE -fno-fast-math)
endif()
target_link_libraries(AilaSchedulerTests PRIVATE Threads::Threads)
add_test(NAME AilaSchedulerTests COMMAND AilaSchedulerTests)
set_tests_properties(AilaSchedulerTests PROPERTIES LABELS "cpu;inference" TIMEOUT 60)

add_executable(AilaSessionBatchTests
    tests/inference/SessionBatchTests.cpp
    src/inference/Session.cpp
    src/inference/Request.cpp
    src/profile/Profiling.cpp)
target_include_directories(AilaSessionBatchTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(AilaSessionBatchTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(AilaSessionBatchTests PRIVATE -fno-sycl)
target_link_options(AilaSessionBatchTests PRIVATE -fno-sycl)
if(MSVC)
    target_compile_options(AilaSessionBatchTests PRIVATE /fp:precise)
else()
    target_compile_options(AilaSessionBatchTests PRIVATE -fno-fast-math)
endif()
target_link_libraries(AilaSessionBatchTests PRIVATE Threads::Threads)
add_test(NAME AilaSessionBatchTests COMMAND AilaSessionBatchTests)
set_tests_properties(AilaSessionBatchTests PROPERTIES LABELS "cpu;inference;batch" TIMEOUT 30)

add_executable(AilaSessionMemoryTests tests/inference/SessionMemoryTests.cpp)
target_include_directories(AilaSessionMemoryTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(AilaSessionMemoryTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(AilaSessionMemoryTests PRIVATE -fno-sycl)
target_link_options(AilaSessionMemoryTests PRIVATE -fno-sycl)
add_test(NAME AilaSessionMemoryTests COMMAND AilaSessionMemoryTests)
set_tests_properties(AilaSessionMemoryTests PROPERTIES LABELS "cpu;inference" TIMEOUT 30)

add_executable(AilaExecutionResourcesTests tests/inference/ExecutionResourcesTests.cpp)
target_compile_definitions(AilaExecutionResourcesTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
if(MSVC)
    target_compile_options(AilaExecutionResourcesTests PRIVATE /fp:precise)
else()
    target_compile_options(AilaExecutionResourcesTests PRIVATE -fno-fast-math)
endif()
target_link_libraries(AilaExecutionResourcesTests PRIVATE AilaLib)
add_test(NAME AilaExecutionResourcesTests COMMAND AilaExecutionResourcesTests)
set_tests_properties(AilaExecutionResourcesTests PROPERTIES
    LABELS "gpu;inference" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu TIMEOUT 300)

add_executable(AilaLoraExecutionTests tests/inference/LoraExecutionTests.cpp)
target_compile_definitions(AilaLoraExecutionTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
if(MSVC)
    target_compile_options(AilaLoraExecutionTests PRIVATE /fp:precise)
else()
    target_compile_options(AilaLoraExecutionTests PRIVATE -fno-fast-math)
endif()
target_link_libraries(AilaLoraExecutionTests PRIVATE AilaLib)
add_test(NAME AilaLoraExecutionFusedTests COMMAND AilaLoraExecutionTests fused)
add_test(NAME AilaLoraExecutionDnnlTests COMMAND AilaLoraExecutionTests dnnl)
set_tests_properties(AilaLoraExecutionFusedTests AilaLoraExecutionDnnlTests PROPERTIES
    LABELS "gpu;inference;lora" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu TIMEOUT 120)

add_executable(AilaLayerReferenceTests tests/inference/LayerReferenceTests.cpp)
target_compile_definitions(AilaLayerReferenceTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
if(MSVC)
    target_compile_options(AilaLayerReferenceTests PRIVATE /fp:precise)
else()
    target_compile_options(AilaLayerReferenceTests PRIVATE -fno-fast-math)
endif()
target_link_libraries(AilaLayerReferenceTests PRIVATE AilaLib)
# Explicit --write/--compare inputs prevent a test from generating its own oracle.
add_test(NAME AilaLayerReferenceTests COMMAND AilaLayerReferenceTests)
set_tests_properties(AilaLayerReferenceTests PROPERTIES
    LABELS "gpu;inference;reference" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu SKIP_RETURN_CODE 77 TIMEOUT 600)

foreach(batchTarget IN ITEMS Qwen3Batch HybridBatch BatchInference DecodeRowProjection)
    add_executable(Aila${batchTarget}Tests tests/inference/${batchTarget}Tests.cpp)
    target_compile_definitions(Aila${batchTarget}Tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
    if(MSVC)
        target_compile_options(Aila${batchTarget}Tests PRIVATE /fp:precise)
    else()
        target_compile_options(Aila${batchTarget}Tests PRIVATE -fno-fast-math)
    endif()
    target_link_libraries(Aila${batchTarget}Tests PRIVATE AilaLib)
    add_test(NAME Aila${batchTarget}Tests COMMAND Aila${batchTarget}Tests)
    set_tests_properties(Aila${batchTarget}Tests PROPERTIES
        LABELS "gpu;inference;batch" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu SKIP_RETURN_CODE 77 TIMEOUT 600)
endforeach()

add_executable(AilaModelAllocationTests tests/inference/ModelAllocationTests.cpp)
target_compile_definitions(AilaModelAllocationTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
if(MSVC)
    target_compile_options(AilaModelAllocationTests PRIVATE /fp:precise)
else()
    target_compile_options(AilaModelAllocationTests PRIVATE -fno-fast-math)
endif()
target_link_libraries(AilaModelAllocationTests PRIVATE AilaLib)
add_test(NAME AilaModelAllocationTests COMMAND AilaModelAllocationTests)
set_tests_properties(AilaModelAllocationTests PROPERTIES
    LABELS "gpu;inference" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu SKIP_RETURN_CODE 77 TIMEOUT 1200)

add_executable(AilaQwen3SessionTests tests/inference/Qwen3SessionTests.cpp)
target_compile_definitions(AilaQwen3SessionTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
if(MSVC)
    target_compile_options(AilaQwen3SessionTests PRIVATE /fp:precise)
else()
    target_compile_options(AilaQwen3SessionTests PRIVATE -fno-fast-math)
endif()
target_link_libraries(AilaQwen3SessionTests PRIVATE AilaLib)
add_test(NAME AilaQwen3SessionTests COMMAND AilaQwen3SessionTests)
set_tests_properties(AilaQwen3SessionTests PROPERTIES
    LABELS "gpu;inference" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu
    SKIP_RETURN_CODE 77 TIMEOUT 600)

add_executable(AilaQwen3SchedulerTests tests/inference/Qwen3SchedulerTests.cpp)
target_compile_definitions(AilaQwen3SchedulerTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
if(MSVC)
    target_compile_options(AilaQwen3SchedulerTests PRIVATE /fp:precise)
else()
    target_compile_options(AilaQwen3SchedulerTests PRIVATE -fno-fast-math)
endif()
target_link_libraries(AilaQwen3SchedulerTests PRIVATE AilaLib)
add_test(NAME AilaQwen3SchedulerTests COMMAND AilaQwen3SchedulerTests)
set_tests_properties(AilaQwen3SchedulerTests PROPERTIES
    LABELS "gpu;inference" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu
    SKIP_RETURN_CODE 77 TIMEOUT 600)

add_executable(AilaQwen35SessionTests tests/inference/Qwen35SessionTests.cpp)
target_compile_definitions(AilaQwen35SessionTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
if(MSVC)
    target_compile_options(AilaQwen35SessionTests PRIVATE /fp:precise)
else()
    target_compile_options(AilaQwen35SessionTests PRIVATE -fno-fast-math)
endif()
target_link_libraries(AilaQwen35SessionTests PRIVATE AilaLib)
add_test(NAME AilaQwen35SessionTests COMMAND AilaQwen35SessionTests)
add_test(NAME AilaQwen35Bf16SessionTests COMMAND AilaQwen35SessionTests --storage bf16)
set_tests_properties(AilaQwen35SessionTests AilaQwen35Bf16SessionTests PROPERTIES
    LABELS "gpu;inference" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu
    SKIP_RETURN_CODE 77 TIMEOUT 600)

add_executable(AilaQwen4ExpSessionTests tests/inference/Qwen4ExpSessionTests.cpp)
target_compile_definitions(AilaQwen4ExpSessionTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_link_libraries(AilaQwen4ExpSessionTests PRIVATE AilaLib)
add_test(NAME AilaQwen4ExpSessionTests COMMAND AilaQwen4ExpSessionTests)
set_tests_properties(AilaQwen4ExpSessionTests PROPERTIES
    LABELS "gpu;inference;qwen4exp" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu
    SKIP_RETURN_CODE 77 TIMEOUT 300)

add_executable(AilaQwen35GroupedBf16Tests tests/inference/Qwen35GroupedBf16Tests.cpp)
target_compile_definitions(AilaQwen35GroupedBf16Tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
if(MSVC)
    target_compile_options(AilaQwen35GroupedBf16Tests PRIVATE /fp:precise)
else()
    target_compile_options(AilaQwen35GroupedBf16Tests PRIVATE -fno-fast-math)
endif()
target_link_libraries(AilaQwen35GroupedBf16Tests PRIVATE AilaLib)

add_executable(AilaDeferredLoggingTests
    tests/inference/DeferredLoggingTests.cpp
    src/profile/Profiling.cpp)
target_include_directories(AilaDeferredLoggingTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(AilaDeferredLoggingTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(AilaDeferredLoggingTests PRIVATE -fno-sycl)
target_link_options(AilaDeferredLoggingTests PRIVATE -fno-sycl)
target_link_libraries(AilaDeferredLoggingTests PRIVATE Threads::Threads)
add_test(NAME AilaDeferredLoggingTests COMMAND AilaDeferredLoggingTests)
set_tests_properties(AilaDeferredLoggingTests PROPERTIES LABELS "cpu;inference" TIMEOUT 30)

find_program(AILA_PWSH_EXECUTABLE NAMES pwsh)
if(AILA_PWSH_EXECUTABLE)
    add_test(NAME AilaQwen35GroupedBf16Tests
        COMMAND "${AILA_PWSH_EXECUTABLE}" -NoProfile -File
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/inference/Qwen35GroupedBf16Tests.ps1"
            -Executable "$<TARGET_FILE:AilaQwen35GroupedBf16Tests>"
            -OutputRoot "${CMAKE_CURRENT_BINARY_DIR}/Testing/Temporary")
    set_tests_properties(AilaQwen35GroupedBf16Tests PROPERTIES
        LABELS "gpu;inference" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu TIMEOUT 600)
    add_test(NAME AilaTokenTraceComparisonTests
        COMMAND "${AILA_PWSH_EXECUTABLE}" -NoProfile -File
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/inference/TokenTraceComparisonTests.ps1")
    set_tests_properties(AilaTokenTraceComparisonTests PROPERTIES LABELS "cpu;inference" TIMEOUT 120)
endif()
add_test(
    NAME AilaRuntimeDllSelectionCMakeTests
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/RuntimeDllSelectionTests.cmake"
)

add_executable(AilaBnb4BitLinearKernelTests
    tests/ops/Bnb4BitLinearKernelTests.cpp
    src/core/ExecutionTrace.cpp
    src/memory/LargePages.cpp)
target_compile_definitions(AilaBnb4BitLinearKernelTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaBnb4BitLinearKernelTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaBnb4BitLinearKernelTests PRIVATE
    DNNL::dnnl simdjson AilaSyclOptions)
add_test(NAME AilaBnb4BitLinearKernelTests COMMAND AilaBnb4BitLinearKernelTests)

add_executable(AilaQsaAttentionProbe
    tests/ops/QsaAttentionProbe.cpp
    src/profile/Profiling.cpp
    src/core/ExecutionTrace.cpp
    src/memory/LargePages.cpp)
target_compile_definitions(AilaQsaAttentionProbe PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaQsaAttentionProbe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQsaAttentionProbe PRIVATE
    DNNL::dnnl simdjson AilaSyclOptions)
add_executable(AilaH2DBandwidthProbe
    tests/ops/H2DBandwidthProbe.cpp
    src/profile/Profiling.cpp
    src/core/ExecutionTrace.cpp
    src/memory/LargePages.cpp)
target_compile_definitions(AilaH2DBandwidthProbe PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaH2DBandwidthProbe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
    ${AILA_LEVEL_ZERO_INCLUDE_DIR}
)
target_link_libraries(AilaH2DBandwidthProbe PRIVATE
    DNNL::dnnl simdjson AilaSyclOptions ${AILA_LEVEL_ZERO_LOADER_LIBRARY})
add_executable(AilaKernelSubmitProbe
    tests/ops/KernelSubmitProbe.cpp
    src/core/ExecutionTrace.cpp
    src/memory/LargePages.cpp
    src/profile/Profiling.cpp)
target_compile_definitions(AilaKernelSubmitProbe PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaKernelSubmitProbe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaKernelSubmitProbe PRIVATE
    DNNL::dnnl simdjson AilaSyclOptions)
add_executable(AilaConvOpsKernelTests
    tests/ops/ConvOpsKernelTests.cpp
    src/core/ExecutionTrace.cpp
    src/memory/LargePages.cpp)
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

add_executable(AilaDiffusionGemmaNf4InferRefAdapter
    tests/inferref/DiffusionGemmaNf4InferRefAdapter.cpp
    src/core/ExecutionTrace.cpp
    src/memory/LargePages.cpp)
target_compile_definitions(AilaDiffusionGemmaNf4InferRefAdapter PRIVATE
    NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaDiffusionGemmaNf4InferRefAdapter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaDiffusionGemmaNf4InferRefAdapter PRIVATE
    DNNL::dnnl simdjson AilaSyclOptions)

add_executable(AilaQwen4ExpInferRefAdapter
    tests/inferref/Qwen4ExpInferRefAdapter.cpp
    src/core/ExecutionTrace.cpp
    src/memory/LargePages.cpp)
target_compile_definitions(AilaQwen4ExpInferRefAdapter PRIVATE
    NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaQwen4ExpInferRefAdapter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQwen4ExpInferRefAdapter PRIVATE
    DNNL::dnnl simdjson AilaSyclOptions)

add_executable(AilaQwen4ExpModelSpecTests
    tests/models/Qwen4ExpModelSpecTests.cpp
)
target_compile_definitions(AilaQwen4ExpModelSpecTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpModelSpecTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaQwen4ExpModelSpecTests PRIVATE AilaLib)
add_test(NAME AilaQwen4ExpModelSpecTests COMMAND AilaQwen4ExpModelSpecTests)
set_tests_properties(AilaQwen4ExpModelSpecTests PROPERTIES LABELS "cpu;modelspec" TIMEOUT 60)

add_executable(AilaQwen4ExpKernelTests
    tests/ops/Qwen4ExpKernelTests.cpp
    src/core/ExecutionTrace.cpp
)
target_compile_definitions(AilaQwen4ExpKernelTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpKernelTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQwen4ExpKernelTests PRIVATE AilaLib)
add_test(NAME AilaQwen4ExpKernelTests COMMAND AilaQwen4ExpKernelTests)
set_tests_properties(AilaQwen4ExpKernelTests PROPERTIES LABELS "gpu;ops;qwen4exp" TIMEOUT 60)
add_executable(AilaGdnChunkTests tests/ops/GdnChunkTests.cpp)
target_compile_definitions(AilaGdnChunkTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaGdnChunkTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(AilaGdnChunkTests PRIVATE AilaLib)
add_test(NAME AilaGdnChunkTests COMMAND AilaGdnChunkTests)
set_tests_properties(AilaGdnChunkTests PROPERTIES LABELS "gpu;ops;qwen4_exp" TIMEOUT 90)

add_executable(AilaQwen4ExpMoeBench
    perf/qwen4exp/q4e_moe_bench.cpp
    src/core/ExecutionTrace.cpp
)
target_compile_definitions(AilaQwen4ExpMoeBench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpMoeBench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQwen4ExpMoeBench PRIVATE AilaLib)

add_executable(AilaQwen4ExpGdnBench
    perf/qwen4exp/q4e_gdn_bench.cpp
    src/core/ExecutionTrace.cpp
)
target_compile_definitions(AilaQwen4ExpGdnBench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpGdnBench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQwen4ExpGdnBench PRIVATE AilaLib)

add_executable(AilaQwen4ExpDecodeMoeBench
    perf/qwen4exp/q4e_decode_moe_bench.cpp
    src/core/ExecutionTrace.cpp
)
target_compile_definitions(AilaQwen4ExpDecodeMoeBench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpDecodeMoeBench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQwen4ExpDecodeMoeBench PRIVATE AilaLib)

add_executable(AilaQwen4ExpPrefillSupplyBench perf/qwen4exp/q4e_prefill_supply_bench.cpp)
target_compile_definitions(AilaQwen4ExpPrefillSupplyBench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpPrefillSupplyBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(AilaQwen4ExpPrefillSupplyBench PRIVATE AilaLib)

add_executable(AilaQwen4ExpPrefillBlockBench perf/qwen4exp/q4e_prefill_block_bench.cpp)
target_compile_definitions(AilaQwen4ExpPrefillBlockBench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpPrefillBlockBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(AilaQwen4ExpPrefillBlockBench PRIVATE AilaLib)

add_executable(AilaQwen4ExpDecodeMiscBench perf/qwen4exp/q4e_decode_misc_bench.cpp)
target_compile_definitions(AilaQwen4ExpDecodeMiscBench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpDecodeMiscBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(AilaQwen4ExpDecodeMiscBench PRIVATE AilaLib)

add_executable(AilaQwen4ExpCpuMoePrefillBench perf/qwen4exp/q4e_cpu_moe_prefill_bench.cpp)
target_compile_definitions(AilaQwen4ExpCpuMoePrefillBench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpCpuMoePrefillBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(AilaQwen4ExpCpuMoePrefillBench PRIVATE AilaLib TBB::tbb)

add_executable(AilaQwen4ExpHostReadBench perf/qwen4exp/q4e_hostread_bench.cpp)
target_compile_definitions(AilaQwen4ExpHostReadBench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpHostReadBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(AilaQwen4ExpHostReadBench PRIVATE AilaLib)

add_executable(AilaQwen4ExpArenaDmaBench perf/qwen4exp/q4e_arena_dma_bench.cpp)
target_compile_definitions(AilaQwen4ExpArenaDmaBench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpArenaDmaBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(AilaQwen4ExpArenaDmaBench PRIVATE AilaLib)

add_executable(AilaQwen4ExpModuleTests
    tests/models/Qwen4ExpModuleTests.cpp
    src/core/ExecutionTrace.cpp
)
target_compile_definitions(AilaQwen4ExpModuleTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpModuleTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQwen4ExpModuleTests PRIVATE AilaLib)
add_test(NAME AilaQwen4ExpModuleTests COMMAND AilaQwen4ExpModuleTests)
set_tests_properties(AilaQwen4ExpModuleTests PROPERTIES LABELS "gpu;models;qwen4_exp" TIMEOUT 300)

add_executable(AilaQwen4ExpGenerationTests tests/models/Qwen4ExpGenerationTests.cpp)
target_compile_definitions(AilaQwen4ExpGenerationTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpGenerationTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQwen4ExpGenerationTests PRIVATE AilaLib)
if(NOT MSVC)
    target_compile_options(AilaQwen4ExpGenerationTests PRIVATE -mavx2 -mfma)
endif()
add_test(NAME AilaQwen4ExpGenerationTests COMMAND AilaQwen4ExpGenerationTests)
set_tests_properties(AilaQwen4ExpGenerationTests PROPERTIES LABELS "gpu;models;qwen4_exp;h6" TIMEOUT 60)

add_executable(AilaQwen4ExpToleranceTests tests/models/Qwen4ExpToleranceTests.cpp)
target_compile_definitions(AilaQwen4ExpToleranceTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpToleranceTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQwen4ExpToleranceTests PRIVATE AilaLib)
if(NOT MSVC)
    target_compile_options(AilaQwen4ExpToleranceTests PRIVATE -mavx2 -mfma)
endif()
add_test(NAME AilaQwen4ExpToleranceTests COMMAND AilaQwen4ExpToleranceTests)
set_tests_properties(AilaQwen4ExpToleranceTests PROPERTIES LABELS "gpu;models;qwen4_exp;tolerance" TIMEOUT 60)
add_test(NAME AilaQwen4ExpDpasSmoke COMMAND AilaQwen4ExpToleranceTests --dpas-smoke)
set_tests_properties(AilaQwen4ExpDpasSmoke PROPERTIES LABELS "gpu;models;qwen4_exp;dpas" TIMEOUT 30)
add_test(NAME AilaQwen4ExpInt8DpasAttention COMMAND AilaQwen4ExpKernelTests --kv-int8-dpas-only)
set_tests_properties(AilaQwen4ExpInt8DpasAttention PROPERTIES
    LABELS "gpu;ops;qwen4_exp;dpas" TIMEOUT 60
    ENVIRONMENT "AILA_KV_QUANT_DPAS=1;AILA_ATTN_DECODE_WINDOW=0;AILA_ATTN_DECODE_SINK=0")
add_test(NAME AilaQwen4ExpGroupedXmxTests COMMAND AilaQwen4ExpToleranceTests --grouped-kernel)
set_tests_properties(AilaQwen4ExpGroupedXmxTests PROPERTIES LABELS "gpu;ops;qwen4_exp" TIMEOUT 60)

add_executable(AilaQwen4ExpCpuExpertTests
    tests/ops/Qwen4ExpCpuExpertTests.cpp
)
target_compile_definitions(AilaQwen4ExpCpuExpertTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpCpuExpertTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQwen4ExpCpuExpertTests PRIVATE AilaLib)
if(MSVC)
    target_compile_options(AilaQwen4ExpCpuExpertTests PRIVATE /arch:AVX2)
    set_source_files_properties(src/ops/Qwen4ExpCpuExpertOps.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
else()
    target_compile_options(AilaQwen4ExpCpuExpertTests PRIVATE -mavx2 -mfma)
    set_source_files_properties(src/ops/Qwen4ExpCpuExpertOps.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
endif()
add_test(NAME AilaQwen4ExpCpuExpertTests COMMAND AilaQwen4ExpCpuExpertTests)
set_tests_properties(AilaQwen4ExpCpuExpertTests PROPERTIES LABELS "gpu;ops;qwen4exp;h2a" TIMEOUT 120)

add_executable(AilaQwen4ExpGpuQ8Tests tests/ops/Qwen4ExpGpuQ8Tests.cpp)
target_compile_definitions(AilaQwen4ExpGpuQ8Tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpGpuQ8Tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQwen4ExpGpuQ8Tests PRIVATE AilaLib)
add_test(NAME AilaQwen4ExpGpuQ8Tests COMMAND AilaQwen4ExpGpuQ8Tests)
set_tests_properties(AilaQwen4ExpGpuQ8Tests PROPERTIES
    LABELS "gpu;ops;qwen4exp;q8" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu SKIP_RETURN_CODE 77 TIMEOUT 120)
add_test(NAME AilaQwen4ExpGpuQ8FastTests COMMAND AilaQwen4ExpGpuQ8Tests)
set_tests_properties(AilaQwen4ExpGpuQ8FastTests PROPERTIES
    ENVIRONMENT "AILA_Q4E_GPU_Q8_FAST_RTE=1"
    LABELS "gpu;ops;qwen4exp;q8" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu SKIP_RETURN_CODE 77 TIMEOUT 120)

add_executable(AilaMoeExpertXmxTests tests/ops/MoeExpertXmxTests.cpp)
target_compile_definitions(AilaMoeExpertXmxTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaMoeExpertXmxTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaMoeExpertXmxTests PRIVATE AilaLib)
add_test(NAME AilaMoeExpertXmxTests COMMAND AilaMoeExpertXmxTests)
set_tests_properties(AilaMoeExpertXmxTests PROPERTIES
    LABELS "gpu;ops;qwen4exp;prefill_xmx" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu TIMEOUT 120)

add_executable(AilaQwen4ExpExpertReaderTests
    tests/models/Qwen4ExpExpertReaderTests.cpp
)
target_compile_definitions(AilaQwen4ExpExpertReaderTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpExpertReaderTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQwen4ExpExpertReaderTests PRIVATE AilaLib)
add_test(NAME AilaQwen4ExpExpertReaderTests COMMAND AilaQwen4ExpExpertReaderTests)
set_tests_properties(AilaQwen4ExpExpertReaderTests PROPERTIES LABELS "cpu;models;qwen4_exp;h4" TIMEOUT 120)

add_executable(AilaQwen4ExpHostArenaTests
    tests/models/Qwen4ExpHostArenaTests.cpp
)
target_compile_definitions(AilaQwen4ExpHostArenaTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpHostArenaTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQwen4ExpHostArenaTests PRIVATE AilaLib)
add_test(NAME AilaQwen4ExpHostArenaTests COMMAND AilaQwen4ExpHostArenaTests)
set_tests_properties(AilaQwen4ExpHostArenaTests PROPERTIES LABELS "gpu;models;qwen4_exp;arena" TIMEOUT 120)

add_executable(AilaQwen4ExpAdaptiveCacheTests
    tests/models/Qwen4ExpAdaptiveCacheTests.cpp
)
target_compile_definitions(AilaQwen4ExpAdaptiveCacheTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpAdaptiveCacheTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQwen4ExpAdaptiveCacheTests PRIVATE AilaLib)
add_test(NAME AilaQwen4ExpAdaptiveCacheTests COMMAND AilaQwen4ExpAdaptiveCacheTests)
set_tests_properties(AilaQwen4ExpAdaptiveCacheTests PROPERTIES LABELS "gpu;models;qwen4_exp;cache" TIMEOUT 300)

add_executable(AilaQwen4ExpArenaVerifyTests
    tests/models/Qwen4ExpArenaVerifyTests.cpp
)
target_compile_definitions(AilaQwen4ExpArenaVerifyTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaQwen4ExpArenaVerifyTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaQwen4ExpArenaVerifyTests PRIVATE AilaLib)
if(NOT MSVC)
    target_compile_options(AilaQwen4ExpArenaVerifyTests PRIVATE -mavx2 -mfma)
endif()

foreach(aila_sycl_test_target IN ITEMS
    AilaBnb4BitLinearKernelTests
    AilaQsaAttentionProbe
    AilaH2DBandwidthProbe
    AilaKernelSubmitProbe
    AilaConvOpsKernelTests
    AilaYolo26InferRefAdapter
    AilaDiffusionGemmaNf4InferRefAdapter
    AilaQwen4ExpInferRefAdapter
    AilaQwen4ExpKernelTests
    AilaGdnChunkTests
    AilaQwen4ExpMoeBench
    AilaQwen4ExpGdnBench
    AilaQwen4ExpDecodeMoeBench
    AilaQwen4ExpPrefillSupplyBench
    AilaQwen4ExpPrefillBlockBench
    AilaQwen4ExpDecodeMiscBench
    AilaQwen4ExpCpuMoePrefillBench
    AilaQwen4ExpHostReadBench
    AilaQwen4ExpArenaDmaBench
    AilaQwen4ExpModuleTests
    AilaQwen4ExpGenerationTests
    AilaQwen4ExpToleranceTests
    AilaQwen4ExpGpuQ8Tests
    AilaMoeExpertXmxTests
    AilaQwen4ExpHostArenaTests
    AilaQwen4ExpAdaptiveCacheTests
    AilaQwen4ExpArenaVerifyTests
)
    if(TARGET AilaLevelZero)
        target_link_libraries(${aila_sycl_test_target} PRIVATE AilaLevelZero)
    endif()
    get_target_property(aila_target_srcs ${aila_sycl_test_target} SOURCES)
    list(GET aila_target_srcs 0 aila_main_test_src)
    aila_enable_sycl_pch(${aila_sycl_test_target} SOURCES ${aila_main_test_src})
endforeach()
add_test(NAME AilaQwen4ExpArenaVerifyTests COMMAND AilaQwen4ExpArenaVerifyTests)
set_tests_properties(AilaQwen4ExpArenaVerifyTests PROPERTIES LABELS "gpu;models;qwen4_exp;verify" TIMEOUT 300)

add_executable(AilaDiffusionGemmaModelSpecTests
    tests/models/DiffusionGemmaModelSpecTests.cpp
)
target_compile_definitions(AilaDiffusionGemmaModelSpecTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaDiffusionGemmaModelSpecTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaDiffusionGemmaModelSpecTests PRIVATE AilaLib)
add_test(NAME AilaDiffusionGemmaModelSpecTests COMMAND AilaDiffusionGemmaModelSpecTests)
set_tests_properties(AilaDiffusionGemmaModelSpecTests PROPERTIES LABELS "gpu;models;ops" TIMEOUT 120)

add_executable(AilaDiffusionGemmaFusionTests
    tests/ops/DiffusionGemmaFusionTests.cpp
)
target_compile_definitions(AilaDiffusionGemmaFusionTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaDiffusionGemmaFusionTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaDiffusionGemmaFusionTests PRIVATE AilaLib)
add_test(NAME AilaDiffusionGemmaFusionTests COMMAND AilaDiffusionGemmaFusionTests)

add_executable(AilaDiffusionGemmaVisionTests
    tests/vision/DiffusionGemmaVisionTests.cpp
)
target_compile_definitions(AilaDiffusionGemmaVisionTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaDiffusionGemmaVisionTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaDiffusionGemmaVisionTests PRIVATE AilaLib)
add_test(NAME AilaDiffusionGemmaVisionTests COMMAND AilaDiffusionGemmaVisionTests)

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

add_executable(AilaTokenizerRegressionTests tests/utils/TokenizerRegressionTests.cpp)
target_compile_definitions(AilaTokenizerRegressionTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaTokenizerRegressionTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaTokenizerRegressionTests PRIVATE AilaLib)
add_test(NAME AilaTokenizerRegressionTests COMMAND AilaTokenizerRegressionTests)
set_tests_properties(AilaTokenizerRegressionTests PROPERTIES LABELS "cpu;tokenizer" TIMEOUT 30)

add_executable(AilaMimiEncoderTests tests/audio/MimiEncoderTests.cpp)
target_compile_definitions(AilaMimiEncoderTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaMimiEncoderTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaMimiEncoderTests PRIVATE AilaLib simdjson)
add_test(NAME AilaMimiEncoderTests COMMAND AilaMimiEncoderTests WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(AilaTTSVoiceCloneAlignmentTests tests/audio/TTSVoiceCloneAlignmentTests.cpp)
target_compile_definitions(AilaTTSVoiceCloneAlignmentTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaTTSVoiceCloneAlignmentTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaTTSVoiceCloneAlignmentTests PRIVATE AilaLib simdjson)
add_test(NAME AilaTTSVoiceCloneAlignmentTests COMMAND AilaTTSVoiceCloneAlignmentTests WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

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

# IPC v2 is an internal ordinary-C++ API, independent of the legacy ABI and
# its whole-request WorkerProcess. None of these libraries loads a SYCL runtime.
add_library(AilaInferenceProtocol STATIC
    src/ipc/IpcProtocol.cpp
    src/ipc/InferenceProtocol.cpp)
target_compile_definitions(AilaInferenceProtocol PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(AilaInferenceProtocol PRIVATE -fno-sycl -Wno-unused-command-line-argument)
target_include_directories(AilaInferenceProtocol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(AilaInferenceProtocol PUBLIC simdjson)
if(WIN32)
    target_link_directories(AilaInferenceProtocol PUBLIC "${AILA_COMPILER_ROOT}/lib")
endif()
if(MSVC)
    target_compile_options(AilaInferenceProtocol PRIVATE /fp:precise)
else()
    target_compile_options(AilaInferenceProtocol PRIVATE -fno-fast-math)
endif()

add_library(AilaInferenceWorkerCore STATIC src/worker/InferenceWorker.cpp)
target_compile_definitions(AilaInferenceWorkerCore PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(AilaInferenceWorkerCore PRIVATE -fno-sycl -Wno-unused-command-line-argument)
target_include_directories(AilaInferenceWorkerCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(AilaInferenceWorkerCore PUBLIC AilaInferenceProtocol Threads::Threads)

add_executable(AilaInferenceProtocolTests tests/runtime/InferenceProtocolTests.cpp)
target_compile_definitions(AilaInferenceProtocolTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(AilaInferenceProtocolTests PRIVATE -fno-sycl -Wno-unused-command-line-argument)
target_link_options(AilaInferenceProtocolTests PRIVATE -fno-sycl)
target_link_libraries(AilaInferenceProtocolTests PRIVATE AilaInferenceProtocol)
if(MSVC)
    target_compile_options(AilaInferenceProtocolTests PRIVATE /fp:precise)
else()
    target_compile_options(AilaInferenceProtocolTests PRIVATE -fno-fast-math)
endif()
add_test(NAME AilaInferenceProtocolTests COMMAND AilaInferenceProtocolTests)
set_tests_properties(AilaInferenceProtocolTests PROPERTIES LABELS "cpu;ipc" TIMEOUT 30)

add_executable(AilaInferenceWorkerTests
    tests/runtime/InferenceWorkerTests.cpp
    src/inference/Session.cpp
    src/inference/Request.cpp
    src/inference/Scheduler.cpp
    src/profile/Profiling.cpp)
target_compile_definitions(AilaInferenceWorkerTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(AilaInferenceWorkerTests PRIVATE -fno-sycl -Wno-unused-command-line-argument)
target_link_options(AilaInferenceWorkerTests PRIVATE -fno-sycl)
target_link_libraries(AilaInferenceWorkerTests PRIVATE AilaInferenceWorkerCore)
add_test(NAME AilaInferenceWorkerTests COMMAND AilaInferenceWorkerTests)
set_tests_properties(AilaInferenceWorkerTests PROPERTIES LABELS "cpu;inference;ipc" TIMEOUT 90)

add_executable(AilaRuntimeFaultTests tests/inference/RuntimeFaultTests.cpp)
target_compile_definitions(AilaRuntimeFaultTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(AilaRuntimeFaultTests PRIVATE -fno-sycl -Wno-unused-command-line-argument)
target_link_options(AilaRuntimeFaultTests PRIVATE -fno-sycl)
target_include_directories(AilaRuntimeFaultTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(AilaRuntimeFaultTests PRIVATE Threads::Threads)
if(WIN32)
    target_link_directories(AilaRuntimeFaultTests PRIVATE "${AILA_COMPILER_ROOT}/lib")
endif()
add_test(NAME AilaRuntimeFaultTests COMMAND AilaRuntimeFaultTests)
set_tests_properties(AilaRuntimeFaultTests PROPERTIES LABELS "cpu;inference" TIMEOUT 30)

add_executable(AilaExecutionTraceTests
    tests/inference/ExecutionTraceTests.cpp
    src/core/ExecutionTrace.cpp)
target_compile_definitions(AilaExecutionTraceTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(AilaExecutionTraceTests PRIVATE -fno-sycl -Wno-unused-command-line-argument)
target_link_options(AilaExecutionTraceTests PRIVATE -fno-sycl)
target_include_directories(AilaExecutionTraceTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(AilaExecutionTraceTests PRIVATE Threads::Threads)
if(WIN32)
    target_link_directories(AilaExecutionTraceTests PRIVATE "${AILA_COMPILER_ROOT}/lib")
    set_target_properties(AilaExecutionTraceTests PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_link_libraries(AilaExecutionTraceTests PRIVATE simdjson_proxy)
else()
    target_link_libraries(AilaExecutionTraceTests PRIVATE simdjson)
endif()
add_test(NAME AilaExecutionTraceTests COMMAND AilaExecutionTraceTests)
set_tests_properties(AilaExecutionTraceTests PROPERTIES LABELS "cpu;inference;diagnostics" TIMEOUT 30)

if(WIN32)
    get_filename_component(AILA_WORKER_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(AILA_WORKER_COMPILER_ROOT
        "${AILA_WORKER_COMPILER_BIN_DIR}/.." ABSOLUTE)

    add_executable(AilaWorker
        src/worker/WorkerMain.cpp
        src/worker/CApiWorkerEngine.cpp
        src/worker/WorkerDispatcher.cpp
        src/worker/InferenceWorkerMain.cpp
        src/worker/SharedInferenceEngine.cpp
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
    target_link_libraries(AilaWorker PRIVATE AilaLib simdjson AilaBuildIdentity AilaInferenceWorkerCore bcrypt)

    add_library(AilaInferenceClient STATIC
        src/runtime/InferenceClient.cpp
        src/runtime/InferencePipeTransport.cpp
        src/runtime/RuntimeDirectory.cpp
        src/runtime/ChildEnvironment.cpp
        src/ipc/Win32Pipe.cpp)
    target_compile_definitions(AilaInferenceClient PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(AilaInferenceClient PRIVATE -fno-sycl -Wno-unused-command-line-argument)
    target_link_libraries(AilaInferenceClient PUBLIC AilaInferenceProtocol Threads::Threads)
    if(MSVC)
        target_compile_options(AilaInferenceClient PRIVATE /fp:precise)
    else()
        target_compile_options(AilaInferenceClient PRIVATE -fno-fast-math)
    endif()

    # Public asynchronous token ABI uses the same CRT as InferenceClient.
    # The old MT proxy remains a separate DLL with its unchanged ABI.
    add_library(AilaInference SHARED src/runtime/AilaInferenceApi.cpp)
    target_compile_definitions(AilaInference PRIVATE
        NOMINMAX WIN32_LEAN_AND_MEAN AILA_INFERENCE_BUILDING_DLL)
    target_include_directories(AilaInference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(AilaInference PRIVATE -fno-sycl -Wno-unused-command-line-argument)
    target_link_options(AilaInference PRIVATE -fno-sycl)
    target_link_libraries(AilaInference PRIVATE AilaInferenceClient AilaBuildIdentity)
    if(MSVC)
        target_compile_options(AilaInference PRIVATE /fp:precise)
    else()
        target_compile_options(AilaInference PRIVATE -fno-fast-math)
    endif()

    add_executable(AilaInferenceClientTests tests/runtime/InferenceClientTests.cpp)
    target_compile_definitions(AilaInferenceClientTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(AilaInferenceClientTests PRIVATE -fno-sycl -Wno-unused-command-line-argument)
    target_link_options(AilaInferenceClientTests PRIVATE -fno-sycl)
    target_link_libraries(AilaInferenceClientTests PRIVATE AilaInferenceClient)
    if(MSVC)
        target_compile_options(AilaInferenceClientTests PRIVATE /fp:precise)
    else()
        target_compile_options(AilaInferenceClientTests PRIVATE -fno-fast-math)
    endif()
    add_test(NAME AilaInferenceClientTests COMMAND AilaInferenceClientTests)
    set_tests_properties(AilaInferenceClientTests PROPERTIES LABELS "cpu;inference;ipc" TIMEOUT 90)

    add_executable(AilaQwen3IpcSessionTests tests/inference/Qwen3IpcSessionTests.cpp)
    target_compile_definitions(AilaQwen3IpcSessionTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB
        AILA_IPC_WORKER_PATH="$<TARGET_FILE:AilaWorker>")
    target_link_libraries(AilaQwen3IpcSessionTests PRIVATE AilaLib AilaInferenceClient AilaBuildIdentity)
    if(MSVC)
        target_compile_options(AilaQwen3IpcSessionTests PRIVATE /fp:precise)
    else()
        target_compile_options(AilaQwen3IpcSessionTests PRIVATE -fno-fast-math)
    endif()
    add_dependencies(AilaQwen3IpcSessionTests AilaWorker)
    add_test(NAME AilaQwen3IpcSessionTests COMMAND AilaQwen3IpcSessionTests)
    add_test(NAME AilaQwen35IpcSessionTests COMMAND AilaQwen3IpcSessionTests --backend qwen35_nf4)
    add_test(NAME AilaQwen35Bf16IpcSessionTests COMMAND AilaQwen3IpcSessionTests --backend qwen35_bf16)
    set_tests_properties(AilaQwen3IpcSessionTests AilaQwen35IpcSessionTests AilaQwen35Bf16IpcSessionTests PROPERTIES
        LABELS "gpu;inference;ipc" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu SKIP_RETURN_CODE 77 TIMEOUT 600)

    foreach(fixture IN ITEMS IpcBenchmark)
        add_executable(Aila${fixture} tests/inference/${fixture}.cpp)
        target_compile_definitions(Aila${fixture} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB
            AILA_IPC_WORKER_PATH="$<TARGET_FILE:AilaWorker>")
        target_link_libraries(Aila${fixture} PRIVATE AilaLib AilaInferenceClient AilaBuildIdentity)
        if(MSVC)
            target_compile_options(Aila${fixture} PRIVATE /fp:precise)
        else()
            target_compile_options(Aila${fixture} PRIVATE -fno-fast-math)
        endif()
        add_dependencies(Aila${fixture} AilaWorker)
    endforeach()
    add_executable(AilaFakeInferenceWorker
        tests/runtime/FakeInferenceWorkerMain.cpp
        src/worker/InferenceWorkerMain.cpp
        src/core/ExecutionTrace.cpp
        src/ipc/Win32Pipe.cpp
        src/inference/Session.cpp
        src/inference/Request.cpp
        src/inference/Scheduler.cpp
        src/profile/Profiling.cpp)
    target_compile_definitions(AilaFakeInferenceWorker PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(AilaFakeInferenceWorker PRIVATE -fno-sycl -Wno-unused-command-line-argument)
    target_link_options(AilaFakeInferenceWorker PRIVATE -fno-sycl)
    target_link_libraries(AilaFakeInferenceWorker PRIVATE AilaInferenceWorkerCore AilaBuildIdentity bcrypt)

    add_executable(AilaInferenceCHeaderTests
        tests/runtime/InferenceCHeaderTests.c tests/runtime/InferenceCHeaderMain.cpp)
    target_compile_definitions(AilaInferenceCHeaderTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(AilaInferenceCHeaderTests PRIVATE -fno-sycl -Wno-unused-command-line-argument)
    target_link_options(AilaInferenceCHeaderTests PRIVATE -fno-sycl)
    target_link_libraries(AilaInferenceCHeaderTests PRIVATE AilaInference)
    add_test(NAME AilaInferenceCHeaderTests COMMAND AilaInferenceCHeaderTests)
    set_tests_properties(AilaInferenceCHeaderTests PROPERTIES LABELS "cpu;inference;abi" TIMEOUT 15)
    add_executable(AilaInferenceCApiGpuTests tests/inference/InferenceCApiTests.cpp)
    target_compile_definitions(AilaInferenceCApiGpuTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(AilaInferenceCApiGpuTests PRIVATE -fno-sycl -Wno-unused-command-line-argument)
    target_link_options(AilaInferenceCApiGpuTests PRIVATE -fno-sycl)
    target_link_libraries(AilaInferenceCApiGpuTests PRIVATE AilaInference AilaInferenceClient AilaBuildIdentity)
    if(MSVC)
        target_compile_options(AilaInferenceCApiGpuTests PRIVATE /fp:precise)
    else()
        target_compile_options(AilaInferenceCApiGpuTests PRIVATE -fno-fast-math)
    endif()
    add_dependencies(AilaInferenceCApiGpuTests AilaWorker)
    add_test(NAME AilaInferenceCApiGpuTests COMMAND AilaInferenceCApiGpuTests)
    set_tests_properties(AilaInferenceCApiGpuTests PROPERTIES
        LABELS "gpu;inference;abi;ipc" RUN_SERIAL TRUE RESOURCE_LOCK aila_gpu SKIP_RETURN_CODE 77 TIMEOUT 600)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME AilaInferenceCApiTests COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/inference_c_api_tests.py
            --dll $<TARGET_FILE:AilaInference> --worker $<TARGET_FILE:AilaFakeInferenceWorker>)
        set_tests_properties(AilaInferenceCApiTests PROPERTIES LABELS "cpu;inference;abi;ipc" TIMEOUT 90)
        add_test(NAME AilaLegacyTextApiTests COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/legacy_text_api_tests.py
            --dll $<TARGET_FILE:AilaShared> --worker $<TARGET_FILE:AilaFakeInferenceWorker>)
        set_tests_properties(AilaLegacyTextApiTests PROPERTIES LABELS "cpu;inference;abi;ipc" TIMEOUT 120)
    endif()

    add_executable(AilaLegacyTextSessionTests
        tests/runtime/LegacyTextSessionTests.cpp
        src/proxy/LegacyTextSession.cpp
        src/proxy/LegacyInferenceBridge.cpp
        src/proxy/ProxyLogging.cpp
        src/utils/Tokenizer.cpp
        src/utils/ModelSpec.cpp
        src/profile/Profiling.cpp
        ${CHAT_SOURCES} ${JINJA_SOURCES})
    target_compile_definitions(AilaLegacyTextSessionTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
    target_compile_options(AilaLegacyTextSessionTests PRIVATE -fno-sycl -Wno-unused-command-line-argument /fp:precise)
    target_link_options(AilaLegacyTextSessionTests PRIVATE -fno-sycl)
    set_target_properties(AilaLegacyTextSessionTests PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_include_directories(AilaLegacyTextSessionTests PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
    target_link_directories(AilaLegacyTextSessionTests PRIVATE "${AILA_COMPILER_ROOT}/lib")
    target_link_libraries(AilaLegacyTextSessionTests PRIVATE simdjson_proxy AilaBuildIdentity Threads::Threads)
    add_test(NAME AilaLegacyTextSessionTests COMMAND AilaLegacyTextSessionTests)
    set_tests_properties(AilaLegacyTextSessionTests PROPERTIES LABELS "cpu;inference;abi" TIMEOUT 30)

    add_executable(AilaInferencePipeTests tests/runtime/InferencePipeTests.cpp)
    target_compile_definitions(AilaInferencePipeTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN
        AILA_FAKE_INFERENCE_WORKER_PATH="$<TARGET_FILE:AilaFakeInferenceWorker>")
    target_compile_options(AilaInferencePipeTests PRIVATE -fno-sycl -Wno-unused-command-line-argument)
    target_link_options(AilaInferencePipeTests PRIVATE -fno-sycl)
    target_link_libraries(AilaInferencePipeTests PRIVATE AilaInferenceClient AilaBuildIdentity)
    add_dependencies(AilaInferencePipeTests AilaFakeInferenceWorker)
    add_test(NAME AilaInferencePipeTests COMMAND AilaInferencePipeTests)
    set_tests_properties(AilaInferencePipeTests PROPERTIES LABELS "cpu;inference;ipc" TIMEOUT 90)
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
                "^(ur_loader|ur_loader_shared|ur_win_proxy_loader|ur_adapter_opencl|ur_adapter_level_zero|ur_adapter_level_zero_v2|umf|xptifw|sycl-jit|opencl|intelocl64|common_clang64|libmmd.*|libhwloc-.*|tbb12|tbbbind|tbbbind_2_0|tbbbind_2_5|tbbmalloc|tbbmalloc_proxy|tcm|concrt140|msvcp140.*|vccorlib140|vcruntime140.*)\\.dll$")
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
if(HOST_ONLY)
    set(extra_root_dlls)
endif()

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
    if(HOST_ONLY)
        get_filename_component(host_dependency_name "${dep}" NAME)
        string(TOLOWER "${host_dependency_name}" host_dependency_name)
        if(host_dependency_name MATCHES "(sycl|dnnl|onednn|ze_loader|ur_adapter|ur_loader)")
            message(FATAL_ERROR "Host-only package depends on GPU runtime: ${dep}")
        endif()
    endif()
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
    set(AILA_RELEASE_INCLUDE_DIR "${CMAKE_BINARY_DIR}/Release/include")
    set(AILA_RELEASE_LIB_DIR "${CMAKE_BINARY_DIR}/Release/lib")

    add_custom_target(release
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${AILA_RELEASE_BIN_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${AILA_RELEASE_RUNTIME_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${AILA_RELEASE_RUNTIME_DIR}/licenses/soxr"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${AILA_RELEASE_INCLUDE_DIR}" "${AILA_RELEASE_LIB_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:AilaShared>" "${AILA_RELEASE_BIN_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:AilaInference>" "${AILA_RELEASE_BIN_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_LINKER_FILE:AilaInference>" "${AILA_RELEASE_LIB_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PROJECT_SOURCE_DIR}/include/aila_inference.h" "${AILA_RELEASE_INCLUDE_DIR}"
        COMMAND ${CMAKE_COMMAND}
            "-DTARGET_PATH=$<TARGET_FILE:AilaInference>"
            "-DTARGET_DIR=${AILA_RELEASE_BIN_DIR}"
            "-DTARGET_KIND=LIBRARY"
            "-DHOST_ONLY=ON"
            -P "${CMAKE_CURRENT_BINARY_DIR}/copy_runtime_dependencies.cmake"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:Aila>" "${AILA_RELEASE_RUNTIME_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:AilaWorker>" "${AILA_RELEASE_RUNTIME_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PROJECT_SOURCE_DIR}/third_party/soxr/LICENCE" "${AILA_RELEASE_RUNTIME_DIR}/licenses/soxr/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PROJECT_SOURCE_DIR}/third_party/soxr/COPYING.LGPL" "${AILA_RELEASE_RUNTIME_DIR}/licenses/soxr/"
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
        COMMENT "Staging release binaries and third-party licenses in ${AILA_RELEASE_BIN_DIR}"
        VERBATIM
    )
    add_dependencies(release Aila AilaShared AilaInference AilaWorker soxr)
endif()
