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

if(MSVC)
    # --- Base flags ---
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fsycl /Qstd=c++17 /EHsc /Qftz")
    # Match DG2 native SIMD width: all subgroup-aware kernels benefit (+2.8% decode).
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-default-sub-group-size 16")
    # Per-kernel code split: each kernel in its own module for fast incremental build.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-device-code-split=per_kernel")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fsycl -std=c++17 -fsycl-device-code-split=per_kernel")
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsycl")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl")
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

# Core engine sources shared by the Alia custom runtime and optional tools.
set(COMMON_ENGINE_SOURCES
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
    src/models/cpu/CpuSafetensorsStore.cpp
    src/models/cpu/CpuBnb4.cpp
    src/models/cpu/CpuQ35HybridModel.cpp
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
    src/audio/Qwen3ASRAudioEncoder.cpp
    src/audio/AudioPreprocessor.cpp
    src/audio/GpuMelSpectrogram.cpp
    src/audio/SpeakerEncoder.cpp
    src/audio/GpuSpeakerEncoder.cpp
    src/profile/Device.cpp
    src/profile/Profiling.cpp
    src/lora/LoraLoader.cpp
    src/bench/Benchmark.cpp
)

set(ALIA_ENGINE_SOURCES
    src/alia/RuntimeContext.cpp
    src/alia/ModelSlot.cpp
    src/alia/AliaAsrPipeline.cpp
    src/alia/AliaTtsPipeline.cpp
    src/alia/AliaTtsTextChunker.cpp
    src/alia/AliaTurnScheduler.cpp
    src/alia/AliaForegroundPipeline.cpp
    src/alia/AliaBackgroundPipeline.cpp
    src/alia/BackgroundExtractorFactory.cpp
    src/alia/CpuQ35BackgroundExtractor.cpp
    src/alia/GpuVlmBackgroundExtractor.cpp
    src/alia/AliaContext.cpp
    src/alia/AliaApi.cpp
)

set(LIB_SOURCES
    ${COMMON_ENGINE_SOURCES}
    ${ALIA_ENGINE_SOURCES}
)

set(ALIA_REAL_SMOKE_SOURCES
    tools/alia/AliaRealModelSmoke.cpp
)

set(COMMAND_SUBMISSION_BENCH_SOURCES
    tools/perf/CommandSubmissionBench.cpp
)

set(ALIA_TURN_SCHEDULER_TEST_SOURCES
    tests/alia/AliaTurnSchedulerTests.cpp
    src/alia/AliaTurnScheduler.cpp
)

set(ALIA_TTS_TEXT_CHUNKER_TEST_SOURCES
    tests/alia/AliaTtsTextChunkerTests.cpp
    src/alia/AliaTtsTextChunker.cpp
)

set(ALIA_ASR_LANGUAGE_HINT_TEST_SOURCES
    tests/alia/AliaAsrLanguageHintTests.cpp
)

set(ALIA_BACKGROUND_PIPELINE_TEST_SOURCES
    tests/alia/AliaBackgroundPipelineTests.cpp
    src/alia/AliaBackgroundPipeline.cpp
    src/alia/GpuVlmBackgroundExtractor.cpp
)

set(ALIA_BACKGROUND_EXTRACTOR_TEST_SOURCES
    tests/alia/AliaBackgroundExtractorTests.cpp
)

set(ENV_UTILS_TEST_SOURCES
    tests/utils/EnvUtilsTests.cpp
)

set(SAFE_TENSORS_TEST_SOURCES
    tests/utils/SafeTensorsTests.cpp
    src/utils/SafeTensors.cpp
    src/utils/MemoryMappedFile.cpp
    src/profile/Profiling.cpp
)

set(CPU_SAFETENSORS_STORE_TEST_SOURCES
    tests/models/CpuSafetensorsStoreTests.cpp
    src/models/cpu/CpuSafetensorsStore.cpp
    src/utils/MemoryMappedFile.cpp
)

set(CPU_BNB4_TEST_SOURCES
    tests/models/CpuBnb4Tests.cpp
    src/models/cpu/CpuBnb4.cpp
    src/models/cpu/CpuSafetensorsStore.cpp
    src/utils/MemoryMappedFile.cpp
)

set(CPU_Q35_HYBRID_MODEL_TEST_SOURCES
    tests/models/CpuQ35HybridModelTests.cpp
    src/models/cpu/CpuQ35HybridModel.cpp
    src/models/cpu/CpuBnb4.cpp
    src/models/cpu/CpuSafetensorsStore.cpp
    src/utils/MemoryMappedFile.cpp
)

set(QWEN3_TTS_WARMUP_TEST_SOURCES
    tests/models/Qwen3TTSWarmupTests.cpp
)

set(QWEN35_VISION_WEIGHTS_TEST_SOURCES
    tests/vision/Qwen35VisionWeightsTests.cpp
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

# ============================================================
# Static Library: AilaLib
# ============================================================
add_library(AilaLib STATIC ${LIB_SOURCES})
target_compile_definitions(AilaLib PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_link_libraries(AilaLib PUBLIC DNNL::dnnl simdjson)
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
add_library(AilaShared SHARED ${LIB_SOURCES})
target_compile_definitions(AilaShared PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_BUILDING_DLL)
target_link_libraries(AilaShared PUBLIC DNNL::dnnl simdjson)
if(WIN32)
    target_link_libraries(AilaShared PUBLIC windowscodecs ole32)
endif()
target_include_directories(AilaShared PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)
set_target_properties(AilaShared PROPERTIES OUTPUT_NAME "AilaShared")

# ============================================================
# Executable: AilaAliaRealSmoke
# ============================================================
add_executable(AilaAliaRealSmoke ${ALIA_REAL_SMOKE_SOURCES})
target_compile_definitions(AilaAliaRealSmoke PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaAliaRealSmoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaAliaRealSmoke PRIVATE AilaLib)

add_custom_target(AliaEngine DEPENDS AilaShared)
add_dependencies(AliaEngine AilaAliaRealSmoke)

add_executable(AilaAliaTurnSchedulerTests ${ALIA_TURN_SCHEDULER_TEST_SOURCES})
target_compile_definitions(AilaAliaTurnSchedulerTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaAliaTurnSchedulerTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(AilaAliaTtsTextChunkerTests ${ALIA_TTS_TEXT_CHUNKER_TEST_SOURCES})
target_compile_definitions(AilaAliaTtsTextChunkerTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaAliaTtsTextChunkerTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(AilaAliaAsrLanguageHintTests ${ALIA_ASR_LANGUAGE_HINT_TEST_SOURCES})
target_compile_definitions(AilaAliaAsrLanguageHintTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN AILA_STATIC_LIB)
target_include_directories(AilaAliaAsrLanguageHintTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(AilaAliaAsrLanguageHintTests PRIVATE AilaLib)

add_executable(AilaAliaBackgroundPipelineTests ${ALIA_BACKGROUND_PIPELINE_TEST_SOURCES})
target_compile_definitions(AilaAliaBackgroundPipelineTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaAliaBackgroundPipelineTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaAliaBackgroundPipelineTests PRIVATE AilaLib)

add_executable(AilaAliaBackgroundExtractorTests ${ALIA_BACKGROUND_EXTRACTOR_TEST_SOURCES})
target_compile_definitions(AilaAliaBackgroundExtractorTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaAliaBackgroundExtractorTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaAliaBackgroundExtractorTests PRIVATE AilaLib)

add_executable(AilaEnvUtilsTests ${ENV_UTILS_TEST_SOURCES})
target_compile_definitions(AilaEnvUtilsTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaEnvUtilsTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(AilaSafeTensorsTests ${SAFE_TENSORS_TEST_SOURCES})
target_compile_definitions(AilaSafeTensorsTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaSafeTensorsTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaSafeTensorsTests PRIVATE DNNL::dnnl simdjson)

add_executable(AilaCpuSafetensorsStoreTests ${CPU_SAFETENSORS_STORE_TEST_SOURCES})
target_compile_definitions(AilaCpuSafetensorsStoreTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaCpuSafetensorsStoreTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaCpuSafetensorsStoreTests PRIVATE simdjson)

add_executable(AilaCpuBnb4Tests ${CPU_BNB4_TEST_SOURCES})
target_compile_definitions(AilaCpuBnb4Tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaCpuBnb4Tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaCpuBnb4Tests PRIVATE simdjson)

add_executable(AilaCpuQ35HybridModelTests ${CPU_Q35_HYBRID_MODEL_TEST_SOURCES})
target_compile_definitions(AilaCpuQ35HybridModelTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaCpuQ35HybridModelTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaCpuQ35HybridModelTests PRIVATE simdjson)

add_executable(AilaQwen3TTSWarmupTests ${QWEN3_TTS_WARMUP_TEST_SOURCES})
target_compile_definitions(AilaQwen3TTSWarmupTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaQwen3TTSWarmupTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(AilaQwen35VisionWeightsTests ${QWEN35_VISION_WEIGHTS_TEST_SOURCES})
target_compile_definitions(AilaQwen35VisionWeightsTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaQwen35VisionWeightsTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(AilaBnb4BitLinearKernelTests tests/ops/Bnb4BitLinearKernelTests.cpp)
target_compile_definitions(AilaBnb4BitLinearKernelTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaBnb4BitLinearKernelTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaBnb4BitLinearKernelTests PRIVATE DNNL::dnnl simdjson)

add_executable(AilaBnb4PrefillBench tools/perf/Bnb4PrefillBench.cpp)
target_compile_definitions(AilaBnb4PrefillBench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaBnb4PrefillBench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaBnb4PrefillBench PRIVATE DNNL::dnnl simdjson)

set(AILA_LEVEL_ZERO_SDK_ROOT "C:/level-zero-win-sdk-1.28.2" CACHE PATH
    "Level Zero SDK root for optional command submission benchmarks")
if(EXISTS "${AILA_LEVEL_ZERO_SDK_ROOT}/include/level_zero/ze_api.h" AND
   EXISTS "${AILA_LEVEL_ZERO_SDK_ROOT}/lib/ze_loader.lib")
    add_executable(AilaCommandSubmissionBench ${COMMAND_SUBMISSION_BENCH_SOURCES})
    target_compile_definitions(AilaCommandSubmissionBench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_include_directories(AilaCommandSubmissionBench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        "${AILA_LEVEL_ZERO_SDK_ROOT}/include"
    )
    target_link_libraries(AilaCommandSubmissionBench PRIVATE
        "${AILA_LEVEL_ZERO_SDK_ROOT}/lib/ze_loader.lib"
    )
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
    set(AILA_RUNTIME_SEARCH_DIRS "${AILA_COMPILER_BIN_DIR}")

    get_filename_component(AILA_ONEAPI_ROOT "${AILA_COMPILER_BIN_DIR}/../../.." ABSOLUTE)
    file(GLOB AILA_TBB_BIN_DIRS "${AILA_ONEAPI_ROOT}/tbb/*/bin")
    list(APPEND AILA_RUNTIME_SEARCH_DIRS ${AILA_TBB_BIN_DIRS})

    set(AILA_ONEAPI_RUNTIME_ROOT_PATTERNS
        "ur_loader.dll"
        "ur_win_proxy_loader.dll"
        "ur_adapter_opencl.dll"
        "ur_adapter_level_zero.dll"
        "ur_adapter_level_zero_v2.dll"
        "umf.dll"
        "xptifw.dll"
        "sycl-jit.dll"
        "OpenCL.dll"
        "intelocl64.dll"
        "common_clang64.dll"
        "libhwloc-*.dll"
        "tbb12.dll"
        "tbbbind.dll"
        "tbbbind_2_0.dll"
        "tbbbind_2_5.dll"
        "tbbmalloc.dll"
        "tbbmalloc_proxy.dll"
        "tcm.dll"
    )
    set(AILA_ONEAPI_RUNTIME_ROOT_DLLS)
    foreach(pattern IN LISTS AILA_ONEAPI_RUNTIME_ROOT_PATTERNS)
        file(GLOB AILA_ONEAPI_RUNTIME_MATCHES LIST_DIRECTORIES FALSE "${AILA_COMPILER_BIN_DIR}/${pattern}")
        list(APPEND AILA_ONEAPI_RUNTIME_ROOT_DLLS ${AILA_ONEAPI_RUNTIME_MATCHES})
        foreach(AILA_TBB_BIN_DIR IN LISTS AILA_TBB_BIN_DIRS)
            file(GLOB AILA_TBB_RUNTIME_MATCHES LIST_DIRECTORIES FALSE "${AILA_TBB_BIN_DIR}/${pattern}")
            list(APPEND AILA_ONEAPI_RUNTIME_ROOT_DLLS ${AILA_TBB_RUNTIME_MATCHES})
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES AILA_ONEAPI_RUNTIME_ROOT_DLLS)

    get_filename_component(AILA_LINKER_BIN_DIR "${CMAKE_LINKER}" DIRECTORY)
    if(EXISTS "${AILA_LINKER_BIN_DIR}/msvcp140.dll")
        list(APPEND AILA_RUNTIME_SEARCH_DIRS "${AILA_LINKER_BIN_DIR}")

        get_filename_component(AILA_MSVC_DIR "${AILA_LINKER_BIN_DIR}/../../../../../.." ABSOLUTE)
        get_filename_component(AILA_MSVC_TOOLS_VERSION "${AILA_LINKER_BIN_DIR}/../../.." NAME)
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" AILA_MSVC_VERSION_PREFIX "${AILA_MSVC_TOOLS_VERSION}")

        if(AILA_MSVC_VERSION_PREFIX)
            file(GLOB AILA_MSVC_REDIST_DIRS "${AILA_MSVC_DIR}/Redist/MSVC/${AILA_MSVC_VERSION_PREFIX}*/x64/Microsoft.VC*.CRT")
        endif()
        if(NOT AILA_MSVC_REDIST_DIRS)
            file(GLOB AILA_MSVC_REDIST_DIRS "${AILA_MSVC_DIR}/Redist/MSVC/*/x64/Microsoft.VC*.CRT")
        endif()

        set(AILA_MSVC_RUNTIME_ROOT_DIRS)
        if(AILA_MSVC_REDIST_DIRS)
            list(SORT AILA_MSVC_REDIST_DIRS)
            list(REVERSE AILA_MSVC_REDIST_DIRS)
            list(GET AILA_MSVC_REDIST_DIRS 0 AILA_MSVC_REDIST_DIR)
            set(AILA_MSVC_REDIST_DIRS "${AILA_MSVC_REDIST_DIR}")
            list(APPEND AILA_RUNTIME_SEARCH_DIRS ${AILA_MSVC_REDIST_DIRS})
            list(APPEND AILA_MSVC_RUNTIME_ROOT_DIRS ${AILA_MSVC_REDIST_DIRS})
        else()
            list(APPEND AILA_MSVC_RUNTIME_ROOT_DIRS "${AILA_LINKER_BIN_DIR}")
        endif()

        set(AILA_MSVC_RUNTIME_ROOT_PATTERNS
            "concrt140.dll"
            "msvcp140*.dll"
            "vccorlib140.dll"
            "vcruntime140*.dll"
        )
        foreach(AILA_MSVC_RUNTIME_DIR IN LISTS AILA_MSVC_RUNTIME_ROOT_DIRS)
            foreach(pattern IN LISTS AILA_MSVC_RUNTIME_ROOT_PATTERNS)
                file(GLOB AILA_MSVC_RUNTIME_MATCHES LIST_DIRECTORIES FALSE "${AILA_MSVC_RUNTIME_DIR}/${pattern}")
                list(APPEND AILA_ONEAPI_RUNTIME_ROOT_DLLS ${AILA_MSVC_RUNTIME_MATCHES})
            endforeach()
        endforeach()
        list(REMOVE_DUPLICATES AILA_ONEAPI_RUNTIME_ROOT_DLLS)
    endif()

    if(DNNL_DLL_LOCATION)
        get_filename_component(DNNL_DLL_DIR "${DNNL_DLL_LOCATION}" DIRECTORY)
        list(APPEND AILA_RUNTIME_SEARCH_DIRS "${DNNL_DLL_DIR}")
    endif()
    list(REMOVE_DUPLICATES AILA_RUNTIME_SEARCH_DIRS)

    set(AILA_COPY_RUNTIME_DEPENDENCIES_TEMPLATE [=[
set(target_path "${TARGET_PATH}")
set(target_dir "${TARGET_DIR}")
set(target_kind "${TARGET_KIND}")

set(search_dirs [==[@AILA_RUNTIME_SEARCH_DIRS@]==])
set(extra_root_dlls [==[@AILA_ONEAPI_RUNTIME_ROOT_DLLS@]==])
list(APPEND search_dirs "${target_dir}")

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
    message(WARNING "Unresolved runtime dependencies for ${target_path}: ${unresolved_deps}")
endif()
]=])
    string(CONFIGURE "${AILA_COPY_RUNTIME_DEPENDENCIES_TEMPLATE}" AILA_COPY_RUNTIME_DEPENDENCIES_SCRIPT @ONLY)
    file(GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/copy_runtime_dependencies.cmake"
        CONTENT "${AILA_COPY_RUNTIME_DEPENDENCIES_SCRIPT}"
    )

    function(copy_runtime_dependencies target target_kind)
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

    copy_runtime_dependencies(AilaAliaRealSmoke EXECUTABLE)
    if(TARGET AilaCommandSubmissionBench)
        copy_runtime_dependencies(AilaCommandSubmissionBench EXECUTABLE)
    endif()
    copy_runtime_dependencies(AilaShared LIBRARY)

    set(AILA_RELEASE_BIN_DIR "${CMAKE_BINARY_DIR}/Release/bin")
    set(AILA_RELEASE_TARGETS AilaShared AilaAliaRealSmoke)

    set(AILA_RELEASE_COPY_COMMANDS)
    foreach(AILA_RELEASE_TARGET IN LISTS AILA_RELEASE_TARGETS)
        get_target_property(AILA_RELEASE_TARGET_TYPE ${AILA_RELEASE_TARGET} TYPE)
        if(AILA_RELEASE_TARGET_TYPE STREQUAL "SHARED_LIBRARY")
            set(AILA_RELEASE_TARGET_KIND LIBRARY)
        else()
            set(AILA_RELEASE_TARGET_KIND EXECUTABLE)
        endif()
        list(APPEND AILA_RELEASE_COPY_COMMANDS
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${AILA_RELEASE_TARGET}>" "${AILA_RELEASE_BIN_DIR}"
            COMMAND ${CMAKE_COMMAND}
                "-DTARGET_PATH=$<TARGET_FILE:${AILA_RELEASE_TARGET}>"
                "-DTARGET_DIR=${AILA_RELEASE_BIN_DIR}"
                "-DTARGET_KIND=${AILA_RELEASE_TARGET_KIND}"
                -P "${CMAKE_CURRENT_BINARY_DIR}/copy_runtime_dependencies.cmake"
        )
    endforeach()

    add_custom_target(release
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${AILA_RELEASE_BIN_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${AILA_RELEASE_BIN_DIR}"
        ${AILA_RELEASE_COPY_COMMANDS}
        COMMENT "Staging release binaries in ${AILA_RELEASE_BIN_DIR}"
        VERBATIM
    )
    add_dependencies(release ${AILA_RELEASE_TARGETS})
endif()
