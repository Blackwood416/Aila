# required cmake version
cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)

if(WIN32)
        set(CMAKE_CXX_COMPILER "icx-cl")
else()
        set(CMAKE_CXX_COMPILER "icpx")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project (Aila CXX)

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fsycl /Qstd=c++17 /EHsc -fsycl-device-code-split=per_kernel")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fsycl -std=c++17 -fsycl-device-code-split=per_kernel")
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsycl")

set(SOURCES
    src/main.cpp
    src/ops/Linear.cpp
    src/ops/NormOps.cpp
    src/ops/AttentionOps.cpp
    src/ops/ElementwiseOps.cpp
    src/ops/SamplingOps.cpp
    src/ops/Transpose.cpp
    src/memory/KVCache.cpp
    src/models/Qwen3.cpp
    src/utils/Tokenizer.cpp
    src/utils/MemoryMappedFile.cpp
    src/utils/SafeTensors.cpp
    src/profile/Device.cpp
)
set(SIMDJSON_SOURCES
    third_party/simdjson/simdjson.cpp
)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src/utils)
include_directories(${PROJECT_SOURCE_DIR}/src/core)
include_directories(${PROJECT_SOURCE_DIR}/src/ops)
include_directories(${PROJECT_SOURCE_DIR}/src/memory)
include_directories(${PROJECT_SOURCE_DIR}/src/models)

find_package(dnnl REQUIRED)
add_library(simdjson STATIC ${SIMDJSON_SOURCES})
target_include_directories(simdjson PUBLIC ${PROJECT_SOURCE_DIR}/third_party/simdjson)
add_executable (Aila ${SOURCES})
target_compile_definitions(Aila PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(Aila PRIVATE DNNL::dnnl simdjson)
