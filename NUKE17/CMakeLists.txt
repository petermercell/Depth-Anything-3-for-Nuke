cmake_minimum_required(VERSION 3.18)
project(TRT_DepthAnything3 VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

#-------------------------------------------------------------------------------
# Nuke NDK
#-------------------------------------------------------------------------------
if(NOT DEFINED NUKE_ROOT)
    if(EXISTS "/opt/Nuke15.1v10")
        set(NUKE_ROOT "/opt/Nuke15.1v10")
    elseif(EXISTS "/usr/local/Nuke15.1v10")
        set(NUKE_ROOT "/usr/local/Nuke15.1v10")
    else()
        message(FATAL_ERROR "NUKE_ROOT not set and Nuke not found")
    endif()
endif()

message(STATUS "Nuke root: ${NUKE_ROOT}")

if(NOT EXISTS "${NUKE_ROOT}/include/DDImage/Iop.h")
    message(FATAL_ERROR "Invalid NUKE_ROOT: ${NUKE_ROOT} - DDImage headers not found")
endif()

set(NUKE_INCLUDE_DIRS "${NUKE_ROOT}/include")

find_library(DDIMAGE_LIBRARY
    NAMES DDImage
    PATHS "${NUKE_ROOT}"
    NO_DEFAULT_PATH
    REQUIRED
)

#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")

set(CUDA_LIB_DIR "${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "CUDA lib dir: ${CUDA_LIB_DIR}")

#-------------------------------------------------------------------------------
# TensorRT (static)
#-------------------------------------------------------------------------------
set(TENSORRT_ROOT "/opt/TensorRT-10.9.0.34" CACHE PATH "TensorRT installation root")

if(NOT EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
    message(FATAL_ERROR "TensorRT headers not found in ${TENSORRT_ROOT}/include")
endif()

message(STATUS "TensorRT found: ${TENSORRT_ROOT}")

set(TRT_STATIC_LIB_DIR "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib")

if(NOT EXISTS "${TRT_STATIC_LIB_DIR}/libnvinfer_static.a")
    message(FATAL_ERROR "TensorRT static libs not found in ${TRT_STATIC_LIB_DIR}")
endif()

#-------------------------------------------------------------------------------
# Embedded TensorRT engine file
#-------------------------------------------------------------------------------
set(ENGINE_FILE "${CMAKE_SOURCE_DIR}/DepthAnything3_mono_large_2058x1092_fp16_cuda128.engine"
    CACHE FILEPATH "TensorRT engine file to embed")

if(NOT EXISTS "${ENGINE_FILE}")
    message(FATAL_ERROR "Engine file not found: ${ENGINE_FILE}\n"
                        "Build it with:\n"
                        "  trtexec --onnx=DepthAnything3_mono_large_2058x1092.onnx \\\n"
                        "          --saveEngine=DepthAnything3_mono_large_2058x1092_fp32_cuda128.engine \\\n"
                        "          --memPoolSize=workspace:16G")
endif()

file(SIZE "${ENGINE_FILE}" ENGINE_SIZE)
math(EXPR ENGINE_SIZE_MB "${ENGINE_SIZE} / 1048576")
message(STATUS "Embedding engine: ${ENGINE_FILE} (${ENGINE_SIZE_MB} MB)")

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/engine_data.o
    COMMAND ${CMAKE_COMMAND} -E copy ${ENGINE_FILE} ${CMAKE_BINARY_DIR}/engine.bin
    COMMAND objcopy
        --input binary
        --output elf64-x86-64
        --binary-architecture i386:x86-64
        --rename-section .data=.rodata,alloc,load,readonly,data,contents
        engine.bin engine_data.o
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${ENGINE_FILE}
    COMMENT "Embedding TensorRT engine into object file"
)

#-------------------------------------------------------------------------------
# Source files
#-------------------------------------------------------------------------------
set(SOURCES
    TRT_DepthAnything3.cpp
    ${CMAKE_BINARY_DIR}/engine_data.o
)

#-------------------------------------------------------------------------------
# Build plugin
#-------------------------------------------------------------------------------
add_library(TRT_DepthAnything3 SHARED ${SOURCES})

set_source_files_properties(${CMAKE_BINARY_DIR}/engine_data.o PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
)

target_include_directories(TRT_DepthAnything3 PRIVATE
    ${NUKE_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

#-------------------------------------------------------------------------------
# Static linking
#-------------------------------------------------------------------------------
target_link_libraries(TRT_DepthAnything3 PRIVATE
    ${DDIMAGE_LIBRARY}

    ${TRT_STATIC_LIB_DIR}/libnvinfer_static.a

    ${CUDA_LIB_DIR}/libcudart_static.a
    ${CUDA_LIB_DIR}/libcublas_static.a
    ${CUDA_LIB_DIR}/libcublasLt_static.a

    ${CUDA_LIB_DIR}/libnvrtc_static.a
    ${CUDA_LIB_DIR}/libnvrtc-builtins_static.a

    ${CUDA_LIB_DIR}/libnvptxcompiler_static.a
    ${CUDA_LIB_DIR}/libnvJitLink_static.a

    ${CUDA_LIB_DIR}/libculibos.a

    dl
    rt
    pthread
)

#-------------------------------------------------------------------------------
# Linux-specific settings
#-------------------------------------------------------------------------------
set_target_properties(TRT_DepthAnything3 PROPERTIES
    SUFFIX ".so"
    PREFIX ""
)

target_compile_options(TRT_DepthAnything3 PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
    -fvisibility=hidden
    -O3
)

set_target_properties(TRT_DepthAnything3 PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

#-------------------------------------------------------------------------------
# Installation
#-------------------------------------------------------------------------------
install(TARGETS TRT_DepthAnything3
    LIBRARY DESTINATION .
)

#-------------------------------------------------------------------------------
# Summary
#-------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "==========================================")
message(STATUS "TRT_DepthAnything3 Configuration Summary")
message(STATUS "==========================================")
message(STATUS "  Nuke:          ${NUKE_ROOT}")
message(STATUS "  CUDA:          ${CUDAToolkit_VERSION}")
message(STATUS "  TensorRT:      ${TENSORRT_ROOT} (STATIC)")
message(STATUS "  Engine:        ${ENGINE_FILE} (${ENGINE_SIZE_MB} MB, embedded)")
message(STATUS "  Resolution:    2058x1092 (fixed)")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  Target needs:  NVIDIA driver only (libcuda.so)")
message(STATUS "==========================================")
message(STATUS "")
