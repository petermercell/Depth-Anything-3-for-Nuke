# ============================================================================
# CMakeLists.txt  —  LibTorchDepthAnything3 (Nuke 17 / libtorch 2.7.1 / CUDA 12.8).
# Same recipe as Peter's MegaFlow build:
#   * Link Nuke's bundled .so (libtorch / libtorch_cpu / libtorch_cuda /
#     libc10 / libc10_cuda) — guarantees ABI parity with the libtorch the
#     plugin will load inside.
#   * Headers from a version-matched conda env (da3_nuke17, torch 2.7.1).
#   * -D_GLIBCXX_USE_CXX11_ABI=1.
#   * BUILD_RPATH = Nuke root + CUDA libdir so the test build runs without
#     LD_LIBRARY_PATH; INSTALL_RPATH = $ORIGIN + Nuke for install.
# No OpenCV: Nuke hands us pixels directly.
# ============================================================================
cmake_minimum_required(VERSION 3.18)
project(LibTorchDepthAnything3 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(SRC "${CMAKE_SOURCE_DIR}/src")

# ---- Nuke 17.0 (provides libtorch 2.7.1) ----
set(NUKE_ROOT "/opt/Nuke17.0v1" CACHE PATH "Nuke 17.0 install")
if(NOT EXISTS "${NUKE_ROOT}/libtorch.so")
    message(FATAL_ERROR "Nuke libtorch not found in ${NUKE_ROOT}")
endif()
if(NOT EXISTS "${NUKE_ROOT}/include/DDImage/Iop.h")
    message(FATAL_ERROR "Nuke NDK headers not found in ${NUKE_ROOT}/include")
endif()
set(NUKE_INCLUDE "${NUKE_ROOT}/include")

find_library(DDIMAGE_LIBRARY NAMES DDImage
    PATHS "${NUKE_ROOT}" NO_DEFAULT_PATH REQUIRED)

# ---- libtorch headers (conda env, version-matched to Nuke's 2.7.1) ----
set(TORCH_INCLUDE_ROOT
    "$ENV{HOME}/miniconda3/envs/da3_nuke17/lib/python3.11/site-packages/torch/include"
    CACHE PATH "libtorch headers (conda)")
if(NOT EXISTS "${TORCH_INCLUDE_ROOT}/torch/script.h")
    message(FATAL_ERROR "torch/script.h not found in ${TORCH_INCLUDE_ROOT}\n"
        "Set -DTORCH_INCLUDE_ROOT=... to the include dir of a torch 2.7.1 install.")
endif()
set(TORCH_INCLUDE_DIRS
    "${TORCH_INCLUDE_ROOT}"
    "${TORCH_INCLUDE_ROOT}/torch/csrc/api/include")

set(TORCH_LIBS
    "${NUKE_ROOT}/libtorch.so"
    "${NUKE_ROOT}/libtorch_cpu.so"
    "${NUKE_ROOT}/libtorch_cuda.so"
    "${NUKE_ROOT}/libc10.so"
    "${NUKE_ROOT}/libc10_cuda.so")

# ---- CUDA 12.8 ----
if(NOT DEFINED CUDAToolkit_ROOT AND EXISTS "/usr/local/cuda-12.8")
    set(CUDAToolkit_ROOT "/usr/local/cuda-12.8")
endif()
find_package(CUDAToolkit REQUIRED)

# ---- the plugin ----
add_library(LibTorchDepthAnything3 SHARED
    ${SRC}/LibTorchDepthAnything3.cpp
    ${SRC}/torch_engine.cpp)

target_include_directories(LibTorchDepthAnything3 PRIVATE
    ${SRC}
    ${NUKE_INCLUDE}
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS})

target_compile_definitions(LibTorchDepthAnything3 PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_compile_options(LibTorchDepthAnything3 PRIVATE
    -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations
    -O3 -fvisibility=hidden)

target_link_libraries(LibTorchDepthAnything3 PRIVATE
    ${DDIMAGE_LIBRARY}
    ${TORCH_LIBS}
    CUDA::cudart
    dl rt pthread)

set_target_properties(LibTorchDepthAnything3 PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    BUILD_RPATH  "${NUKE_ROOT};${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "$ORIGIN:${NUKE_ROOT}")

install(TARGETS LibTorchDepthAnything3 LIBRARY DESTINATION .)

message(STATUS "")
message(STATUS "=== LibTorchDepthAnything3 ===")
message(STATUS "  Nuke:    ${NUKE_ROOT}")
message(STATUS "  Torch:   ${NUKE_ROOT}/libtorch.so (headers: ${TORCH_INCLUDE_ROOT})")
message(STATUS "  CUDA:    ${CUDAToolkit_VERSION}")
message(STATUS "  DDImage: ${DDIMAGE_LIBRARY}")
message(STATUS "===============================")
message(STATUS "")
