cmake_minimum_required(VERSION 3.9)
project(APE)

set(CMAKE_CXX_STANDARD 20)

find_package(CUDA REQUIRED)
enable_language(CUDA)
find_package(CUDAToolkit)

set(CUDA_NVCC_FLAGS "-use_fast_math") # --ptxas-options=-v 
include_directories(include)

file(GLOB_RECURSE SRC_CUDA src/kernel/*.cu)
add_library(ape_cuda SHARED ${SRC_CUDA})

file(GLOB_RECURSE SRC src/*.cpp)
add_library(ape SHARED ${SRC})
target_link_libraries(ape ape_cuda CUDA::cublas CUDA::curand CUDA::cudart)


if ("${ARCH}" STREQUAL "")
  set(ARCH "80")
endif()
if ("${TEST}" STREQUAL "")
  set(TEST "ON")
endif()
if ("${ARCH}" STREQUAL "80")
  message("APE ARCH set to sm80")
  add_compile_definitions(ARCH=SM80)
  set_target_properties(ape_cuda PROPERTIES CUDA_ARCHITECTURES "80")
elseif ("${ARCH}" STREQUAL "70")
  message("APE ARCH set to sm70")
  add_compile_definitions(ARCH=SM70)
  set_target_properties(ape_cuda PROPERTIES CUDA_ARCHITECTURES "80")
else()
  message(FATAL_ERROR "Invalid ARCH (set ti 70 or 80)")
  message("error")
endif()

file(GLOB_RECURSE SRC_TEST test/test_gemm_fp32.cpp)
set(TESTS test_gemm_fp32_auto test_gemm_fp32_cublas test_gemm_fp32_fp32f test_gemm_fp32_fp32b test_gemm_fp32_fp32t)
foreach(TEST IN LISTS TESTS)
    cuda_add_executable(${TEST} test/${TEST}.cpp)
    target_link_libraries(${TEST} ape)
endforeach()

