cmake_minimum_required(VERSION 3.9)
project(APE)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CUDA_NVCC_FLAGS "-std=c++11 -use_fast_math -ccbin g++ -expt-relaxed-constexpr") # --ptxas-options=-v 
set(CUDA_VERBOSE_BUILD on)

include_directories(include)
add_subdirectory("src")

set(TESTS test_fp32_cublas test_fp32_fp16 test_fp32_bf16)
foreach(TEST IN LISTS TESTS)
    cuda_add_executable(${TEST} test/${TEST}.cpp)
    target_link_libraries(${TEST} ape -lcublas -lcurand)
    set_target_properties(${TEST} PROPERTIES CUDA_ARCHITECTURES "80")
endforeach()

