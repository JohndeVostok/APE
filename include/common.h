#pragma once

#include <cassert>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

namespace ape {

extern cublasHandle_t ape_cublas_handle;
extern cudaDeviceProp ape_gpu_prop;

void apeInit();

#define ape_error(str) __ape_error(str, __FILE__, __LINE__)
#define ape_warning(str) __ape_warning(str, __FILE__, __LINE__)
#define ape_info(str) __ape_info(str, __FILE__, __LINE__)

inline void __ape_error(std::string str, const char *file, const int line) {
    std::cout << "[ERROR] " << file << "::" << line << " " << str << std::endl;
    exit(-1);
}

inline void __ape_warning(std::string str, const char *file, const int line) {
    std::cout << "[WARNING] " << file << "::" << line << " " << str << std::endl;
#if DEBUG
    exit(-1);
#endif
}

inline void __ape_info(std::string str, const char *file, const int line) {
    std::cout << "[INFO] " << file << "::" << line << " " << str << std::endl;
}

enum ApeTrans {
    APE_TRANS_N = 0,
    APE_TRANS_T,
};

enum ApeAlgo {
    APE_ALGO_AUTO = 0,
    APE_ALGO_CUBLAS,
    APE_ALGO_FP32F,
    APE_ALGO_FP32B,
    APE_ALGO_FP32T,
    APE_ALGO_INT16,
};

#define FP16_MAX 65504.0f
#define FP16_MIN 3.1e-5f
} // namespace ape