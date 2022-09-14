#pragma once

#include <cassert>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <memory>

#define ape_error(str) __ape_error(str, __FILE__, __LINE__)
#define ape_warning(str) __ape_warning(str, __FILE__, __LINE__)
#define ape_info(str) __ape_info(str, __FILE__, __LINE__)
#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define cudaCheckError() __cudaCheckError(__FILE__, __LINE__)
#define cublasSafeCall(err) __cublasSafeCall(err, __FILE__, __LINE__)
#define curandSafeCall(err) __curandSafeCall(err, __FILE__, __LINE__)

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

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
    if (err != cudaSuccess) {
        std::cout << "[ERROR] " << file << "::" << line << ": cudaSafeCall() failed. " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
    return;
}

inline void __cudaCheckError(const char *file, const int line) {
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "[ERROR] " << file << "::" << line << ": cudaCheckError() failed. " << cudaGetErrorString(err)
                  << std::endl;
        exit(-1);
    }

#ifdef DEBUG
    // This checking will affect performance.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "[ERROR] " << file << "::" << line << ": cudaCheckError() with sync failed. " << cudaGetErrorString(err)
                  << std::endl;
        exit(-1);
    }
#endif

    return;
}

inline const char *cublasGetErrorString(cublasStatus_t err) {
    switch (err) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::cout << "[ERROR]" << file << "::" << line << ": cublasSafeCall() failed. " << cublasGetErrorString(err)
                  << std::endl;
        exit(-1);
    }
}

inline void __curandSafeCall(curandStatus_t err, const char *file, const int line) {
    if (err != CURAND_STATUS_SUCCESS) {
        std::cout << "[ERROR]" << file << "::" << line << ": curandSafeCall() failed. " << err << std::endl;
        exit(-1);
    }
}

namespace ape {

class APEHandler {
  private:
    APEHandler() {}
    static APEHandler *instance;
    cublasHandle_t ape_cublas_handle;
    void *buf;
    size_t buf_size;

  public:
    static inline APEHandler *getInstance() {
        static APEHandler instance;
        return &instance;
    }
    static inline cublasHandle_t getCublasHandle() { return getInstance()->ape_cublas_handle; }
    static inline void initCublas() {
        cublasHandle_t handle;
        cublasSafeCall(cublasCreate(&handle));
        cublasSafeCall(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        getInstance()->ape_cublas_handle = handle;
    }
    static inline void initBuffer(size_t buf_size) {
        void *buf;
        cudaSafeCall(cudaMalloc((void **)&buf, buf_size));
        getInstance()->buf_size = buf_size;
        getInstance()->buf = buf;
    }
    static inline size_t getBufSize() { return getInstance()->buf_size; }
    static inline void *getBuf() { return getInstance()->buf; }
};

} // namespace ape