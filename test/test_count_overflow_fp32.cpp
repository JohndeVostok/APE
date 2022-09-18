#include "curand.h"
#include "curand_kernel.h"
#include "iostream"
#include "common.h"
#include "cuda_utils.h"
#include "kernel.h"

int host_count_overflow(const float* src, int size) {
    int count = 0;

    for (int i = 0; i < size; i++) {
        count += (src[i] < ape::FP32F_MIN || src[i] > ape::FP32F_MAX);
    }

    return count;
}

//#define DEBUG

int main() {
    float *d_array;
    float *h_array;

    curandGenerator_t gen;
    curandSafeCall(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT));

    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);

    for (int size = 128; size < 32768; size <<= 1) {
        h_array = (float*) malloc(sizeof(float) * size*size);
        cudaSafeCall(cudaMalloc((void**) &d_array, sizeof(float) * size*size));
        curandSafeCall(curandGenerateLogNormal(gen, d_array, size*size, 0, 10));
        cudaSafeCall(cudaMemcpy(h_array, d_array, size*size * sizeof(float),
            cudaMemcpyDeviceToHost));

        #ifdef DEBUG
        for (int i = 0; i < size; i++)
            std::cout << h_array[i] << std::endl;
        #endif

        int h_count, d_count;
        h_count = host_count_overflow(h_array, size*size);

        for (int i = 0; i < 128; i++)
            ape::count_overflow_fp32(d_array, size, size);

        float ms;
        cudaEventRecord(st);
        for (int i = 0; i < 128; i++)
            d_count = ape::count_overflow_fp32(d_array, size, size);
        cudaEventRecord(ed);
        cudaEventSynchronize(st);
        cudaEventSynchronize(ed);
        cudaEventElapsedTime(&ms, st, ed);

        if (h_count == d_count) {
            std::cout << "correct" << std::endl;
        } else {
            std::cout << "unmatched: " << h_count << " " << d_count << std::endl;
        }

        std::cout << size << "x" << size << ": " << size*size * sizeof(int) * 128.0 / (ms * 1024.0 * 1024.0) << " GB/s" << std::endl;
        cudaFree(d_array);
    }

    return 0;
}