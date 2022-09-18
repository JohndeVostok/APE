#include "cstdlib"
#include "iostream"
#include "common.h"
#include "cuda_utils.h"
#include "kernel.h"

int host_count_overflow(const int16_t* src, int size) {
    int count = 0;

    for (int i = 0; i < size; i++) {
        count += (src[i] > ape::INT16C_MAX);
    }

    return count;
}

void host_randint(int16_t* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = rand() % 65536 - 32768;
    }
}

//#define DEBUG

int main() {
    int16_t *d_array;
    int16_t *h_array;

    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);

    for (int size = 128; size < 32768; size <<= 1) {
        h_array = (int16_t*) malloc(sizeof(int16_t) * size*size);
        host_randint(h_array, size*size);
        cudaSafeCall(cudaMalloc((void**) &d_array, sizeof(int16_t) * size*size));
        cudaSafeCall(cudaMemcpy(d_array, h_array, size*size * sizeof(int16_t),
            cudaMemcpyHostToDevice));

        #ifdef DEBUG
        for (int i = 0; i < size; i++)
            std::cout << h_array[i] << std::endl;
        #endif

        int h_count, d_count;
        h_count = host_count_overflow(h_array, size*size);

        for (int i = 0; i < 128; i++)
            ape::count_overflow_int16c(d_array, size, size);

        float ms;
        cudaEventRecord(st);
        for (int i = 0; i < 128; i++)
            d_count = ape::count_overflow_int16c(d_array, size, size);
        cudaEventRecord(ed);
        cudaEventSynchronize(st);
        cudaEventSynchronize(ed);
        cudaEventElapsedTime(&ms, st, ed);

        if (h_count == d_count) {
            std::cout << "correct" << std::endl;
        } else {
            std::cout << "unmatched: " << h_count << " " << d_count << std::endl;
        }

        std::cout << size << "x" << size << ": " << size*size * sizeof(int16_t) * 128.0 / (ms * 1024.0 * 1024.0) << " GB/s" << std::endl;
        cudaFree(d_array);
    }

    return 0;
}