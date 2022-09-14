#include <iostream>

#include "kernel.h"

void host_create_mask_fp32(const float *src, int m, int n, ape::ApeTrans trans, int8_t *mask) {
    int row_blocks = (m - 1) / ape::AUTO_BLOCK + 1;
    int col_blocks = (n - 1) / ape::AUTO_BLOCK + 1;

    for (int block_m = 0; block_m < row_blocks; block_m++) {
        for (int block_n = 0; block_n < col_blocks; block_n++) {
            if (trans == ape::APE_TRANS_T) {
                mask[block_m * col_blocks + block_n] = 1;
            } else {
                mask[block_n * row_blocks + block_m] = 1;
            }
            for (int i = 0; i < ape::AUTO_BLOCK; i++) {     // row
                for (int j = 0; j < ape::AUTO_BLOCK; j++) { // col
                    if (trans == ape::APE_TRANS_T) {
                        int index = (i + block_m * ape::AUTO_BLOCK) * n + block_n * ape::AUTO_BLOCK + j;
                        if (index >= m * n)
                            goto kernelend;
                        if (src[index] < ape::FP32F_MIN || src[index] > ape::FP32F_MAX) {
                            mask[block_m * col_blocks + block_n] = 0;
                            goto kernelend;
                        }
                    } else {
                        int index = (j + block_n * ape::AUTO_BLOCK) * m + block_m * ape::AUTO_BLOCK + i;
                        if (index >= m * n)
                            goto kernelend;
                        if (src[index] < ape::FP32F_MIN || src[index] > ape::FP32F_MAX) {
                            mask[block_n * row_blocks + block_m] = 0;
                            goto kernelend;
                        }
                    }
                }
            }

        kernelend:
            continue;
        }
    }
}

//#define DEBUG
//#define RESULT

int main() {
    float *d_array, *h_array;
    int8_t *h_mask, *d_mask;

    curandGenerator_t gen;
    curandSafeCall(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);

    for (int size = 128; size <= 8192; size <<= 1) {
        int num_blocks = (size - 1) / ape::AUTO_BLOCK + 1;
        h_array = (float *)malloc(sizeof(float) * size * size);
        h_mask = (int8_t *)malloc(sizeof(int8_t) * (num_blocks * num_blocks));
        int8_t *tmp = (int8_t *)malloc(sizeof(int8_t) * (num_blocks * num_blocks));
        memset(h_mask, 0, sizeof(int8_t) * (num_blocks * num_blocks));
        cudaSafeCall(cudaMalloc((void **)&d_array, sizeof(float) * size * size));
        curandSafeCall(curandGenerateUniform(gen, d_array, size * size));
        cudaSafeCall(cudaMemcpy(h_array, d_array, size * size * sizeof(float), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMalloc((void **)&d_mask, sizeof(int8_t) * (num_blocks * num_blocks)));
        cudaSafeCall(cudaMemset(d_mask, 0, sizeof(int8_t) * (num_blocks * num_blocks)));

#ifdef DEBUG
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                std::cout << h_array[i * size + j] << " ";
            }

            std::cout << std::endl;
        }
#endif

        host_create_mask_fp32(h_array, size, size, ape::APE_TRANS_N, h_mask);

        for (int i = 0; i < 128; i++)
            ape::create_mask_fp32(d_array, size, size, ape::APE_TRANS_N, d_mask);

        float ms;
        cudaEventRecord(st);
        for (int i = 0; i < 128; i++)
            ape::create_mask_fp32(d_array, size, size, ape::APE_TRANS_N, d_mask);
        cudaEventRecord(ed);
        cudaEventSynchronize(st);
        cudaEventSynchronize(ed);
        cudaEventElapsedTime(&ms, st, ed);

        cudaSafeCall(cudaMemcpy(tmp, d_mask, sizeof(int8_t) * (num_blocks * num_blocks), cudaMemcpyDeviceToHost));

        int i;
        for (i = 0; i < num_blocks * num_blocks; i++) {
            if (tmp[i] != h_mask[i]) {
                std::cout << "unmatched at " << i << ": " << int(tmp[i]) << " " << int(h_mask[i]) << std::endl;
                break;
            }
        }

#ifdef RESULT
        for (int i = 0; i < num_blocks; i++) {
            for (int j = 0; j < num_blocks; j++) {
                std::cout << int(tmp[i * num_blocks + j]) << " " << int(h_mask[i * num_blocks + j]) << std::endl;
            }
        }
#endif

        if (i == num_blocks * num_blocks)
            std::cout << "correct" << std::endl;

        std::cout << size << "x" << size << ": " << size * size * sizeof(float) * 128.0 / (ms * 1024.0 * 1024.0) << " GB/s"
                  << std::endl;
        free(h_array);
        free(h_mask);
        free(tmp);
        cudaFree(d_mask);
        cudaFree(d_array);
    }

    return 0;
}