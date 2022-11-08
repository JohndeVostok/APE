#include "kernel.h"
#include "thrust/device_vector.h"
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

namespace ape {

struct OpFP32F {
    __device__ int operator()(float x) { return (fabs(x) > FP32F_MAX); }
};

struct OpFP32FStrict {
    __device__ int operator()(float x) { return (fabs(x) < FP32F_MIN || fabs(x) > FP32F_MAX); }
};

__global__ void kernel_create_mask_fp32(const float *src, size_t row, size_t col, ApeTrans trans, int8_t *mask) {
    extern __shared__ char smem[];
    int8_t *shmem = (int8_t *)&smem[0];
    uint32_t warp_id = threadIdx.y / 32;
    uint32_t lane_id = threadIdx.y % 32;
    if (blockIdx.y * blockDim.y + threadIdx.y >= col) {
        return;
    }

    const float *base;
    size_t step;
    if (trans == APE_TRANS_T) {
        base = src + blockIdx.x * AUTO_BLOCK * col + blockIdx.y * AUTO_BLOCK + threadIdx.y;
        step = col;
    } else {
        base = src + blockIdx.y * AUTO_BLOCK * row + blockIdx.x * AUTO_BLOCK + threadIdx.y * row;
        step = 1;
    }

    int8_t flag = 1;
    for (int i = 0; i < AUTO_BLOCK && blockIdx.x * AUTO_BLOCK + i < row; i++) {
        float tmp = fabs(base[i * step]);
        if (tmp < FP32F_MIN || tmp > FP32F_MAX) {
            flag = 0;
        }
    }

    for (int i = 16; i > 0; i >>= 1) {
        int tmp = __shfl_down_sync(0xffffffff, flag, i); // warp shuffle
        if (tmp == 0) {
            flag = 0;
        }
    }

    if (lane_id == 0) {
        shmem[warp_id] = flag;
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        for (int i = 0; i < (blockDim.y - 1) / 32 + 1; i++) {
            if (shmem[i] == 0) {
                flag = 0;
                break;
            }
        }

        if (trans == APE_TRANS_T) {
            mask[blockIdx.x * gridDim.y + blockIdx.y] = flag;
        } else {
            mask[blockIdx.y * gridDim.x + blockIdx.x] = flag;
        }
    }
}

void create_mask_fp32(const float *src, size_t row, size_t col, ApeTrans trans, int8_t *mask) {
    dim3 grid((row - 1) / AUTO_BLOCK + 1, (col - 1) / AUTO_BLOCK + 1, 1);
    dim3 block(1, AUTO_BLOCK, 1);
    kernel_create_mask_fp32<<<grid, block, ((AUTO_BLOCK - 1) / 32 + 1) * sizeof(int8_t)>>>(src, row, col, trans, mask);
    cudaCheckError();
}

int count_overflow_fp32f(const float *src, size_t row, size_t col) {
    thrust::device_ptr<float> d_src(const_cast<float *>(src));
    return thrust::transform_reduce(d_src, d_src + row * col, OpFP32F(), 0, thrust::plus<int>());
}

int count_overflow_fp32f_strict(const float *src, size_t row, size_t col) {
    thrust::device_ptr<float> d_src(const_cast<float *>(src));
    return thrust::transform_reduce(d_src, d_src + row * col, OpFP32FStrict(), 0, thrust::plus<int>());
}

} // namespace ape
