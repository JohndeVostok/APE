#include "cuda_utils.h"
#include "kernel.h"

namespace ape {
__global__ void kernel_create_mask_a(float *src, size_t m, size_t k, ApeTrans transa,
                                               int *mask) {
    extern __shared__ char smem[];
    int *shmem = (int *)&smem[0];
    uint32_t warp_id = threadIdx.y / 32;
    uint32_t lane_id = threadIdx.y % 32;

    if (blockIdx.y * blockDim.y + blockDim.y >= k || blockIdx.x * AUTO_BLOCK + AUTO_BLOCK >= m) {
        mask[blockIdx.x * gridDim.y + blockIdx.y] = 0;
        return;
    }

    float* base;
    int step;
    if (transa == APE_TRANS_T) {    
        base = src + blockIdx.x * AUTO_BLOCK * k +
                    blockIdx.y * AUTO_CHUNK + threadIdx.y;
        step = k;
    } else {
        base = src + blockIdx.y * AUTO_CHUNK * m + blockIdx.x * AUTO_BLOCK + threadIdx.y * m;
        step = 1;
    }

    int flag = 1;
    for (int i = 0; i < AUTO_BLOCK; i++) {
        float tmp = fabs(base[i * step]);
        if (tmp < FP16_MIN || tmp > FP16_MAX) {
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
        for (int i = 0; i < (blockDim.y-1) / 32 + 1; i++) {
            if (shmem[i] == 0) {
                flag = 0;
                break;
            }
        }
        mask[blockIdx.x * gridDim.y + blockIdx.y] = flag;
    }
}

void create_mask_a(float *src, size_t m, size_t k, ApeTrans transa,
                                               int *mask) {
    dim3 grid((m - 1) / AUTO_BLOCK + 1, (k - 1) / AUTO_CHUNK + 1, 1);
    dim3 block(1, AUTO_CHUNK, 1);

    kernel_create_mask_a<<<grid, block, ((AUTO_CHUNK - 1)/32+1) * sizeof (int)>>>(src, m, k, transa, mask);

    cudaCheckError();
}


__global__ void kernel_create_mask_b(float *src, size_t k, size_t n, ApeTrans transb,
                                               int *mask) {
    extern __shared__ char smem[];
    int *shmem = (int *)&smem[0];
    uint32_t warp_id = threadIdx.y / 32;
    uint32_t lane_id = threadIdx.y % 32;

    if (blockIdx.y * blockDim.y + blockDim.y >= k || blockIdx.x * AUTO_BLOCK + AUTO_BLOCK >= n) {
        mask[blockIdx.x * gridDim.y + blockIdx.y] = 0;
        return;
    }

    float* base;
    int step;
    if (transb == APE_TRANS_N) {
        base = src + blockIdx.x * AUTO_BLOCK * k +
                    blockIdx.y * AUTO_CHUNK + threadIdx.y;
        step = k;
    } else {
        base = src + blockIdx.x * AUTO_BLOCK +
                    blockIdx.y * AUTO_CHUNK * n + threadIdx.y * n;
        step = 1;
    }

    int flag = 1;
    for (int i = 0; i < AUTO_BLOCK; i++) {
        float tmp = fabs(base[i * step]);
        if (tmp < FP16_MIN || tmp > FP16_MAX) {
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
        for (int i = 0; i < (blockDim.y-1) / 32 + 1; i++) {
            if (shmem[i] == 0) {
                flag = 0;
                break;
            }
        }
        mask[blockIdx.x * gridDim.y + blockIdx.y] = flag;
    }
}

void create_mask_b(float *src, size_t k, size_t n, ApeTrans transb,
                                               int *mask) {
    dim3 grid((n - 1) / AUTO_BLOCK + 1, (k - 1) / AUTO_CHUNK + 1, 1);
    dim3 block(1, AUTO_CHUNK, 1);

    kernel_create_mask_b<<<grid, block, ((AUTO_CHUNK - 1)/32+1) * sizeof (int)>>>(src, k, n, transb, mask);

    cudaCheckError();
}
} // namespace gemm
