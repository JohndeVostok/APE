#include "cuda_utils.h"
#include "kernel.h"

namespace ape {
__global__ void kernel_create_mask_a_int16c(const int16_t *src, size_t m, size_t k, ApeTrans transa,
                                               int8_t *mask) {
    extern __shared__ char smem[];
    int8_t *shmem = (int8_t *)&smem[0];
    uint32_t warp_id = threadIdx.y / 32;
    uint32_t lane_id = threadIdx.y % 32;

    if (blockIdx.y * blockDim.y + blockDim.y >= k || blockIdx.x * AUTO_BLOCK + AUTO_BLOCK >= m) {
        mask[blockIdx.x * gridDim.y + blockIdx.y] = 0;
        return;
    }

    const int16_t* base;
    size_t step;
    if (transa == APE_TRANS_T) {    
        base = src + blockIdx.x * AUTO_BLOCK * k +
                    blockIdx.y * AUTO_CHUNK + threadIdx.y;
        step = k;
    } else {
        base = src + blockIdx.y * AUTO_CHUNK * m + blockIdx.x * AUTO_BLOCK + threadIdx.y * m;
        step = 1;
    }

    int8_t flag = 1;
    for (int i = 0; i < AUTO_BLOCK; i++) {
        int16_t tmp = base[i * step];
        if (tmp > INT16C_MAX) {
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

void create_mask_a_int16c(const int16_t *src, size_t m, size_t k, ApeTrans transa,
                                               int8_t *mask) {
    dim3 grid((m - 1) / AUTO_BLOCK + 1, (k - 1) / AUTO_CHUNK + 1, 1);
    dim3 block(1, AUTO_CHUNK, 1);

    kernel_create_mask_a_int16c<<<grid, block, ((AUTO_CHUNK - 1)/32+1) * sizeof (int8_t)>>>(src, m, k, transa, mask);

    cudaCheckError();
}


__global__ void kernel_create_mask_b_int16c(const int16_t *src, size_t k, size_t n, ApeTrans transb,
                                               int8_t *mask) {
    extern __shared__ char smem[];
    int8_t *shmem = (int8_t *)&smem[0];
    uint32_t warp_id = threadIdx.y / 32;
    uint32_t lane_id = threadIdx.y % 32;

    if (blockIdx.y * blockDim.y + blockDim.y >= k || blockIdx.x * AUTO_BLOCK + AUTO_BLOCK >= n) {
        mask[blockIdx.x * gridDim.y + blockIdx.y] = 0;
        return;
    }

    const int16_t* base;
    size_t step;
    if (transb == APE_TRANS_N) {
        base = src + blockIdx.x * AUTO_BLOCK * k +
                    blockIdx.y * AUTO_CHUNK + threadIdx.y;
        step = k;
    } else {
        base = src + blockIdx.x * AUTO_BLOCK +
                    blockIdx.y * AUTO_CHUNK * n + threadIdx.y * n;
        step = 1;
    }

    int8_t flag = 1;
    for (int i = 0; i < AUTO_BLOCK; i++) {
        int16_t tmp = base[i * step];
        if (tmp > INT16C_MAX) {
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

void create_mask_b_int16c(const int16_t *src, size_t k, size_t n, ApeTrans transb,
                                               int8_t *mask) {
    dim3 grid((n - 1) / AUTO_BLOCK + 1, (k - 1) / AUTO_CHUNK + 1, 1);
    dim3 block(1, AUTO_CHUNK, 1);

    kernel_create_mask_b_int16c<<<grid, block, ((AUTO_CHUNK - 1)/32+1) * sizeof (int8_t)>>>(src, k, n, transb, mask);

    cudaCheckError();
}

__global__ void kernel_count_overflow_a_int16c(const int16_t *src, size_t m, size_t k, ApeTrans transa,
                                               int8_t *block_count) {
    extern __shared__ char smem[];
    int8_t *shmem = (int8_t *)&smem[0];
    uint32_t warp_id = threadIdx.y / 32;
    uint32_t lane_id = threadIdx.y % 32;

    if (blockIdx.y * blockDim.y + threadIdx.y >= k) {
        return;
    }

    const int16_t* base;
    size_t step;
    if (transa == APE_TRANS_T) {    
        base = src + blockIdx.x * AUTO_BLOCK * k +
                    blockIdx.y * AUTO_CHUNK + threadIdx.y;
        step = k;
    } else {
        base = src + blockIdx.y * AUTO_CHUNK * m + blockIdx.x * AUTO_BLOCK + threadIdx.y * m;
        step = 1;
    }

    int8_t count = 0;
    for (int i = 0; i < AUTO_BLOCK && blockIdx.x * AUTO_BLOCK + i < m; i++) {
        int16_t tmp = base[i * step];
        if (tmp > INT16C_MAX) {
            count += 1;
        }
    }

    for (int i = 16; i > 0; i >>= 1) {
        int tmp = __shfl_down_sync(0xffffffff, count, i); // warp shuffle
        count += tmp;
    }

    if (lane_id == 0) {
        shmem[warp_id] = count;
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        count = 0;
        for (int i = 0; i < (blockDim.y-1) / 32 + 1; i++) {
            count += shmem[i];
        }
        block_count[blockIdx.x * gridDim.y + blockIdx.y] = count;
    }
}

void count_overflow_a_int16c(const int16_t *src, size_t m, size_t k, ApeTrans transa,
                                               int8_t *block_count) {
    dim3 grid((m - 1) / AUTO_BLOCK + 1, (k - 1) / AUTO_CHUNK + 1, 1);
    dim3 block(1, AUTO_CHUNK, 1);

    kernel_count_overflow_a_int16c<<<grid, block, ((AUTO_CHUNK - 1)/32+1) * sizeof (int8_t)>>>(src, m, k, transa, block_count);

    cudaCheckError();
}

__global__ void kernel_count_overflow_b_int16c(const int16_t *src, size_t k, size_t n, ApeTrans transb,
                                               int8_t *block_count) {
    extern __shared__ char smem[];
    int8_t *shmem = (int8_t *)&smem[0];
    uint32_t warp_id = threadIdx.y / 32;
    uint32_t lane_id = threadIdx.y % 32;

    if (blockIdx.y * blockDim.y + threadIdx.y >= k) {
        return;
    }

    const int16_t* base;
    size_t step;
    if (transb == APE_TRANS_N) {
        base = src + blockIdx.x * AUTO_BLOCK * k +
                    blockIdx.y * AUTO_CHUNK + threadIdx.y;
        step = k;
    } else {
        base = src + blockIdx.x * AUTO_BLOCK +
                    blockIdx.y * AUTO_CHUNK * n + threadIdx.y * n;
        step = 1;
    }

    int8_t count = 0;
    for (int i = 0; i < AUTO_BLOCK && blockIdx.x * AUTO_BLOCK + i < n; i++) {
        int16_t tmp = base[i * step];
        if (tmp > INT16C_MAX) {
            count += 1;
        }
    }

    for (int i = 16; i > 0; i >>= 1) {
        int tmp = __shfl_down_sync(0xffffffff, count, i); // warp shuffle
        count += tmp;
    }

    if (lane_id == 0) {
        shmem[warp_id] = count;
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        count = 0;
        for (int i = 0; i < (blockDim.y-1) / 32 + 1; i++) {
            count += shmem[i];
        }
        block_count[blockIdx.x * gridDim.y + blockIdx.y] = count;
    }
}

void count_overflow_b_int16c(const int16_t *src, size_t k, size_t n, ApeTrans transb,
                                               int8_t *block_count) {
    dim3 grid((n - 1) / AUTO_BLOCK + 1, (k - 1) / AUTO_CHUNK + 1, 1);
    dim3 block(1, AUTO_CHUNK, 1);

    kernel_count_overflow_b_int16c<<<grid, block, ((AUTO_CHUNK - 1)/32+1) * sizeof (int8_t)>>>(src, k, n, transb, block_count);

    cudaCheckError();
}
} // namespace gemm
