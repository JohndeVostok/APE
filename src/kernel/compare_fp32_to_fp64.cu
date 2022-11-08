#include "common.h"
#include "kernel.h"

namespace ape {

__global__ void kernel_calc_error(double *buf_sum, double *buf_max, const float *src, const double *dst, size_t size) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    __shared__ double sbuf_sum[32];
    __shared__ double sbuf_max[32];

    uint32_t base = (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t step = blockDim.x * gridDim.x;
    double sum = 0, max = 0;
    for (uint32_t i = base; i < size; i += step) {
        double err = fabs(double(src[i]) - dst[i]) / fabs(dst[i]);
        sum += err;
        max = fmax(max, err);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        max = fmax(max, __shfl_down_sync(0xffffffff, max, offset));
    }

    sbuf_sum[warp_id] = sum;
    sbuf_max[warp_id] = max;
    __syncthreads();
    if (warp_id == 0) {
        sum = sbuf_sum[lane_id];
        max = sbuf_max[lane_id];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
            max = fmax(max, __shfl_down_sync(0xffffffff, max, offset));
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        buf_sum[blockIdx.x] = sum;
        buf_max[blockIdx.x] = max;
    }
}

void compare_fp32_to_fp64(const float *src, const double *dst, size_t size, double &max_error, double &mean_error) {
    double *buf_sum, *buf_max;
    cudaSafeCall(cudaMalloc((void **)&buf_sum, 108 * sizeof(double)));
    cudaSafeCall(cudaMalloc((void **)&buf_max, 108 * sizeof(double)));

    dim3 grid_size(108, 1);
    dim3 block_size(1024, 1);
    kernel_calc_error<<<grid_size, block_size>>>(buf_sum, buf_max, src, dst, size);
    cudaCheckError();

    double *buf_sum_host = (double *)malloc(108 * sizeof(double));
    double *buf_max_host = (double *)malloc(108 * sizeof(double));
    cudaSafeCall(cudaMemcpy(buf_sum_host, buf_sum, 108 * sizeof(double), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(buf_max_host, buf_max, 108 * sizeof(double), cudaMemcpyDeviceToHost));

    double sum = 0, max = 0;
    for (int i = 0; i < 108; i++) {
        sum += buf_sum_host[i];
        max = std::max(max, buf_max_host[i]);
    }
    max_error = max;
    mean_error = sum / size;
}

} // namespace ape