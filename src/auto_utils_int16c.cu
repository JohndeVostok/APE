#include "thrust/device_vector.h"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include "cuda_utils.h"
#include "kernel.h"

namespace ape {

struct OverflowOpInt16
{
    __host__ __device__
        int operator()(const int16_t& x) const { 
            if (x > INT16C_MAX) {
                return 1;
            } else {
                return 0;
            }
        }
};

int count_overflow_int16c(const int16_t *src, size_t row, size_t col) {
    OverflowOpInt16   unary_op;
    thrust::plus<int> binary_op;
    int init = 0;

    thrust::device_ptr<int16_t> d_src(const_cast<int16_t*>(src));
    int count = thrust::transform_reduce(d_src, d_src+row*col, unary_op, init, binary_op);
    
    return count;
}
} // namespace gemm
