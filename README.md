# APE on CUDA

This project is an APE implementation on NVIDIA GPU using the cuBLAS backend.

APE is a method of emulating high-bitwidth computation with low-bitwidth data types.
For example, APE can use $3$ or $6$ Tensor Core low-bitwidth computation to emulate an FP32 computation with up to $5.3\times$ theoretical speedup.
This project provides the following:

* GEMM implementations using Tensor Cores with FP32-precision and various representation ranges.
* Auto-adapted algorithm selection that guarantees end-to-end correctness.
* INT16 GEMM implementation using Tensor Cores.

For more details, please see our [paper](https://dl.acm.org/doi/abs/10.1145/3524059.3532377).

## Usage

### Build

```shell
mkdir build && cd build
cmake ..
make -j
```

### API

APE provides a blas-like API, and users only need to include ape.h to use APE to accelerate FP32 applications directly.

```c++
void apeGemmFP32(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc, const ApeAlgo algo = APE_ALGO_AUTO);
```

FP32 GEMM supports $5$ algorithms:

* **APE_ALGO_AUTO**: Select the fastest algorithm without overflow.

* **APE_ALGO_AUTO_STRICT**: Select the fastest algorithm without overflow and underflow.

* **APE_ALGO_FP32F**: Use FP16 emulated FP32. (1-bit precision loss, narrow representation range, overflow may occur.)

* **APE_ALGO_FP32B**: Use BF16 emulated FP32. (no precision loss, large representation range, overflow does not occur.)

* **APE_ALGO_FP32T**: Use TF32 emulated FP32. (1-bit precision loss, large representation range, overflow does not occur.)

```c++
void apeGemmINT16(ApeTrans transa, ApeTrans transb, int m, int n, int k, const int16_t *alpha, const int16_t *A, int lda, const int16_t *B, int ldb, const int32_t *beta, int32_t *C, int ldc, ApeAlgo algo = APE_ALGO_AUTO);
```

INT16 GEMM supports $2$ algorithms:

* **APE_ALGO_AUTO**: Select the algorithm without overflow.

* **APE_ALGO_INT16**: Use INT8 emulate INT16. (The upper bound is $32639$. Native INT16's is $32767$. Overflow may occur.)

## Authors
- [Zixuan Ma](https://github.com/JohndeVostok)
- [Yanzhuo Chen](https://github.com/yz-chen18)


## Citation

Ma, Zixuan, et al. "Efficiently emulating high-bitwidth computation with low-bitwidth hardware." Proceedings of the 36th ACM International Conference on Supercomputing. 2022.

If you find this work useful in your research, please cite it using the following BibTeX:

```bibtex
@inproceedings{ma2022efficiently,
author = {Ma, Zixuan and Wang, Haojie and Feng, Guanyu and Zhang, Chen and Xie, Lei and He, Jiaao and Chen, Shengqi and Zhai, Jidong},
title = {Efficiently Emulating High-Bitwidth Computation with Low-Bitwidth Hardware},
year = {2022},
isbn = {9781450392815},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3524059.3532377},
doi = {10.1145/3524059.3532377},
booktitle = {Proceedings of the 36th ACM International Conference on Supercomputing},
articleno = {5},
numpages = {12},
keywords = {emulation, tensor core, domain specific accelerator},
location = {Virtual Event},
series = {ICS '22}
}
```
