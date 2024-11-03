// include/kernels/matmul_naive.cuh
#pragma once
#include <cuda_runtime.h>

namespace matmul {

__global__ void naiveMatrixMultiply(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
);

} // namespace matmul
