// include/kernels/matmul_tiled.cuh
#pragma once
#include <cuda_runtime.h>

namespace matmul {

constexpr int TILE_SIZE = 16;

__global__ void tiledMatrixMultiply(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
);

} //namespace matmul
