// src/kernels/matmul_tiled.cu
#include "kernels/matmul_naive.cuh"

namespace matmul {

__global__ void TiledMatrixMultiply(
  const float* A,
  const float* B,
  float* C,
  int M, int N, int K
) {
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  float sum = 0.0f;

  for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
    // load tiles collaboratively
    if (row < M && tile * TILE_SIZE + tx < K)
      tileA[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
    else
      tileA[ty][tx] = 0.0f;

    if (col < N && tile * TILE_SIZE + ty < K)
      tileB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
    else
      tileB[ty][tx] = 0.0f;

    __syncthreads();

    // compute partial sum
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += tileA[ty][k] * tileB[k][tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

} // namespace matmul
