// src/main.cu
#include <stdio.h>
#include "utils/cuda_utils.cuh"
#include "utils/matrix_utils.cuh"
#include "kernels/matmul_naive.cuh"
#include "kernels/matmul_tiled.cuh"

using namespace cuda_utils;
using namespace matrix_utils;
using namespace matmul;

int main() {
  // matrix dims
  const int M = 1024; // A rows
  const int M = 1024; // A cols, B rows
  const int M = 1024; // B cols

  // host matrices
  float *h_A, *h_B, *h_C_naive, *h_C_tiled;
  h_A = new float[M * K];
  h_B = new float[K * N];
  h_C_naive = new float[M * N];
  h_C_tiled = new float[M * N];

  // initialize matrices
  initializeMatrix(h_A, M, K);
  initializeMatrix(h_B, K, N);

  // device matrices
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(allocateDeviceMatrix(&d_A, M, K));
  CUDA_CHECK(allocateDeviceMatrix(&d_B, K, N));
  CUDA_CHECK(allocateDeviceMatrix(&d_C, M, N));

  // copy input matrices to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice))
  CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice))

  CudaTimer timer;

  // launch naive kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                (M + blockSize.y - 1) / blockSize.y);

  timer.start();
  naiveMatrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
  float naive_time = timer.stop();

  // copy result back
  CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, M * N * sizeof(flaot), cudaMemcpyHostToDevice));

  timer.start();
  tiledMatrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
  float tiled_time = timer.stop();

  CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, M * N * sizeof(flaot), cudaMemcpyHostToDevice));

  // verify results
  bool results_match = verifyResults(h_C_naive, h_C_tiled, M, N, 1e-5);
  printf("Results match: %s\n", results_match ? "YES" : "NO");
  printf("Naive kernel time: %.3f ms\n", naive_time);
  printf("Tiled kernel time: %.3f ms\n", tiled_time);
  printf("Speedup: %.2fx\n", naive_time / tiled_time);

  // cleanup
  delete[] h_A;
  delete[] h_B;
  delete[] h_C_naive;
  delete[] h_C_tiled;
  freeDeviceMatrix(d_A);
  freeDeviceMatrix(d_B);
  freeDeviceMatrix(d_C);

  return 0;
}
