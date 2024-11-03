// include/utils/matrix_utils.cuh
#pragma once
#include <cuda_runtime.h>

namespace matrix_utils {

// host functions
void initializeMatrix(float* matrix, int rows, int cols, bool randomize = true);
void verifyResults(const float* A, const float* B, const float* C,
    int M, int N, int K, float tolerance = 1e-5);

// memory management
cudaError_t allocateDeviceMatrix(float** matrix, int rows, int cols);
void freeDeviceMatrix(float* matrix);

} // namespace matrix_utils
