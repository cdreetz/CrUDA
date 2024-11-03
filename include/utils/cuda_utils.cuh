// include/utils/cuda_utils.cuh
#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

namespace cuda_utils {

// Error checking macro
#define CUDA_CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

// Timer class for performance measurement
class CudaTimer {
public:
    CudaTimer();
    ~CudaTimer();
    void start();
    float stop(); // Returns elapsed time in ms

private:
    cudaEvent_t startEvent, stopEvent;
};

} // namespace cuda_utils
