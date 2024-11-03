cmake_minimum_required(VERSION 3.8)
project(cruda CUDA CXX)

# set CUDA architecture for your GPU
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCES
  "src/*.cu"
  "src/kernels/*.cu"
  "src/utils/*.cu"
)

add_executable(matmul ${SOURCES})
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


