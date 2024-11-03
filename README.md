cuda-matmul/
├── CMakeLists.txt
├── include/
│ ├── kernels/
│ │ ├── matmul_naive.cuh
│ │ └── matmul_tiled.cuh
│ └── utils/
│ ├── cuda_utils.cuh
│ └── matrix_utils.cuh
├── src/
│ ├── kernels/
│ │ ├── matmul_naive.cu
│ │ └── matmul_tiled.cu
│ ├── utils/
│ │ ├── cuda_utils.cu
│ │ └── matrix_utils.cu
│ └── main.cu
└── README.md
