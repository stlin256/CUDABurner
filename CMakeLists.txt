cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(CUDABurner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

find_package(Threads REQUIRED)
find_library(NVML_LIB nvidia-ml DOC "Path to NVML library")
if(NOT NVML_LIB)
    message(FATAL_ERROR "NVML library (nvidia-ml) not found.")
endif()

add_executable(CUDABurner
    main.cu # <-- KEY CHANGE
    monitoring/monitor.cpp
    operators/gemm_generic.cu
    operators/gemm_tensorcore.cu
    operators/vector_add.cu
    operators/power_virus_kernel.cu
    operators/gemm_sparse_tensorcore.cu
    operators/operator_factory.cpp
    strategies/stress_strategy.cpp
    strategies/benchmark_strategy.cpp
    ui/tui.cpp
)

target_include_directories(CUDABurner PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

# --- 终极釜底抽薪之计 ---
# 我们把所有库都链接到最终的可执行文件上
# 并且把 cudart 放在链接列表的最后，这有时能解决顺序问题
target_link_libraries(CUDABurner PRIVATE
    Threads::Threads
    ${NVML_LIB}
    cublas
    cublasLt
    cudart
)

target_compile_options(CUDABurner PRIVATE -Wall -Wextra -O3)
install(TARGETS CUDABurner DESTINATION bin)
