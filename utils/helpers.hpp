#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error("CUDA error");                         \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t status = call;                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                              \
            fprintf(stderr, "cuBLAS Error at %s:%d\n", __FILE__, __LINE__); \
            throw std::runtime_error("cuBLAS error");                       \
        }                                                                   \
    } while (0)

#define NVML_CHECK(call)                                                    \
    do {                                                                    \
        nvmlReturn_t result = call;                                         \
        if (result != NVML_SUCCESS) {                                       \
            fprintf(stderr, "NVML Error at %s:%d: %s\n", __FILE__, __LINE__, nvmlErrorString(result)); \
            throw std::runtime_error("NVML error");                         \
        }                                                                   \
    } while(0)
