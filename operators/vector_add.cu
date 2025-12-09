#include "operators/vector_add.hpp"
#include "utils/helpers.hpp"

__global__ void vectorAddKernel(const float* a, const float* b, float* c, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

VectorAddOperator::VectorAddOperator(size_t num_elements) : num_elements_(num_elements) {
    descriptor_ = {Precision::FP32, Sparsity::DENSE}; // Placeholder descriptor
    CUDA_CHECK(cudaMalloc(&d_A_, num_elements_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_, num_elements_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_, num_elements_ * sizeof(float)));
}

VectorAddOperator::~VectorAddOperator() {
    cudaFree(d_A_);
    cudaFree(d_B_);
    cudaFree(d_C_);
}

void VectorAddOperator::execute(cudaStream_t stream) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements_ + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A_, d_B_, d_C_, num_elements_);
}

// For memory-bound ops, FLOPS are not a good metric. We can return 0.
double VectorAddOperator::get_gflops_or_gops() const {
    return 0; 
}
