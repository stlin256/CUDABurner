#include "gemm_tensorcore.hpp"
#include "utils/helpers.hpp"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>

// GemmTensorCore 类的构造函数
GemmTensorCore::GemmTensorCore(const OperatorDescriptor& desc, const GpuProperties& props, int m, int n, int k) 
    : m_(m), n_(n), k_(k), descriptor_(desc), is_native_(true) {

    size_t a_size = 0, b_size = 0, c_size = 0;
    
    // 设置 cuBLAS 计算类型
    if (descriptor_.precision == Precision::TF32 && props.cc_major >= 8) {
        compute_type_ = CUBLAS_COMPUTE_32F_FAST_TF32;
        data_type_ = CUDA_R_32F;
        a_size = b_size = c_size = sizeof(float);
    } else if (descriptor_.precision == Precision::FP16) {
        compute_type_ = CUBLAS_COMPUTE_32F; // often compute in FP32 for stability
        data_type_ = CUDA_R_16F;
        a_size = b_size = c_size = sizeof(__half);
    } else if (descriptor_.precision == Precision::BF16) {
        compute_type_ = CUBLAS_COMPUTE_32F;
        data_type_ = CUDA_R_16BF;
        a_size = b_size = c_size = sizeof(__nv_bfloat16);
    } else if (descriptor_.precision == Precision::INT8 && props.cc_major >= 7) {
        compute_type_ = CUBLAS_COMPUTE_32I;
        data_type_ = CUDA_R_8I;
        a_size = b_size = sizeof(int8_t);
        c_size = sizeof(int32_t); // Output for INT8 is often INT32
    } else {
        is_native_ = false; // Should not happen if factory is correct
        throw std::runtime_error("Unsupported precision for GemmTensorCore");
    }
    
    CUDA_CHECK(cudaMalloc(&d_A_, m_ * k_ * a_size));
    CUDA_CHECK(cudaMalloc(&d_B_, k_ * n_ * b_size));
    CUDA_CHECK(cudaMalloc(&d_C_, m_ * n_ * c_size));

    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    
    if (descriptor_.precision == Precision::TF32) {
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_TF32_TENSOR_OP_MATH));
    } else {
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH));
    }
}

GemmTensorCore::~GemmTensorCore() {
    cudaFree(d_A_);
    cudaFree(d_B_);
    cudaFree(d_C_);
    cublasDestroy(cublas_handle_);
}

void GemmTensorCore::execute(cudaStream_t stream) {
    float alpha = 1.0f, beta = 0.0f;
    int32_t alpha_i = 1, beta_i = 0;
    void *alpha_ptr = &alpha, *beta_ptr = &beta;
    
    if(descriptor_.precision == Precision::INT8) {
        alpha_ptr = &alpha_i;
        beta_ptr = &beta_i;
    }

    cublasSetStream(cublas_handle_, stream);
    
    CUBLAS_CHECK(cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                              m_, n_, k_, alpha_ptr,
                              d_A_, data_type_, m_,
                              d_B_, data_type_, k_, beta_ptr,
                              d_C_, (descriptor_.precision == Precision::INT8) ? CUDA_R_32I : data_type_, m_,
                              compute_type_, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

double GemmTensorCore::get_gflops_or_gops() const {
    // For GEMM, operations = 2 * M * N * K
    return 2.0 * m_ * n_ * k_ / 1e9;
}

const OperatorDescriptor& GemmTensorCore::get_descriptor() const {
    return descriptor_;
}

bool GemmTensorCore::is_native() const {
    return is_native_;
}
