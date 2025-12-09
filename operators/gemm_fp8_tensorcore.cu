#include "operators/gemm_fp8_tensorcore.hpp"
#include "utils/helpers.hpp"

#if CUDA_VERSION >= 11080
#include <cuda_fp8.h>

// Helper to check cuBLAS LT errors
#define CUBLASLT_CHECK(call)                                                \
    do {                                                                    \
        cublasStatus_t status = call;                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                              \
            std::string errMsg = "cuBLASLt Error in " + std::string(__FILE__) + \
                                 ":" + std::to_string(__LINE__) + " - " +    \
                                 std::to_string(status);                     \
            throw std::runtime_error(errMsg);                               \
        }                                                                   \
    } while (0)

GemmFp8TensorCore::GemmFp8TensorCore(const OperatorDescriptor& desc, int m, int n, int k)
    : descriptor_(desc), m_(m), n_(n), k_(k) {
    
    CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle_));

    cudaDataType_t data_type = CUDA_R_8F_E4M3; // E4M3 is the standard for GEMM
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    size_t element_size = sizeof(__nv_fp8_e4m3);

    CUDA_CHECK(cudaMalloc(&d_A_, (size_t)m_ * k_ * element_size));
    CUDA_CHECK(cudaMalloc(&d_B_, (size_t)k_ * n_ * element_size));
    CUDA_CHECK(cudaMalloc(&d_C_, (size_t)m_ * n_ * element_size));

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmul_desc_, compute_type, CUDA_R_32F));
    if(descriptor_.sparsity == Sparsity::SPARSE){
        cublasOperation_t op_trans = CUBLAS_OP_T;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &op_trans, sizeof(op_trans)));
    }

    // For dense FP8, A is m x k, so leading dimension is m
    // For sparse FP8, A is transposed, so it's k x m, leading dimension is k
    size_t lda = (descriptor_.sparsity == Sparsity::SPARSE) ? k : m;

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&A_desc_, data_type, (descriptor_.sparsity == Sparsity::SPARSE) ? k : m, (descriptor_.sparsity == Sparsity::SPARSE) ? m : k, lda));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&B_desc_, data_type, k, n, k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&C_desc_, data_type, m, n, m));

    gflops_or_gops_ = (2.0 * m_ * n_ * k_) / 1e9;
    if(descriptor_.sparsity == Sparsity::SPARSE) gflops_or_gops_ *= 2.0;
}

GemmFp8TensorCore::~GemmFp8TensorCore() {
    cudaFree(d_A_);
    cudaFree(d_B_);
    cudaFree(d_C_);
    if (A_desc_) cublasLtMatrixLayoutDestroy(A_desc_);
    if (B_desc_) cublasLtMatrixLayoutDestroy(B_desc_);
    if (C_desc_) cublasLtMatrixLayoutDestroy(C_desc_);
    if (matmul_desc_) cublasLtMatmulDescDestroy(matmul_desc_);
    if (cublaslt_handle_) cublasLtDestroy(cublaslt_handle_);
}

void GemmFp8TensorCore::execute(cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLASLT_CHECK(cublasLtMatmul(cublaslt_handle_, matmul_desc_, &alpha, d_A_, A_desc_, d_B_, B_desc_, &beta, d_C_, C_desc_, d_C_, C_desc_, NULL, NULL, 0, stream));
}

double GemmFp8TensorCore::get_gflops_or_gops() const {
    return gflops_or_gops_;
}

#endif // CUDA_VERSION >= 11080
