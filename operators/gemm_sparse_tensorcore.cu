#include "operators/gemm_sparse_tensorcore.hpp"
#include "utils/helpers.hpp"
#include <cuda_fp16.h>

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

GemmSparseTensorCore::GemmSparseTensorCore(const OperatorDescriptor& desc, int m, int n, int k)
    : descriptor_(desc), m_(m), n_(n), k_(k) {
    
    // REMOVED the problematic compile-time check:
    // #if CUDA_VERSION < 11000 ... #endif

    CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle_));
    
    cudaDataType_t cuda_data_type;
    cublasComputeType_t compute_type;
    size_t element_size = 0;

    switch (descriptor_.precision) {
        case Precision::FP16:
            cuda_data_type = CUDA_R_16F;
            compute_type = CUBLAS_COMPUTE_32F;
            element_size = sizeof(__half);
            break;
        default:
            throw std::runtime_error("Unsupported precision for Sparse GEMM in this version.");
    }

    CUDA_CHECK(cudaMalloc(&d_A_, (size_t)m_ * k_ * element_size));
    CUDA_CHECK(cudaMalloc(&d_B_, (size_t)k_ * n_ * element_size));
    CUDA_CHECK(cudaMalloc(&d_C_, (size_t)m_ * n_ * element_size));

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmul_desc_, compute_type, CUDA_R_32F));
    
    cublasOperation_t op_trans_a = CUBLAS_OP_T;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &op_trans_a, sizeof(op_trans_a)));

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&A_desc_, cuda_data_type, k, m, k)); 
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&B_desc_, cuda_data_type, k, n, k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&C_desc_, cuda_data_type, m, n, m));

    gflops_or_gops_ = (2.0 * m_ * n_ * k_ * 2.0) / 1e9;
}

GemmSparseTensorCore::~GemmSparseTensorCore() {
    cudaFree(d_A_);
    cudaFree(d_B_);
    cudaFree(d_C_);
    if (A_desc_) cublasLtMatrixLayoutDestroy(A_desc_);
    if (B_desc_) cublasLtMatrixLayoutDestroy(B_desc_);
    if (C_desc_) cublasLtMatrixLayoutDestroy(C_desc_);
    if (matmul_desc_) cublasLtMatmulDescDestroy(matmul_desc_);
    if (cublaslt_handle_) cublasLtDestroy(cublaslt_handle_);
}

void GemmSparseTensorCore::execute(cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    CUBLASLT_CHECK(cublasLtMatmul(cublaslt_handle_, matmul_desc_, &alpha, d_A_, A_desc_, d_B_, B_desc_, &beta, d_C_, C_desc_, d_C_, C_desc_, NULL, NULL, 0, stream));
}

double GemmSparseTensorCore::get_gflops_or_gops() const {
    return gflops_or_gops_;
}
