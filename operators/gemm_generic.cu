#include "operators/gemm_generic.hpp"
#include "utils/helpers.hpp"

// Forward declarations of cuBLAS functions for template specialization
template <typename T> cublasStatus_t cublasGemm(cublasHandle_t handle,
                                                cublasOperation_t transa, cublasOperation_t transb,
                                                int m, int n, int k,
                                                const T *alpha, const T *A, int lda,
                                                const T *B, int ldb, const T *beta,
                                                T *C, int ldc);

template <> inline cublasStatus_t cublasGemm<double>(cublasHandle_t handle,
                                            cublasOperation_t transa, cublasOperation_t transb,
                                            int m, int n, int k,
                                            const double *alpha, const double *A, int lda,
                                            const double *B, int ldb, const double *beta,
                                            double *C, int ldc) {
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <> inline cublasStatus_t cublasGemm<float>(cublasHandle_t handle,
                                           cublasOperation_t transa, cublasOperation_t transb,
                                           int m, int n, int k,
                                           const float *alpha, const float *A, int lda,
                                           const float *B, int ldb, const float *beta,
                                           float *C, int ldc) {
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


template <typename T>
GemmGeneric<T>::GemmGeneric(const OperatorDescriptor& desc, int m, int n, int k, bool force_no_tensor_cores) 
    : m_(m), n_(n), k_(k), descriptor_(desc) { 
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));

    // *** KEY CHANGE: Explicitly control Tensor Core usage ***
    if (force_no_tensor_cores) {
        // This forces cuBLAS to use only standard CUDA cores for FP32 GEMM.
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH));
    } else {
        // Allow cuBLAS to use Tensor Cores if available (default behavior).
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_TENSOR_OP_MATH));
    }

    CUDA_CHECK(cudaMalloc(&d_A_, (size_t)m_ * k_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_B_, (size_t)k_ * n_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_C_, (size_t)m_ * n_ * sizeof(T)));
    gflops_ = (2.0 * m_ * n_ * k_) / 1e9;
}

template <typename T>
GemmGeneric<T>::~GemmGeneric() {
    cudaFree(d_A_);
    cudaFree(d_B_);
    cudaFree(d_C_);
    cublasDestroy(cublas_handle_);
}

template <typename T>
void GemmGeneric<T>::execute(cudaStream_t stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream));
    const T alpha = 1.0;
    const T beta = 0.0;
    CUBLAS_CHECK(cublasGemm<T>(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                               m_, n_, k_, &alpha,
                               d_A_, m_,
                               d_B_, k_, &beta,
                               d_C_, m_));
}

template <typename T>
double GemmGeneric<T>::get_gflops_or_gops() const {
    return gflops_;
}

// Explicit template instantiations
template class GemmGeneric<double>;
template class GemmGeneric<float>;
