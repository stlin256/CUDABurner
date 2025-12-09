#pragma once
#include "operators/base_operator.hpp"
#include "core/gpu_props.hpp"
#include <cublasLt.h>

// Use CUDA_VERSION for a reliable compile-time check
#if CUDA_VERSION >= 11080
class GemmFp8TensorCore : public BaseOperator {
public:
    GemmFp8TensorCore(const OperatorDescriptor& desc, int m, int n, int k);
    ~GemmFp8TensorCore() override;

    void execute(cudaStream_t stream) override;
    double get_gflops_or_gops() const override;
    const OperatorDescriptor& get_descriptor() const override { return descriptor_; }
    bool is_native() const override { return true; }

private:
    int m_, n_, k_;
    void *d_A_{nullptr}, *d_B_{nullptr}, *d_C_{nullptr};
    cublasLtHandle_t cublaslt_handle_;
    cublasLtMatmulDesc_t matmul_desc_;
    cublasLtMatrixLayout_t A_desc_{nullptr}, B_desc_{nullptr}, C_desc_{nullptr};
    
    OperatorDescriptor descriptor_;
    double gflops_or_gops_;
};
#endif // CUDA_VERSION >= 11080
