#pragma once

#include "base_operator.hpp"
#include "core/gpu_props.hpp"
#include <cublas_v2.h>

class GemmTensorCore : public BaseOperator {
public:
    GemmTensorCore(const OperatorDescriptor& desc, const GpuProperties& props, int m, int n, int k);
    ~GemmTensorCore();

    void execute(cudaStream_t stream) override;
    double get_gflops_or_gops() const override;
    const OperatorDescriptor& get_descriptor() const override;
    bool is_native() const override;

private:
    int m_, n_, k_;
    OperatorDescriptor descriptor_;
    bool is_native_;

    void *d_A_ = nullptr, *d_B_ = nullptr, *d_C_ = nullptr;
    
    cublasHandle_t cublas_handle_;
    cudaDataType_t data_type_;
    cublasComputeType_t compute_type_;
};
