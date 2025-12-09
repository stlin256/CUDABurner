#pragma once
#include "operators/base_operator.hpp"
#include <cublas_v2.h>

template <typename T>
class GemmGeneric : public BaseOperator {
public:
    // Added a flag to explicitly control Tensor Core usage
    GemmGeneric(const OperatorDescriptor& desc, int m, int n, int k, bool force_no_tensor_cores = false);
    ~GemmGeneric() override;

    void execute(cudaStream_t stream) override;
    double get_gflops_or_gops() const override;
    const OperatorDescriptor& get_descriptor() const override { return descriptor_; }
    bool is_native() const override { return true; }

private:
    int m_, n_, k_;
    OperatorDescriptor descriptor_;
    double gflops_;
    T *d_A_, *d_B_, *d_C_;
    cublasHandle_t cublas_handle_;
};
