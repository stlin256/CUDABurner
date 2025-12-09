#pragma once
#include "operators/base_operator.hpp"

class VectorAddOperator : public BaseOperator {
public:
    VectorAddOperator(size_t num_elements);
    ~VectorAddOperator() override;

    void execute(cudaStream_t stream) override;
    double get_gflops_or_gops() const override;
    const OperatorDescriptor& get_descriptor() const override { return descriptor_; }
    bool is_native() const override { return true; }

private:
    size_t num_elements_;
    float *d_A_, *d_B_, *d_C_;
    OperatorDescriptor descriptor_;
};
