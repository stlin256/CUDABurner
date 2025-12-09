#pragma once
#include "operator_traits.hpp"
#include <cuda_runtime.h>

class BaseOperator {
public:
    virtual ~BaseOperator() = default;
    virtual void execute(cudaStream_t stream) = 0;
    virtual double get_gflops_or_gops() const = 0;
    virtual const OperatorDescriptor& get_descriptor() const = 0;
    virtual bool is_native() const = 0;
};
