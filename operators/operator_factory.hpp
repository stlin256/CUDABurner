#pragma once
#include "base_operator.hpp"
#include "core/gpu_props.hpp"
#include <memory>
#include <string>

namespace OperatorFactory {
    std::unique_ptr<BaseOperator> create_gemm_operator(
        const OperatorDescriptor& descriptor,
        const GpuProperties& gpu_props,
        std::string& out_notes // <-- 新增参数
    );
}
