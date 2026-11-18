#pragma once
#include "base_operator.hpp"
#include "core/gpu_props.hpp"
#include <memory>
#include <string>

namespace OperatorFactory {
    // dim == 0：按 MXBURNER_DIM 环境变量（upstream / tuned / 数字）决定，
    //           未设置时回落到本文件内置的实测标定值。
    // dim > 0  ：强制使用该 M=N=K（供运行时 autotune 使用）。
    std::unique_ptr<BaseOperator> create_gemm_operator(
        const OperatorDescriptor& descriptor,
        const GpuProperties& gpu_props,
        std::string& out_notes,
        int dim = 0
    );
}
