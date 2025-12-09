#include "operators/operator_factory.hpp"
#include "operators/gemm_generic.hpp"
#include "operators/gemm_tensorcore.hpp"
#include "operators/gemm_sparse_tensorcore.hpp" // <-- 新增 include

namespace OperatorFactory {

std::unique_ptr<BaseOperator> create_gemm_operator(
    const OperatorDescriptor& desc, const GpuProperties& props, std::string& out_notes) {
    
    int m = 4096, n = 4096, k = 4096;
    if(desc.precision == Precision::FP64) { m=2048; n=2048; k=2048; }
    if(desc.precision == Precision::INT8) { m=8192; n=8192; k=8192; }

    // 在 switch (desc.precision) 之前添加对稀疏的判断
    if (desc.sparsity == Sparsity::SPARSE) {
        // Ampere+ GPUs support structured sparsity.
        if (props.cc_major >= 8) {
            if (desc.precision == Precision::FP16 || desc.precision == Precision::INT8) {
                out_notes = "Sparse Tensor Core";
                return std::make_unique<GemmSparseTensorCore>(desc, m, n, k);
            }
        }
    }

    // 原有的 switch 语句保持不变
    switch (desc.precision) {
        case Precision::FP64:
            if (props.cc_major >= 6) { 
                out_notes = "CUDA Core (Native)";
                return std::make_unique<GemmGeneric<double>>(desc, m, n, k, true); // Force no TC for doubles
            }
            break;

        case Precision::FP32:
            // This now correctly creates an operator that *only* uses CUDA Cores.
            out_notes = "CUDA Core (TC Disabled)";
            return std::make_unique<GemmGeneric<float>>(desc, m, n, k, true); // Pass 'true' to force TC off

        case Precision::TF32:
            if (props.cc_major >= 8) { // Ampere+
                out_notes = "Tensor Core (Default FP32)";
                // This operator will internally enable Tensor Cores for TF32.
                return std::make_unique<GemmTensorCore>(desc, props, m, n, k);
            }
            break;
        // ... (FP16, BF16, INT8, etc., cases remain the same as they correctly use GemmTensorCore) ...
        case Precision::FP16:
            if (props.cc_major >= 7) { out_notes = "Tensor Core"; return std::make_unique<GemmTensorCore>(desc, props, m, n, k); }
            break;
        case Precision::BF16:
            if (props.cc_major >= 8) { out_notes = "Tensor Core"; return std::make_unique<GemmTensorCore>(desc, props, m, n, k); }
            break;
        case Precision::INT8:
            if (props.cc_major >= 7) { out_notes = "Tensor Core"; return std::make_unique<GemmTensorCore>(desc, props, m, n, k); }
            break;
        case Precision::FP8:
             if (props.cc_major >= 9) { out_notes = "Tensor Core"; }
             break;
        case Precision::NF4:
             out_notes = "Emulation Required";
             break;
    }

    if(out_notes.empty()) out_notes = "Not Supported";
    return nullptr;
}

} // namespace OperatorFactory
