#include "operators/operator_factory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>


#include "operators/gemm_generic.hpp"
#include "operators/gemm_tensorcore.hpp"
#include "operators/gemm_sparse_tensorcore.hpp"
#include "operators/gemm_fp8_tensorcore.hpp"
#include "operators/gemm_fp4_tensorcore.hpp"

namespace {

// --- 矩阵尺寸选择 ---------------------------------------------------------
// 历史固定值：FP64=2048, INT8=8192, 其余=4096。
// 这是**未做探测时的保守默认值**，不是最优值：实测表明吞吐随边长变化很大
// （实测：同一份 FP16 代码在 MetaX C550 上，边长 2048 只有 175 TFLOPS，16384 是 278）。
// benchmark 模式会先做运行时探测（见 BenchmarkStrategy::autotune_dim）再决定实际边长，
// 所以真正生效的值通常不是这里。
int default_dim(Precision p) {
    if (p == Precision::FP64) return 2048;
    if (p == Precision::INT8) return 8192;
    return 4096;
}

int resolve_dim(Precision p) {
    const char* env = std::getenv("CUDABURNER_DIM");
    if (env == nullptr) return default_dim(p);
    if (std::strcmp(env, "default") == 0 || std::strcmp(env, "legacy") == 0) return default_dim(p);
    int d = std::atoi(env);
    return d > 0 ? d : default_dim(p);
}

}  // namespace

namespace OperatorFactory {

std::unique_ptr<BaseOperator> create_gemm_operator(
    const OperatorDescriptor& desc, const GpuProperties& props, std::string& out_notes, int dim_override) {

    const int dim = dim_override > 0 ? dim_override : resolve_dim(desc.precision);
    int m = dim, n = dim, k = dim;

    if (desc.sparsity == Sparsity::SPARSE) {
        if (props.cc_major >= 9) {
            if (desc.precision == Precision::FP8) {
                out_notes = "Sparse Tensor Core";
                #if CUDA_VERSION >= 11080
                    return std::make_unique<GemmFp8TensorCore>(desc, m, n, k);
                #else
                    out_notes = "FP8 requires CUDA Toolkit 11.8+";
                    return nullptr;
                #endif
            }
        }
        if (props.cc_major >= 8) {
            if (desc.precision == Precision::FP16 || desc.precision == Precision::INT8) {
                out_notes = "Sparse Tensor Core";
                return std::make_unique<GemmSparseTensorCore>(desc, m, n, k);
            }
        }
    }

    switch (desc.precision) {
        case Precision::FP64:
            if (props.cc_major >= 6) { 
                out_notes = "CUDA Core (Native)";
                return std::make_unique<GemmGeneric<double>>(desc, m, n, k, true);
            }
            break;

        case Precision::FP32:
            out_notes = "CUDA Core (TC Disabled)";
            return std::make_unique<GemmGeneric<float>>(desc, m, n, k, true);

        case Precision::TF32:
            if (props.cc_major >= 8) {
                out_notes = "Tensor Core (Default FP32)";
                return std::make_unique<GemmTensorCore>(desc, props, m, n, k);
            }
            break;
        
        case Precision::FP16:
        case Precision::BF16:
        case Precision::INT8:
            if (props.cc_major >= 7) {
                 out_notes = "Tensor Core";
                 return std::make_unique<GemmTensorCore>(desc, props, m, n, k);
            }
            break;
            
        case Precision::FP8:
            {
                // Official FP8 support is on Hopper (sm_9x) and Ada (sm_89).
                bool is_supported = (props.cc_major == 9) || (props.cc_major == 8 && props.cc_minor == 9);
                #if CUDA_VERSION < 11080
                    is_supported = false;
                #endif

                if (is_supported) {
                    out_notes = "Tensor Core";
                    return std::make_unique<GemmFp8TensorCore>(desc, m, n, k);
                } else {
                    out_notes = "FP32 Simulation (Unsupported)";
                    return std::make_unique<GemmGeneric<float>>(desc, m, n, k, true);
                }
            }
            
        case Precision::FP4:
            {
                // Assume FP4 has the same support as FP8 for now.
                bool is_supported = (props.cc_major == 9) || (props.cc_major == 8 && props.cc_minor == 9);
                #if CUDA_VERSION < 12030
                    is_supported = false;
                #endif

                if (is_supported) {
                    out_notes = "Tensor Core";
                    return std::make_unique<GemmFp4TensorCore>(desc, m, n, k);
                } else {
                    out_notes = "FP32 Simulation (Unsupported)";
                    return std::make_unique<GemmGeneric<float>>(desc, m, n, k, true);
                }
            }
    }

    if(out_notes.empty()) out_notes = "Not Supported";
    return nullptr;
}

}
