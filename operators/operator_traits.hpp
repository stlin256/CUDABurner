#pragma once
#include <string>
#include <vector>
#include <map>

enum class Precision { FP64, FP32, TF32, FP16, BF16, INT8, FP8, NF4 };
enum class Sparsity { DENSE, SPARSE };

const std::map<Precision, std::string> PRECISION_NAMES = {
    {Precision::FP64, "FP64"}, {Precision::FP32, "FP32"},
    {Precision::TF32, "TF32"}, {Precision::FP16, "FP16"},
    {Precision::BF16, "BF16"}, {Precision::INT8, "INT8"},
    {Precision::FP8,  "FP8"},  {Precision::NF4,  "NF4"}
};

struct OperatorDescriptor {
    Precision precision;
    Sparsity sparsity;
    // ... 可以添加其他维度，如算子类型 (GEMM, CONV)
};

struct BenchmarkResult {
    OperatorDescriptor descriptor;
    double performance = 0.0; // TFLOPS or TOPS
    bool is_native = false;
    bool is_supported = true;
    std::string unit = "TFLOPS";
    std::string notes; // <-- 新增字段，用于存放 "Tensor Core" 等备注
};
