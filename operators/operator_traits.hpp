#pragma once
#include <string>
#include <vector>
#include <map>

// --- KEY CORRECTION: Define enums BEFORE they are used ---
enum class Precision { FP64, FP32, TF32, FP16, BF16, INT8, FP8, FP4 };
enum class Sparsity { DENSE, SPARSE };

// The struct that uses the enums
struct OperatorDescriptor {
    Precision precision;
    Sparsity sparsity;
};

// Now, the map that uses the enums
const std::map<Precision, std::string> PRECISION_NAMES = {
    {Precision::FP64, "FP64"}, {Precision::FP32, "FP32"},
    {Precision::TF32, "TF32"}, {Precision::FP16, "FP16"},
    {Precision::BF16, "BF16"}, {Precision::INT8, "INT8"},
    {Precision::FP8,  "FP8"},  {Precision::FP4,  "FP4"}
};

struct BenchmarkResult {
    OperatorDescriptor descriptor;
    double performance = 0.0; // TFLOPS or TOPS
    bool is_native = false;
    bool is_supported = true;
    std::string unit = "TFLOPS";
    std::string notes;
    bool was_throttled = false;
};
