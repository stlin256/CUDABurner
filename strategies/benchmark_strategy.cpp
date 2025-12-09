#include "strategies/benchmark_strategy.hpp"
#include "operators/operator_factory.hpp"
#include "utils/helpers.hpp"
#include <chrono>

BenchmarkStrategy::BenchmarkStrategy(const GpuProperties& props) : gpu_props_(props) {
    // Defines the full list of compute capabilities we want to test.
    tests_to_run_ = {
        {Precision::FP64, Sparsity::DENSE},
        {Precision::FP32, Sparsity::DENSE}, // This will now correctly test pure CUDA cores.
        {Precision::TF32, Sparsity::DENSE}, // This will test Tensor Cores.
        {Precision::FP16, Sparsity::DENSE},
        {Precision::FP16, Sparsity::SPARSE}, // <-- 新增稀疏 FP16 测试
        {Precision::BF16, Sparsity::DENSE},
        {Precision::INT8, Sparsity::DENSE},
        {Precision::INT8, Sparsity::SPARSE}, // <-- 新增稀疏 INT8 测试
        {Precision::FP8,  Sparsity::DENSE},
        {Precision::NF4,  Sparsity::DENSE},
    };
}

const std::vector<BenchmarkResult>& BenchmarkStrategy::get_results() const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    return results_;
}

void BenchmarkStrategy::start() {
    stop_flag_ = false;
    worker_thread_ = std::thread(&BenchmarkStrategy::run_loop, this);
}

void BenchmarkStrategy::run_loop() {
    CUDA_CHECK(cudaSetDevice(gpu_props_.device_id));

    const double test_duration_seconds = 10.0; // Target duration for each test

    for (const auto& test_desc : tests_to_run_) {
        if (stop_flag_) break;

        BenchmarkResult current_result;
        current_result.descriptor = test_desc;
        auto op = OperatorFactory::create_gemm_operator(test_desc, gpu_props_, current_result.notes);

        if (op == nullptr) {
            current_result.is_supported = false;
        } else {
            current_result.is_supported = true;
            // --- New Time-Based Benchmarking Logic ---

            // 1. Warm-up phase
            for (int i = 0; i < 10; ++i) {
                op->execute(0);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // 2. Timed execution phase
            long long iterations = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            double elapsed_seconds = 0;

            // This loop will run for at least `test_duration_seconds`.
            while (elapsed_seconds < test_duration_seconds) {
                op->execute(0); // Launch one operation
                iterations++;
                auto end_time = std::chrono::high_resolution_clock::now();
                elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
            }
            
            // 3. Final synchronization and performance calculation
            CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all launched ops to finish
            auto final_end_time = std::chrono::high_resolution_clock::now();
            double precise_duration = std::chrono::duration<double>(final_end_time - start_time).count();

            double total_gops = op->get_gflops_or_gops() * iterations;
            current_result.performance = total_gops / precise_duration / 1000.0; // TFLOPS/TOPS
            current_result.is_native = op->is_native();
            current_result.unit = (test_desc.precision == Precision::INT8) ? "TOPS" : "TFLOPS";
        }
        
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            results_.push_back(current_result);
        }
    }
    done_flag_ = true;
}
