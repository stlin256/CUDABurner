#include "strategies/benchmark_strategy.hpp"
#include "monitoring/monitor.hpp" // Include GpuMonitor header
#include "operators/operator_factory.hpp"
#include "utils/helpers.hpp"
#include <chrono>
#include <algorithm>
#include <map>

BenchmarkStrategy::BenchmarkStrategy(GpuMonitor& monitor, const GpuProperties& props, const std::vector<std::string>& selected_precisions)
    : monitor_(monitor), gpu_props_(props) {
    
    std::vector<OperatorDescriptor> all_tests = {
        {Precision::FP64, Sparsity::DENSE},
        {Precision::FP32, Sparsity::DENSE},
        {Precision::TF32, Sparsity::DENSE},
        {Precision::FP16, Sparsity::DENSE},
        {Precision::FP16, Sparsity::SPARSE},
        {Precision::BF16, Sparsity::DENSE},
        {Precision::INT8, Sparsity::DENSE},
        {Precision::INT8, Sparsity::SPARSE},
        // {Precision::FP8,  Sparsity::DENSE},
        // {Precision::FP8,  Sparsity::SPARSE},
        // {Precision::FP4,  Sparsity::DENSE},
    };

    if (selected_precisions.empty()) {
        tests_to_run_ = all_tests;
    } else {
        // Create a reverse map from string to Precision enum
        std::map<std::string, Precision> name_to_precision;
        for(const auto& pair : PRECISION_NAMES) {
            name_to_precision[pair.second] = pair.first;
        }

        for (const auto& selected : selected_precisions) {
            if (name_to_precision.count(selected)) {
                Precision p = name_to_precision.at(selected);
                // Find all tests (dense and sparse) matching this precision
                for(const auto& test : all_tests){
                    if(test.precision == p){
                        tests_to_run_.push_back(test);
                    }
                }
            } else {
                std::cerr << "Warning: Unknown precision '" << selected << "' ignored." << std::endl;
            }
        }
    }
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

    const double test_duration_seconds = 10.0;

    for (const auto& test_desc : tests_to_run_) {
        if (stop_flag_) break;

        BenchmarkResult current_result;
        current_result.descriptor = test_desc;
        auto op = OperatorFactory::create_gemm_operator(test_desc, gpu_props_, current_result.notes);

        if (op == nullptr) {
            current_result.is_supported = false;
        } else {
            current_result.is_supported = true;
            for (int i = 0; i < 10; ++i) {
                op->execute(0);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            long long iterations = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            double elapsed_seconds = 0;

            while (elapsed_seconds < test_duration_seconds) {
                op->execute(0);
                iterations++;
                auto end_time = std::chrono::high_resolution_clock::now();
                elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
                
                // Check for throttling during the test
                if (monitor_.get_state().throttled) {
                    current_result.was_throttled = true;
                }
            }
            
            CUDA_CHECK(cudaDeviceSynchronize());
            auto final_end_time = std::chrono::high_resolution_clock::now();
            double precise_duration = std::chrono::duration<double>(final_end_time - start_time).count();

            double total_gops = op->get_gflops_or_gops() * iterations;
            current_result.performance = total_gops / precise_duration / 1000.0;
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
