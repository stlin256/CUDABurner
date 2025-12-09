#pragma once
#include "base_strategy.hpp"
#include "operators/operator_traits.hpp"
#include "core/gpu_props.hpp"
#include <vector>
#include <mutex>

class BenchmarkStrategy : public BaseStrategy {
public:
    BenchmarkStrategy(const GpuProperties& props);
    void start() override;
    const std::vector<BenchmarkResult>& get_results() const;
    
    // Benchmark模式下这两个函数意义不大，可以返回固定值
    double get_current_performance() const override { return 0; }
    std::string get_active_operators_name() const override { return "Benchmarking..."; }

private:
    void run_loop();
    
    GpuProperties gpu_props_;
    std::vector<OperatorDescriptor> tests_to_run_;
    
    mutable std::mutex results_mutex_;
    std::vector<BenchmarkResult> results_;
};
