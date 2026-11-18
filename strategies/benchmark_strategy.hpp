#pragma once
#include "base_strategy.hpp"
#include "operators/operator_traits.hpp"
#include "core/gpu_props.hpp"
#include <vector>
#include <mutex>
#include <map>

class GpuMonitor; // Forward declaration

class BenchmarkStrategy : public BaseStrategy {
public:
    BenchmarkStrategy(GpuMonitor& monitor, const GpuProperties& props, const std::vector<std::string>& selected_precisions = {});
    void start() override;
    const std::vector<BenchmarkResult>& get_results() const;
    
    // Benchmark模式下这两个函数意义不大，可以返回固定值
    double get_current_performance() const override { return 0; }
    std::string get_active_operators_name() const override { return "Benchmarking..."; }
    int get_countdown() const override { return countdown_seconds_; }

private:
    void run_loop();

    // 运行时尺寸自动探测：对每个 (precision, sparsity) 在候选尺寸上短测，
    // 取实测吞吐最高者。结果缓存，避免每项重复探测。
    int autotune_dim(const OperatorDescriptor& desc);
    std::map<int, int> tuned_dim_cache_;   // key: precision*10 + sparsity

    
    GpuMonitor& monitor_;
    GpuProperties gpu_props_;
    std::vector<OperatorDescriptor> tests_to_run_;
    
    mutable std::mutex results_mutex_;
    std::vector<BenchmarkResult> results_;
    std::atomic<int> countdown_seconds_{0};
};
