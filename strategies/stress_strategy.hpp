#pragma once
#include "strategies/base_strategy.hpp"
#include "core/gpu_props.hpp"
#include "operators/base_operator.hpp"
#include <vector>
#include <memory>
#include <string>
#include <thread> // For std::thread

class StressStrategy : public BaseStrategy {
public:
    StressStrategy(const GpuProperties& props);
    ~StressStrategy() override;

    void start() override;
    void stop() override; // Need a custom stop to join threads
    std::string get_active_operators_name() const;
    double get_current_performance() const override { return 0; }

private:
    // This function will be executed by each worker thread.
    void worker_thread_loop(cudaStream_t stream);

    GpuProperties gpu_props_;
    std::unique_ptr<BaseOperator> gemm_op_;
    
    // For multi-threading
    std::vector<std::thread> worker_threads_;
    std::vector<cudaStream_t> streams_;
};
