#include "strategies/stress_strategy.hpp"
#include "operators/operator_factory.hpp"
#include "utils/helpers.hpp"
#include <chrono> // For std::this_thread::sleep_for

StressStrategy::StressStrategy(const GpuProperties& props) : gpu_props_(props) {
    CUDA_CHECK(cudaSetDevice(gpu_props_.device_id));

    // --- 1. 准备一个可被所有线程共享的、最高效的 GEMM 算子 (FP16) ---
    std::string notes;
    OperatorDescriptor gemm_desc;
    
    // FP16 GEMM is the best candidate for raw power draw on modern GPUs.
    if (props.cc_major >= 7) {
        gemm_desc = {Precision::FP16, Sparsity::DENSE};
    } else {
        gemm_desc = {Precision::FP32, Sparsity::DENSE}; // Fallback
    }
    gemm_op_ = OperatorFactory::create_gemm_operator(gemm_desc, props, notes);
    if (!gemm_op_) { 
        throw std::runtime_error("Could not create any suitable GEMM operator for stress test."); 
    }

    // --- 2. 创建8个独立的CUDA Stream，每个线程一个 ---
    const int num_threads = 8;
    streams_.resize(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }
}

StressStrategy::~StressStrategy() {
    stop(); // Ensure threads are joined
    for (auto& stream : streams_) {
        cudaStreamDestroy(stream);
    }
}

std::string StressStrategy::get_active_operators_name() const {
    if (gemm_op_) {
        return PRECISION_NAMES.at(gemm_op_->get_descriptor().precision) + " GEMM (x8 Pipelined Threads)";
    }
    return "N/A";
}

// This is the intelligent loop, executed by each of the 8 worker threads.
void StressStrategy::worker_thread_loop(cudaStream_t stream) {
    // This is critical: set the CUDA context for this specific CPU thread.
    CUDA_CHECK(cudaSetDevice(gpu_props_.device_id));

    // Each thread gets its own independent 2-slot event pipeline.
    const int pipeline_depth = 2;
    cudaEvent_t events[pipeline_depth];
    for (int i = 0; i < pipeline_depth; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
    }

    int event_idx = 0;

    // Pre-fill the pipeline to ensure the GPU starts work immediately.
    for (int i = 0; i < pipeline_depth; ++i) {
        gemm_op_->execute(stream);
        CUDA_CHECK(cudaEventRecord(events[i], stream));
    }

    // The main efficient loop for this thread.
    while (!stop_flag_) {
        // This is the core of the gpuburner logic: wait for the oldest slot to be free.
        // The while loop with a sleep is the key to preventing 100% CPU usage.
        while (cudaEventQuery(events[event_idx]) == cudaErrorNotReady) {
            if (stop_flag_) break; // Exit promptly if stop is requested.
            // Yield the CPU. This is the most important line of code.
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        if (stop_flag_) break;
        // Check for any CUDA errors from the completed event.
        CUDA_CHECK(cudaEventSynchronize(events[event_idx]));


        // The slot is now free, so we launch a new operation into it.
        gemm_op_->execute(stream);
        CUDA_CHECK(cudaEventRecord(events[event_idx], stream));

        // Move to the next slot for the next iteration.
        event_idx = (event_idx + 1) % pipeline_depth;
    }

    // Clean up events created by this thread.
    for (int i = 0; i < pipeline_depth; ++i) {
        cudaEventDestroy(events[i]);
    }
}

void StressStrategy::start() {
    stop_flag_ = false;
    worker_threads_.clear();
    // Launch all 8 worker threads. Each will run its own pipeline on its own stream.
    for (const auto& stream : streams_) {
        worker_threads_.emplace_back(&StressStrategy::worker_thread_loop, this, stream);
    }
}

void StressStrategy::stop() {
    if (stop_flag_.exchange(true)) {
        return; // Already stopping
    }

    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    // Final synchronization to ensure all GPU work is truly finished.
    CUDA_CHECK(cudaDeviceSynchronize());
    done_flag_ = true;
}
