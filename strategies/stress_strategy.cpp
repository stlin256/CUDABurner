#include "strategies/stress_strategy.hpp"
#include "operators/operator_factory.hpp"
#include "utils/helpers.hpp"
#include <chrono> // For std::this_thread::sleep_for

#include "monitoring/monitor.hpp" // We need the full definition for GpuMonitor
#include <iomanip> // For std::setprecision

StressStrategy::StressStrategy(const GpuProperties& props, GpuMonitor& monitor)
    : gpu_props_(props), monitor_(monitor) {
    CUDA_CHECK(cudaSetDevice(gpu_props_.device_id));

    std::cout << "Stress Test: Finding most power-hungry kernel..." << std::endl;

    const std::vector<Precision> precisions_to_test = {
        Precision::FP16,
        Precision::TF32,
        Precision::FP32
    };

    double max_avg_power = 0.0;
    OperatorDescriptor best_desc = {Precision::FP16, Sparsity::DENSE}; // Default

    for (const auto& p : precisions_to_test) {
        std::string notes;
        OperatorDescriptor current_desc = {p, Sparsity::DENSE};
        auto op = OperatorFactory::create_gemm_operator(current_desc, props, notes);

        if (!op) continue;

        std::cout << "  - Testing " << PRECISION_NAMES.at(p) << " for 5 seconds..." << std::flush;

        const double test_duration_seconds = 5.0;
        std::vector<double> power_samples;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        double elapsed_seconds = 0;

        // Let the GPU warm up for a second to stabilize power draw
        auto warmup_start = std::chrono::high_resolution_clock::now();
        while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - warmup_start).count() < 1.0) {
            op->execute(0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        while (elapsed_seconds < test_duration_seconds) {
            op->execute(0);
            // Synchronize to ensure the kernel has finished and power usage is high
            CUDA_CHECK(cudaDeviceSynchronize());
            power_samples.push_back(monitor_.get_state().power_usage);
            auto end_time = std::chrono::high_resolution_clock::now();
            elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        if (power_samples.empty()) continue;

        double total_power = 0;
        for (double sample : power_samples) {
            total_power += sample;
        }
        double avg_power = total_power / power_samples.size();
        std::cout << " Avg Power: " << std::fixed << std::setprecision(1) << avg_power << "W" << std::endl;

        if (avg_power > max_avg_power) {
            max_avg_power = avg_power;
            best_desc = current_desc;
        }
    }

    std::cout << "  - Selected " << PRECISION_NAMES.at(best_desc.precision)
              << " as the best kernel." << std::endl << std::endl;
    
    // Create the final, best operator for the main stress test loop
    std::string final_notes;
    gemm_op_ = OperatorFactory::create_gemm_operator(best_desc, props, final_notes);
    if (!gemm_op_) {
        throw std::runtime_error("Could not create the selected best GEMM operator for stress test.");
    }
    gflops_per_op_ = gemm_op_->get_gflops_or_gops();

    // Create 8 independent CUDA Streams for pipelining
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

       // A GPU operation has just completed, so we increment our counter.
       total_iterations_++;

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

double StressStrategy::get_current_performance() const {
   auto now = std::chrono::high_resolution_clock::now();
   double elapsed_seconds = std::chrono::duration<double>(now - start_time_).count();
   if (elapsed_seconds < 1e-6) {
       return 0.0;
   }
   double total_gops = gflops_per_op_ * total_iterations_.load();
   return total_gops / elapsed_seconds / 1000.0; // Convert to T-units
}

std::string StressStrategy::get_performance_units() const {
   if (gemm_op_ && gemm_op_->get_descriptor().precision == Precision::INT8) {
       return "TOPS";
   }
   return "TFLOPS";
}

void StressStrategy::start() {
    stop_flag_ = false;
   total_iterations_ = 0;
   start_time_ = std::chrono::high_resolution_clock::now();
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
