#include "strategies/benchmark_strategy.hpp"
#include "monitoring/monitor.hpp" // Include GpuMonitor header
#include "operators/operator_factory.hpp"
#include "utils/helpers.hpp"
#include <chrono>
#include <algorithm>
#include <map>
#include <cstdlib>
#include <string>

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

namespace {
// 候选边长的上界：按精度限制，避免 FP64 在大尺寸上单次数秒、拖垮启动时间。
int max_candidate_dim(Precision p) {
    return (p == Precision::FP64) ? 4096 : 16384;
}
}  // namespace

// 运行时尺寸自动探测：在候选边长上各短测一小段，取实测吞吐最高者。
// 显式给了 CUDABURNER_DIM（--size）时跳过探测，尊重显式选择。
int BenchmarkStrategy::autotune_dim(const OperatorDescriptor& desc) {
    if (std::getenv("CUDABURNER_DIM") != nullptr) return 0;

    const int key = static_cast<int>(desc.precision) * 10 + static_cast<int>(desc.sparsity);
    auto it = tuned_dim_cache_.find(key);
    if (it != tuned_dim_cache_.end()) return it->second;

    static const int cands[] = {2048, 4096, 8192, 16384};
    const int max_dim = max_candidate_dim(desc.precision);
    int best_dim = 0;
    double best_rate = 0.0;

    for (int d : cands) {
        if (d > max_dim) continue;
        try {
            std::string notes;
            auto op = OperatorFactory::create_gemm_operator(desc, gpu_props_, notes, d);
            if (!op) continue;
            for (int i = 0; i < 2; ++i) op->execute(0);
            CUDA_CHECK(cudaDeviceSynchronize());

            // 与正式测试同构：每 8 次提交 sync 一次，跑满 1 秒。
            // 不能用「固定 8 次迭代」——大边长的短测与长测差异极大
            // （实测：某平台上 TF32 16384³ 的 8 次短测约 120 TFLOPS，跑满 10 秒只有 60）。
            const double tune_seconds = 1.0;
            const int kSyncEvery = 8;
            long long iters = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            double elapsed = 0.0;
            while (elapsed < tune_seconds) {
                for (int k = 0; k < kSyncEvery; ++k) {
                    op->execute(0);
                    iters++;
                }
                CUDA_CHECK(cudaDeviceSynchronize());
                elapsed = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t0).count();
            }
            if (iters == 0 || elapsed <= 0.0) continue;

            // get_gflops_or_gops() 返回单次 GEMM 的 GFLOP(GOPS)
            double rate = iters * op->get_gflops_or_gops() / elapsed / 1000.0;  // -> TFLOPS/TOPS
            if (rate > best_rate) {
                best_rate = rate;
                best_dim = d;
            }
        } catch (const std::exception&) {
            cudaGetLastError();
        }
    }

    tuned_dim_cache_[key] = best_dim;
    return best_dim;
}

void BenchmarkStrategy::run_loop() {
    CUDA_CHECK(cudaSetDevice(gpu_props_.device_id));

    const double test_duration_seconds = 10.0;

    for (const auto& test_desc : tests_to_run_) {
        if (stop_flag_) break;

        BenchmarkResult current_result;
        current_result.descriptor = test_desc;

        // 逐项隔离：某些后端对特定 compute/数据类型组合返回
        // CUBLAS_STATUS_NOT_SUPPORTED（例如 mcblasLt 的 INT8 稀疏路径抛异常），
        // 原实现无 try/catch，会让整个进程 terminate，丢掉后面所有项。
        // 整项包在 try/catch 内：某些后端对特定 compute/数据类型组合
        // 返回 CUBLAS_STATUS_NOT_SUPPORTED，且是在 execute() 阶段抛出（例如 mcblasLt 的
        // INT8 稀疏路径）。原实现无捕获，会让整个进程 terminate，丢掉后续所有项。
        const int tuned_dim = autotune_dim(test_desc);

        std::unique_ptr<BaseOperator> op_holder;
        try {
            op_holder = OperatorFactory::create_gemm_operator(test_desc, gpu_props_, current_result.notes, tuned_dim);
            if (tuned_dim > 0) {
                // factory 会写 out_notes，所以在它之后追加，标明本次实测所用边长
                current_result.notes += " @" + std::to_string(tuned_dim);
            }
            BaseOperator* op = op_holder.get();
            if (op == nullptr) {
                if (current_result.notes.empty()) current_result.is_supported = false;
            } else {
                current_result.is_supported = true;
                for (int i = 0; i < 10; ++i) {
                    op->execute(0);
                }
                CUDA_CHECK(cudaDeviceSynchronize());

                // 防积压：原实现在 10 秒内异步提交成千上万次而不 sync，
                // 在快卡上会积压到分钟级（提交 μs 级、单次 GEMM ms 级，可积压数百万次调用）。
                // 改为每 kSyncEvery 次提交同步一次，队列深度有界。
                const int kSyncEvery = 16;
                long long iterations = 0;
                auto start_time = std::chrono::high_resolution_clock::now();
                double elapsed_seconds = 0;

                while (elapsed_seconds < test_duration_seconds) {
                    for (int k = 0; k < kSyncEvery; ++k) {
                        op->execute(0);
                        iterations++;
                    }
                    CUDA_CHECK(cudaDeviceSynchronize());
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
        } catch (const std::exception& e) {
            current_result.is_supported = false;
            current_result.is_native = false;
            current_result.performance = 0.0;
            current_result.notes = std::string("unsupported on this backend: ") + e.what();
            cudaGetLastError();  // 清掉可能残留的错误状态，避免污染后续项
        }
        
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            results_.push_back(current_result);
        }

        // Pause for 5 seconds between tests to allow GPU to cool down
        if (!stop_flag_) {
            for (int i = 5; i > 0; --i) {
                countdown_seconds_ = i;
                if(stop_flag_) break;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            countdown_seconds_ = 0;
        }
    }
    done_flag_ = true;
}
