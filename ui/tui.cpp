#include "ui/tui.hpp"
#include "strategies/stress_strategy.hpp"
#include "strategies/benchmark_strategy.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sstream>

TUI::TUI(GpuMonitor& monitor, BaseStrategy& strategy, const std::string& mode) 
    : monitor_(monitor), strategy_(strategy), mode_(mode), stop_flag_(false) {}

TUI::~TUI() {
    stop();
}

void TUI::start() {
    stop_flag_ = false;
    render_thread_ = std::thread(&TUI::render_loop, this);
}

void TUI::stop() {
    stop_flag_ = true;
    if (render_thread_.joinable()) {
        render_thread_.join();
    }
}

void TUI::clear_screen() {
    std::cout << "\033[2J\033[1;1H";
}

void TUI::render_loop() {
    while (!stop_flag_) {
        GpuState state = monitor_.get_state();
        
        clear_screen();
        std::cout << "============================= CUDABurner v1.0 ==============================" << std::endl;
        std::cout << "[Global Info]" << std::endl;
        std::cout << "- Test Mode  : " << mode_ << std::endl;
        std::cout << "- Target GPU : [" << state.device_id << "] " << state.name << std::endl;
        std::cout << "----------------------------------------------------------------------------" << std::endl;
        std::cout << "[GPU " << state.device_id << " Vitals]" << std::endl;
        
        double power_percent = (state.power_limit > 0) ? (static_cast<double>(state.power_usage) / state.power_limit * 100.0) : 0.0;
        std::cout << "- Temperature : " << state.temperature << "°C" << std::endl;
        std::cout << "- Power Usage : " << state.power_usage << " W / " << state.power_limit << " W (" 
                  << std::fixed << std::setprecision(1) << power_percent << "%)" << std::endl;
        std::cout << "- GPU Clock   : " << state.gpu_clock << " MHz" << std::endl;
        std::cout << "- Memory Clock: " << state.mem_clock << " MHz" << std::endl;
        std::cout << "- Utilization : GPU " << state.gpu_util << "% | VRAM " << state.mem_util << "%" << std::endl;
        std::cout << "----------------------------------------------------------------------------" << std::endl;

        if (mode_ == "stress") {
            auto* stress_strategy = dynamic_cast<StressStrategy*>(&strategy_);
            if (stress_strategy) {
                std::string op_name = stress_strategy->get_active_operators_name();
                double perf = stress_strategy->get_current_performance();
                std::string units = stress_strategy->get_performance_units();

                std::cout << "[Real-time Performance]" << std::endl;
                std::cout << "- Current Ops : " << op_name << std::endl;
                std::cout << "- Performance : " << std::fixed << std::setprecision(2) << perf << " " << units << std::endl;
            }
        } else if (mode_ == "benchmark") {
            auto* benchmark_strategy = dynamic_cast<BenchmarkStrategy*>(&strategy_);
            std::cout << "[Benchmark Results]" << (strategy_.is_done() ? " (Completed)" : " (In Progress...)") << std::endl;
            std::cout << std::left
                      << std::setw(10) << "Precision"
                      << std::setw(8)  << "Mode" // New column for Dense/Sparse
                      << std::setw(20) << "Performance"
                      << std::setw(25) << "Engine / Notes"
                      << std::endl;
            std::cout << "----------------------------------------------------------------------------" << std::endl;
            if(benchmark_strategy) {
                for (const auto& res : benchmark_strategy->get_results()) {
                    std::string perf_str = res.is_supported ? 
                        (std::stringstream() << std::fixed << std::setprecision(2) << res.performance << " " << res.unit).str() : 
                        "N/A";
                    
                    std::string mode_str = (res.descriptor.sparsity == Sparsity::SPARSE) ? "Sparse" : "Dense";
                    std::cout << std::left
                              << std::setw(10) << PRECISION_NAMES.at(res.descriptor.precision)
                              << std::setw(8)  << mode_str
                              << std::setw(20) << perf_str
                              << std::setw(25) << res.notes
                              << std::endl;
                }
            }
        }
        
        std::cout << "----------------------------------------------------------------------------" << std::endl;
        if(strategy_.is_done()) {
            std::cout << "Test finished. Press Ctrl+C to exit." << std::endl;
        } else {
            std::cout << "Press Ctrl+C to stop..." << std::endl;
        }
        std::cout << std::flush;

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
