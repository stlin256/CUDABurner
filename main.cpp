#include <iostream>
#include <csignal>
#include <memory>
#include <atomic>
#include <string>
#include <thread>
#include <chrono>

#include "core/gpu_props.hpp"
#include "monitoring/monitor.hpp"
#include "strategies/base_strategy.hpp"
#include "strategies/stress_strategy.hpp"
#include "strategies/benchmark_strategy.hpp"
#include "ui/tui.hpp"

// Global flag to signal shutdown from Ctrl+C
std::atomic<bool> g_shutdown_flag(false);

void signal_handler(int signum) {
    if (signum == SIGINT) {
        g_shutdown_flag = true;
    }
}

int main(int argc, char** argv) {
    unsigned int device_id = 0;
    std::string mode;

    // ===================================================================
    //  NEW: Interactive Menu Logic
    // ===================================================================
    // If command-line arguments are provided, use them. Otherwise, show the menu.
    if (argc > 1) {
        if (std::string(argv[1]) == "--mode" && argc > 2) {
            mode = std::string(argv[2]);
        }
    } else {
        // No arguments given, so we display the interactive menu.
        int choice = 0;
        std::cout << "=================================" << std::endl;
        std::cout << "  CUDABurner - Mode Selection    " << std::endl;
        std::cout << "=================================" << std::endl;
        std::cout << "  1. Stress Test (Max Power)     " << std::endl;
        std::cout << "  2. Benchmark Test (All Precisions)" << std::endl;
        std::cout << "---------------------------------" << std::endl;
        std::cout << "Enter your choice (1 or 2): ";
        
        std::cin >> choice;

        switch (choice) {
            case 1:
                mode = "stress";
                break;
            case 2:
                mode = "benchmark";
                break;
            default:
                std::cerr << "Invalid choice. Exiting." << std::endl;
                return 1;
        }
    }

    if (mode != "stress" && mode != "benchmark") {
        std::cerr << "Invalid mode specified. Use 'stress' or 'benchmark'." << std::endl;
        return 1;
    }

    signal(SIGINT, signal_handler);

    try {
        std::cout << "Initializing CUDABurner for device " << device_id << " in '" << mode << "' mode..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2)); // Give user time to see the mode
        
        GpuProperties gpu_props(device_id);
        GpuMonitor monitor(device_id);
        
        std::unique_ptr<BaseStrategy> strategy;
        if (mode == "benchmark") {
            strategy = std::make_unique<BenchmarkStrategy>(monitor, gpu_props);
        } else { // "stress"
            strategy = std::make_unique<StressStrategy>(gpu_props);
        }

        TUI tui(monitor, *strategy, mode);

        // Start all threads
        monitor.start();
        strategy->start();
        tui.start();

        // Wait for shutdown signal or for the strategy to complete
        while (!g_shutdown_flag && !strategy->is_done()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        if (strategy->is_done()) {
            std::cout << "\nTest completed. Waiting for exit signal..." << std::endl;
            while (!g_shutdown_flag) { // Wait for Ctrl+C if test finishes on its own
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        std::cout << "\nShutdown signal received. Stopping threads..." << std::endl;

        // Gracefully stop all threads in reverse order of dependency
        strategy->stop();
        monitor.stop();
        tui.stop();

        std::cout << "CUDABurner finished cleanly." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nAn unrecoverable error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
