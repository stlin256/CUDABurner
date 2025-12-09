#pragma once
#include "monitoring/monitor.hpp"
#include "strategies/base_strategy.hpp"
#include <atomic>
#include <thread>
#include <string>

class TUI {
public:
    TUI(GpuMonitor& monitor, BaseStrategy& strategy, const std::string& mode);
    ~TUI();
    void start();
    void stop();

private:
    void render_loop();
    void clear_screen();

    GpuMonitor& monitor_;
    BaseStrategy& strategy_;
    std::string mode_;
    std::atomic<bool> stop_flag_{false};
    std::thread render_thread_;
};
