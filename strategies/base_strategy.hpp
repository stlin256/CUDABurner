#pragma once
#include <string>
#include <thread>
#include <atomic>

class BaseStrategy {
public:
    virtual ~BaseStrategy() { stop(); }
    virtual void start() = 0;
    virtual void stop() { 
        if(stop_flag_.exchange(true)) return;
        if(worker_thread_.joinable()) worker_thread_.join();
        done_flag_ = true;
    }
    virtual double get_current_performance() const = 0;
    virtual std::string get_active_operators_name() const = 0;

    bool is_done() const {
        return done_flag_;
    }

protected:
    std::thread worker_thread_;
    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> done_flag_{false};
};
