#pragma once
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <nvml.h>

struct GpuState {
    unsigned int device_id;
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    char driver_version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
    unsigned int temperature;
    unsigned int power_usage; // in Watts
    unsigned int power_limit; // in Watts
    unsigned int gpu_clock;   // in MHz
    unsigned int mem_clock;   // in MHz
    unsigned int gpu_util;    // in %
    unsigned int mem_util;    // in %
    
    // Performance limiting factors
    std::string perf_limiters;
    bool throttled = false;
};

class GpuMonitor {
public:
    GpuMonitor(unsigned int device_id);
    ~GpuMonitor();

    void start();
    void stop();
    GpuState get_state();

private:
    void monitor_loop();

    unsigned int device_id_;
    nvmlDevice_t device_handle_;
    GpuState current_state_{};
    
    std::thread monitor_thread_;
    std::atomic<bool> stop_flag_;
    std::mutex state_mutex_;
};
