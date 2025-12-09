#include "monitoring/monitor.hpp"
#include "utils/helpers.hpp"
#include <chrono>
#include <cstring>

GpuMonitor::GpuMonitor(unsigned int device_id) : device_id_(device_id), stop_flag_(false) {
    NVML_CHECK(nvmlInit());
    NVML_CHECK(nvmlDeviceGetHandleByIndex(device_id_, &device_handle_));
    NVML_CHECK(nvmlDeviceGetName(device_handle_, current_state_.name, NVML_DEVICE_NAME_BUFFER_SIZE));
    current_state_.device_id = device_id_;
}

GpuMonitor::~GpuMonitor() {
    stop();
    // nvmlShutdown() is called in the main executable to avoid conflicts if multiple monitors were created.
    // In our case, it's safe here, but as a good practice, it's often managed globally.
    nvmlShutdown();
}

void GpuMonitor::start() {
    stop_flag_ = false;
    monitor_thread_ = std::thread(&GpuMonitor::monitor_loop, this);
}

void GpuMonitor::stop() {
    stop_flag_ = true;
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

GpuState GpuMonitor::get_state() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_state_;
}

void GpuMonitor::monitor_loop() {
    while (!stop_flag_) {
        GpuState new_state{};
        new_state.device_id = device_id_;
        nvmlDeviceGetName(device_handle_, new_state.name, NVML_DEVICE_NAME_BUFFER_SIZE);

        nvmlTemperatureSensors_t sensor_type = NVML_TEMPERATURE_GPU;
        nvmlDeviceGetTemperature(device_handle_, sensor_type, &new_state.temperature);

        unsigned int power_milliwatts;
        nvmlDeviceGetPowerUsage(device_handle_, &power_milliwatts);
        new_state.power_usage = power_milliwatts / 1000;

        nvmlDeviceGetEnforcedPowerLimit(device_handle_, &power_milliwatts);
        new_state.power_limit = power_milliwatts / 1000;
        
        nvmlUtilization_t utilization{};
        nvmlDeviceGetUtilizationRates(device_handle_, &utilization);
        new_state.gpu_util = utilization.gpu;
        new_state.mem_util = utilization.memory;

        nvmlClockType_t clock_type_gfx = NVML_CLOCK_GRAPHICS;
        nvmlDeviceGetClockInfo(device_handle_, clock_type_gfx, &new_state.gpu_clock);

        nvmlClockType_t clock_type_mem = NVML_CLOCK_MEM;
        nvmlDeviceGetClockInfo(device_handle_, clock_type_mem, &new_state.mem_clock);

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            current_state_ = new_state;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}
