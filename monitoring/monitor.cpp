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
        nvmlReturn_t result;
        result = nvmlDeviceGetTemperature(device_handle_, sensor_type, &new_state.temperature);
        if (result != NVML_SUCCESS) new_state.temperature = 0; // Don't crash, just report 0

        unsigned int power_milliwatts;
        result = nvmlDeviceGetPowerUsage(device_handle_, &power_milliwatts);
        new_state.power_usage = (result == NVML_SUCCESS) ? power_milliwatts / 1000 : 0;

        result = nvmlDeviceGetEnforcedPowerLimit(device_handle_, &power_milliwatts);
        new_state.power_limit = (result == NVML_SUCCESS) ? power_milliwatts / 1000 : 0;
        
        nvmlUtilization_t utilization{};
        result = nvmlDeviceGetUtilizationRates(device_handle_, &utilization);
        if (result == NVML_SUCCESS) {
            new_state.gpu_util = utilization.gpu;
            new_state.mem_util = utilization.memory;
        } else {
            new_state.gpu_util = 0;
            new_state.mem_util = 0;
        }

        nvmlClockType_t clock_type_gfx = NVML_CLOCK_GRAPHICS;
        result = nvmlDeviceGetClockInfo(device_handle_, clock_type_gfx, &new_state.gpu_clock);
        if (result != NVML_SUCCESS) new_state.gpu_clock = 0;

        nvmlClockType_t clock_type_mem = NVML_CLOCK_MEM;
        result = nvmlDeviceGetClockInfo(device_handle_, clock_type_mem, &new_state.mem_clock);
        if (result != NVML_SUCCESS) new_state.mem_clock = 0;

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            current_state_ = new_state;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}
