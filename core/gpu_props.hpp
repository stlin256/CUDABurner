#pragma once
#include <cuda_runtime.h>
#include <string>

struct GpuProperties {
    int device_id;
    std::string name;
    int cc_major; // Compute Capability Major
    int cc_minor; // Compute Capability Minor

    GpuProperties(int dev_id = 0) : device_id(dev_id) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev_id);
        name = prop.name;
        cc_major = prop.major;
        cc_minor = prop.minor;
    }
};
