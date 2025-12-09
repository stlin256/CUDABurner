#include "power_virus_kernel.hpp"

// This kernel is designed to maximize ALU activity on CUDA cores.
// It contains a long chain of dependent FMA (Fused Multiply-Add) instructions
// for both floating-point and integer types to keep the ALUs busy.
// The "volatile" keyword prevents the compiler from optimizing away the calculations.
__global__ void PowerVirusKernel() {
    float f_val = 1.01f;
    int i_val = 1;

    #pragma unroll(256)
    for(int i = 0; i < 1024; ++i) {
        f_val = fmaf(f_val, 1.000001f, 0.000001f);
        i_val = i_val * 3 + 7;
    }
    
    // Write back to a volatile pointer to ensure the work is not optimized out.
    volatile float* out_f = &f_val;
    *out_f = f_val;
    volatile int* out_i = &i_val;
    *out_i = i_val;
}

void launch_power_virus_kernel(cudaStream_t stream) {
    // Launch a massive number of threads to saturate the entire GPU.
    // 2048 blocks * 1024 threads/block = over 2 million threads.
    PowerVirusKernel<<<2048, 1024, 0, stream>>>();
}
