// 内存子系统带宽测试（CUDABurner 不含此项）
#include "monitoring/memory_bench.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <functional>

namespace {

double gbps(size_t bytes, double sec) { return sec > 0 ? bytes / sec / 1e9 : 0.0; }

double timeit(int iters, const std::function<void()>& fn, bool sync_each = false) {
    fn();  // warmup
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) { fn(); if (sync_each) cudaDeviceSynchronize(); }
    cudaDeviceSynchronize();
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
}

}  // namespace

int run_memory_bench(int device_id) {
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    if (device_id < 0 || device_id >= ndev) {
        fprintf(stderr, "invalid device %d (count=%d)\n", device_id, ndev);
        return 1;
    }

    printf("============================= CUDABurner memory ============================\n");
    printf("[Global Info]\n- Test Mode  : memory\n- Target GPU : [%d]\n", device_id);
    printf("---------------------------------------------------------------------------\n");

    cudaSetDevice(device_id);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device_id);
    printf("- GPU Name   : %s\n", prop.name);
    printf("- Memory     : %.1f GiB\n", prop.totalGlobalMem / 1073741824.0);
    printf("---------------------------------------------------------------------------\n");
    printf("%-22s %-14s %s\n", "Test", "Bandwidth", "Notes");
    printf("---------------------------------------------------------------------------\n");

    const size_t N = 256ull << 20;  // 256 MiB
    void* a = nullptr; void* b = nullptr;
    void* h = nullptr;
    if (cudaMalloc(&a, N) != cudaSuccess || cudaMalloc(&b, N) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n");
        return 1;
    }
    cudaMemset(a, 1, N);

    // HBM D2D（读 + 写）
    {
        double dt = timeit(10, [&]{ cudaMemcpy(b, a, N, cudaMemcpyDeviceToDevice); });
        printf("%-22s %8.1f GB/s  %s\n", "HBM D2D copy", gbps(2.0 * N * 10, dt),
               "(read+write, 256 MiB x10)");
    }

    // H2D / D2H（pinned）
    if (cudaMallocHost(&h, N) == cudaSuccess) {
        double dt_h2d = timeit(5, [&]{ cudaMemcpy(a, h, N, cudaMemcpyHostToDevice); });
        printf("%-22s %8.1f GB/s  %s\n", "PCIe H2D copy", gbps((double)N * 5, dt_h2d), "(pinned)");
        double dt_d2h = timeit(5, [&]{ cudaMemcpy(h, a, N, cudaMemcpyDeviceToHost); });
        printf("%-22s %8.1f GB/s  %s\n", "PCIe D2H copy", gbps((double)N * 5, dt_d2h), "(pinned)");
    } else {
        printf("%-22s %-14s %s\n", "PCIe H2D/D2H", "skipped", "cudaMallocHost failed");
    }

    // P2P：与相邻卡、与最远卡
    int peers[] = {-1, -1};
    int cand[] = {device_id + 1, ndev - 1 - device_id};
    int found = 0;
    for (int c : cand) {
        if (c == device_id || c < 0 || c >= ndev) continue;
        int can = 0;
        cudaDeviceCanAccessPeer(&can, device_id, c);
        if (can) { peers[found++] = c; if (found >= 2) break; }
    }
    for (int i = 0; i < found; ++i) {
        int peer = peers[i];
        cudaSetDevice(device_id);
        cudaDeviceEnablePeerAccess(peer, 0);
        void* dp = nullptr;
        cudaSetDevice(peer); cudaMalloc(&dp, N);
        cudaSetDevice(device_id);
        if (dp) {
            double dt = timeit(10, [&]{ cudaMemcpyPeer(dp, peer, a, device_id, N); });
            char label[64];
            snprintf(label, sizeof(label), "P2P %d->%d", device_id, peer);
            printf("%-22s %8.1f GB/s  %s\n", label, gbps((double)N * 10, dt), "(cudaMemcpyPeer, 256 MiB x10)");
            cudaSetDevice(peer); cudaFree(dp); cudaSetDevice(device_id);
        }
    }
    if (found == 0) {
        printf("%-22s %-14s %s\n", "P2P", "unavailable", "no peer access");
    }

    printf("---------------------------------------------------------------------------\n");
    printf("注意：cudaMemcpyPeer 可能走 PCIe 而非 MetaXLink；\n");
    printf("      MetaXLink 真实带宽需用 MCCL(libmccl) all-to-all 验证（本工具未覆盖）。\n");
    printf("MEMORY_BENCH_DONE\n");

    if (h) cudaFreeHost(h);
    cudaFree(a);
    cudaFree(b);
    return 0;
}
