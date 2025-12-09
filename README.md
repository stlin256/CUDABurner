<a name="en"></a>
# CUDABurner

[中文](#zh)

A simple yet powerful stress and benchmark utility for NVIDIA GPUs. CUDABurner is designed to test the performance and stability of your graphics card by running various compute-intensive CUDA kernels.

## Screenshot

```
============================= CUDABurner v1.0 ==============================
[Global Info]
- Test Mode  : benchmark
- Target GPU : [0] NVIDIA GeForce RTX 5060 Ti
----------------------------------------------------------------------------
[GPU 0 Vitals]
- Temperature : 70°C
- Power Usage : 147 W / 180 W (81.7%)
- GPU Clock   : 2745 MHz
- Memory Clock: 13801 MHz
- Utilization : GPU 100% | VRAM 34%
----------------------------------------------------------------------------
[Benchmark Results] (In Progress...)
Precision Mode    Performance         Engine / Notes           
----------------------------------------------------------------------------
FP64      Dense   0.33 TFLOPS         CUDA Core (Native)       
FP32      Dense   15.10 TFLOPS        CUDA Core (TC Disabled)  
TF32      Dense   24.26 TFLOPS        Tensor Core (Default FP32)
FP16      Dense   49.39 TFLOPS        Tensor Core              
FP16      Sparse  96.68 TFLOPS        Sparse Tensor Core       
BF16      Dense   46.90 TFLOPS        Tensor Core              
INT8      Dense   51.33 TOPS          Tensor Core              
INT8      Sparse  367.87 TOPS         Sparse Tensor Core       
----------------------------------------------------------------------------
```

## Usage

### Prerequisites
- NVIDIA Driver
- CUDA Toolkit (I use 13.1)
- CMake (>= 3.18)
- A C++17 compliant compiler

### Build
```bash
# Clone the repository
git clone https://github.com/stlin256/CUDABurner.git
cd CUDABurner

mkdir build
cd build
cmake ..
make -j
```

### Run
The executable `CUDABurner` will be located in the `build` directory.

```bash
# Run with interactive menu
./CUDABurner

# Run Benchmark Test (all precisions)
./CUDABurner --mode benchmark

# Run Benchmark Test (specific precisions)
./CUDABurner --mode benchmark --precision FP32 FP16 TF32

# Run Stress Test
./CUDABurner --mode stress
```

## Known Issues
- **FP8/FP4 Not Implemented**: Benchmark tests for FP8 and FP4 precision are not yet implemented.
- **Insufficient Stress**: The current stress-test (`power_virus_kernel`) might not be sufficient to push GPUs to their absolute power limit.

---

<a name="zh"></a>
# CUDABurner

[English](#en)

一个简洁而强大的 NVIDIA GPU 压力与算力基准测试工具。CUDABurner 旨在通过运行多种计算密集型的 CUDA 核心，来测试您显卡的性能与稳定性。

## 程序截图

```
============================= CUDABurner v1.0 ==============================
[Global Info]
- Test Mode  : benchmark
- Target GPU : [0] NVIDIA GeForce RTX 5060 Ti
----------------------------------------------------------------------------
[GPU 0 Vitals]
- Temperature : 70°C
- Power Usage : 147 W / 180 W (81.7%)
- GPU Clock   : 2745 MHz
- Memory Clock: 13801 MHz
- Utilization : GPU 100% | VRAM 34%
----------------------------------------------------------------------------
[Benchmark Results] (In Progress...)
Precision Mode    Performance         Engine / Notes           
----------------------------------------------------------------------------
FP64      Dense   0.33 TFLOPS         CUDA Core (Native)       
FP32      Dense   15.10 TFLOPS        CUDA Core (TC Disabled)  
TF32      Dense   24.26 TFLOPS        Tensor Core (Default FP32)
FP16      Dense   49.39 TFLOPS        Tensor Core              
FP16      Sparse  96.68 TFLOPS        Sparse Tensor Core       
BF16      Dense   46.90 TFLOPS        Tensor Core              
INT8      Dense   51.33 TOPS          Tensor Core              
INT8      Sparse  367.87 TOPS         Sparse Tensor Core       
----------------------------------------------------------------------------
```

## 使用方法

### 环境要求
- NVIDIA 驱动
- CUDA Toolkit (作者使用13.1)
- CMake (>= 3.18)
- 支持 C++17 的编译器

### 编译
```bash
# 克隆仓库
git clone https://github.com/stlin256/CUDABurner.git
cd CUDABurner

mkdir build
cd build
cmake ..
make -j
```

### 运行
可执行文件 `CUDABurner` 将会生成在 `build` 目录下。

```bash
# 以交互式菜单运行
./CUDABurner

# 运行算力基准测试 (所有精度)
./CUDABurner --mode benchmark

# 运行算力基准测试 (指定精度)
./CUDABurner --mode benchmark --precision FP32 FP16 TF32

# 运行压力测试 (烤机)
./CUDABurner --mode stress
```

## 已知问题
- **FP8/FP4 未实现**: 针对 FP8 和 FP4 精度的算力基准测试尚未实现。
- **烤机压力不足**: 当前的压力测试程序不足以让现代的高功耗 GPU 达到其极限功耗。
