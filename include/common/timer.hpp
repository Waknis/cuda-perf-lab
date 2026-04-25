#pragma once

#include "common/cuda_check.hpp"

#include <cuda_runtime.h>

namespace common {

class GpuTimer {
 public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }

  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start(cudaStream_t stream = nullptr) {
    CUDA_CHECK(cudaEventRecord(start_, stream));
  }

  float stop_ms(cudaStream_t stream = nullptr) {
    CUDA_CHECK(cudaEventRecord(stop_, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
    return elapsed_ms;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

}  // namespace common
