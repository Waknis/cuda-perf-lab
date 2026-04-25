#include "softmax/softmax_kernels.cuh"

#include "common/cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace softmax {
namespace {

constexpr int kThreadRowsBlockSize = 128;
constexpr int kBlockReduceSize = 256;
constexpr int kWarpsPerBlock = 8;
constexpr int kWarpSmallRowBlockSize = kWarpsPerBlock * 32;
constexpr float kNegInf = -3.402823466e+38F;

std::size_t div_up(std::size_t x, std::size_t y) {
  return (x + y - 1) / y;
}

int checked_grid_size(std::size_t blocks, const char* context) {
  if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    std::ostringstream message;
    message << context << " needs " << blocks << " blocks, which exceeds this benchmark's launch limit";
    throw std::runtime_error(message.str());
  }
  return static_cast<int>(blocks);
}

__device__ float warp_reduce_sum(float value) {
  unsigned mask = 0xffffffffu;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__device__ float warp_reduce_max(float value) {
  unsigned mask = 0xffffffffu;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(mask, value, offset));
  }
  return value;
}

__global__ void naive_kernel(const float* input, float* output, std::size_t rows, std::size_t cols) {
  const std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const std::size_t base = row * cols;
  float sum = 0.0f;
  for (std::size_t col = 0; col < cols; ++col) {
    sum += expf(input[base + col]);
  }

  for (std::size_t col = 0; col < cols; ++col) {
    output[base + col] = expf(input[base + col]) / sum;
  }
}

__global__ void stable_two_pass_kernel(const float* input, float* output, std::size_t rows, std::size_t cols) {
  const std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const std::size_t base = row * cols;
  float row_max = kNegInf;
  for (std::size_t col = 0; col < cols; ++col) {
    row_max = fmaxf(row_max, input[base + col]);
  }

  float sum = 0.0f;
  for (std::size_t col = 0; col < cols; ++col) {
    sum += expf(input[base + col] - row_max);
  }

  for (std::size_t col = 0; col < cols; ++col) {
    output[base + col] = expf(input[base + col] - row_max) / sum;
  }
}

__global__ void block_reduce_kernel(const float* input, float* output, std::size_t rows, std::size_t cols) {
  extern __shared__ float shared[];

  const std::size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const unsigned int tid = threadIdx.x;
  const std::size_t base = row * cols;

  float local_max = kNegInf;
  for (std::size_t col = tid; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, input[base + col]);
  }

  shared[tid] = local_max;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }

  const float row_max = shared[0];

  float local_sum = 0.0f;
  for (std::size_t col = tid; col < cols; col += blockDim.x) {
    local_sum += expf(input[base + col] - row_max);
  }

  shared[tid] = local_sum;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float denom = shared[0];
  for (std::size_t col = tid; col < cols; col += blockDim.x) {
    output[base + col] = expf(input[base + col] - row_max) / denom;
  }
}

__global__ void warp_small_row_kernel(const float* input, float* output, std::size_t rows, std::size_t cols) {
  const unsigned int tid = threadIdx.x;
  const unsigned int lane = tid % warpSize;
  const unsigned int warp = tid / warpSize;
  const std::size_t row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) {
    return;
  }

  const std::size_t base = row * cols;

  float local_max = kNegInf;
  for (std::size_t col = lane; col < cols; col += warpSize) {
    local_max = fmaxf(local_max, input[base + col]);
  }
  float row_max = warp_reduce_max(local_max);
  row_max = __shfl_sync(0xffffffffu, row_max, 0);

  float local_sum = 0.0f;
  for (std::size_t col = lane; col < cols; col += warpSize) {
    local_sum += expf(input[base + col] - row_max);
  }
  float denom = warp_reduce_sum(local_sum);
  denom = __shfl_sync(0xffffffffu, denom, 0);

  for (std::size_t col = lane; col < cols; col += warpSize) {
    output[base + col] = expf(input[base + col] - row_max) / denom;
  }
}

}  // namespace

std::vector<std::string> variant_names() {
  return {
      "naive",
      "stable_two_pass",
      "block_reduce",
      "warp_small_row",
  };
}

const char* variant_name(Variant variant) {
  switch (variant) {
    case Variant::kNaive:
      return "naive";
    case Variant::kStableTwoPass:
      return "stable_two_pass";
    case Variant::kBlockReduce:
      return "block_reduce";
    case Variant::kWarpSmallRow:
      return "warp_small_row";
  }
  return "unknown";
}

bool parse_variant(const std::string& name, Variant* variant) {
  if (name == "naive") {
    *variant = Variant::kNaive;
  } else if (name == "stable_two_pass") {
    *variant = Variant::kStableTwoPass;
  } else if (name == "block_reduce") {
    *variant = Variant::kBlockReduce;
  } else if (name == "warp_small_row") {
    *variant = Variant::kWarpSmallRow;
  } else {
    return false;
  }
  return true;
}

void launch(Variant variant,
            const float* d_input,
            float* d_output,
            std::size_t rows,
            std::size_t cols,
            cudaStream_t stream) {
  if (rows == 0 || cols == 0) {
    throw std::runtime_error("softmax requires rows > 0 and cols > 0");
  }

  switch (variant) {
    case Variant::kNaive: {
      const int grid = checked_grid_size(div_up(rows, kThreadRowsBlockSize), "naive softmax");
      naive_kernel<<<grid, kThreadRowsBlockSize, 0, stream>>>(d_input, d_output, rows, cols);
      CUDA_CHECK(cudaGetLastError());
      return;
    }
    case Variant::kStableTwoPass: {
      const int grid = checked_grid_size(div_up(rows, kThreadRowsBlockSize), "stable_two_pass softmax");
      stable_two_pass_kernel<<<grid, kThreadRowsBlockSize, 0, stream>>>(d_input, d_output, rows, cols);
      CUDA_CHECK(cudaGetLastError());
      return;
    }
    case Variant::kBlockReduce: {
      const int grid = checked_grid_size(rows, "block_reduce softmax");
      block_reduce_kernel<<<grid, kBlockReduceSize, kBlockReduceSize * sizeof(float), stream>>>(
          d_input, d_output, rows, cols);
      CUDA_CHECK(cudaGetLastError());
      return;
    }
    case Variant::kWarpSmallRow: {
      const int grid = checked_grid_size(div_up(rows, kWarpsPerBlock), "warp_small_row softmax");
      warp_small_row_kernel<<<grid, kWarpSmallRowBlockSize, 0, stream>>>(d_input, d_output, rows, cols);
      CUDA_CHECK(cudaGetLastError());
      return;
    }
  }
}

}  // namespace softmax
