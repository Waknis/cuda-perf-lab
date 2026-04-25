#include "reduction/reduction_kernels.cuh"

#include "common/cuda_check.hpp"

#include <cub/cub.cuh>

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace reduction {
namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxVectorizedBlocks = 4096;

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
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__global__ void naive_pairwise_kernel(const float* input, std::size_t n, float* output) {
  const std::size_t out_index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t in_index = out_index * 2;

  if (in_index >= n) {
    return;
  }

  float sum = input[in_index];
  if (in_index + 1 < n) {
    sum += input[in_index + 1];
  }
  output[out_index] = sum;
}

__global__ void shared_interleaved_kernel(const float* input, std::size_t n, float* output) {
  extern __shared__ float shared[];

  const unsigned int tid = threadIdx.x;
  const std::size_t base = blockIdx.x * blockDim.x * 2 + tid;

  float value = 0.0f;
  if (base < n) {
    value = input[base];
  }
  if (base + blockDim.x < n) {
    value += input[base + blockDim.x];
  }

  shared[tid] = value;
  __syncthreads();

  // This is the classic interleaved tree. It is intentionally kept because
  // the modulo condition creates divergence and exposes why addressing matters.
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0 && tid + stride < blockDim.x) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = shared[0];
  }
}

__global__ void shared_sequential_kernel(const float* input, std::size_t n, float* output) {
  extern __shared__ float shared[];

  const unsigned int tid = threadIdx.x;
  const std::size_t base = blockIdx.x * blockDim.x * 2 + tid;

  float value = 0.0f;
  if (base < n) {
    value = input[base];
  }
  if (base + blockDim.x < n) {
    value += input[base + blockDim.x];
  }

  shared[tid] = value;
  __syncthreads();

  // Sequential addressing removes the expensive modulo pattern and keeps
  // active threads contiguous as the tree shrinks.
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = shared[0];
  }
}

__global__ void warp_shuffle_kernel(const float* input, std::size_t n, float* output) {
  __shared__ float warp_sums[32];

  const unsigned int tid = threadIdx.x;
  const unsigned int lane = tid % warpSize;
  const unsigned int warp = tid / warpSize;
  const unsigned int warps_per_block = blockDim.x / warpSize;
  const std::size_t base = blockIdx.x * blockDim.x * 2 + tid;

  float value = 0.0f;
  if (base < n) {
    value = input[base];
  }
  if (base + blockDim.x < n) {
    value += input[base + blockDim.x];
  }

  value = warp_reduce_sum(value);

  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  float block_sum = 0.0f;
  if (warp == 0) {
    block_sum = lane < warps_per_block ? warp_sums[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      output[blockIdx.x] = block_sum;
    }
  }
}

__global__ void vectorized_float4_kernel(const float* input, std::size_t n, float* output) {
  __shared__ float warp_sums[32];

  const float4* input4 = reinterpret_cast<const float4*>(input);
  const std::size_t n4 = n / 4;
  const std::size_t tail_start = n4 * 4;
  const std::size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;

  float value = 0.0f;

  // Vectorized loads reduce instruction count for the full float4 portion.
  // The scalar tail path keeps the variant correct for any n.
  for (std::size_t i = thread_index; i < n4; i += stride) {
    const float4 v = input4[i];
    value += v.x + v.y + v.z + v.w;
  }

  for (std::size_t i = tail_start + thread_index; i < n; i += stride) {
    value += input[i];
  }

  const unsigned int tid = threadIdx.x;
  const unsigned int lane = tid % warpSize;
  const unsigned int warp = tid / warpSize;
  const unsigned int warps_per_block = blockDim.x / warpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  float block_sum = 0.0f;
  if (warp == 0) {
    block_sum = lane < warps_per_block ? warp_sums[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      output[blockIdx.x] = block_sum;
    }
  }
}

void copy_single_async(const float* input, float* output, cudaStream_t stream) {
  if (input != output) {
    CUDA_CHECK(cudaMemcpyAsync(output, input, sizeof(float), cudaMemcpyDeviceToDevice, stream));
  }
}

void ensure_workspace(std::size_t needed, std::size_t available) {
  if (needed > available) {
    throw std::runtime_error("reduction scratch workspace is too small");
  }
}

void reduce_pairwise(const float* d_input,
                     std::size_t n,
                     float* d_output,
                     float* d_tmp1,
                     float* d_tmp2,
                     std::size_t tmp_elements,
                     cudaStream_t stream) {
  if (n == 1) {
    copy_single_async(d_input, d_output, stream);
    return;
  }

  const float* current = d_input;
  std::size_t current_n = n;
  float* next = d_tmp1;

  while (current_n > 1) {
    const std::size_t partials = div_up(current_n, 2);
    ensure_workspace(partials, tmp_elements);

    const int grid = checked_grid_size(div_up(partials, kBlockSize), "naive_global");
    naive_pairwise_kernel<<<grid, kBlockSize, 0, stream>>>(current, current_n, next);
    CUDA_CHECK(cudaGetLastError());

    current = next;
    current_n = partials;
    next = (next == d_tmp1) ? d_tmp2 : d_tmp1;
  }

  copy_single_async(current, d_output, stream);
}

enum class BlockKernelKind {
  kSharedInterleaved,
  kSharedSequential,
  kWarpShuffle
};

void launch_block_kernel(BlockKernelKind kind,
                         const float* current,
                         std::size_t current_n,
                         float* next,
                         int grid,
                         cudaStream_t stream) {
  switch (kind) {
    case BlockKernelKind::kSharedInterleaved:
      shared_interleaved_kernel<<<grid, kBlockSize, kBlockSize * sizeof(float), stream>>>(current, current_n, next);
      break;
    case BlockKernelKind::kSharedSequential:
      shared_sequential_kernel<<<grid, kBlockSize, kBlockSize * sizeof(float), stream>>>(current, current_n, next);
      break;
    case BlockKernelKind::kWarpShuffle:
      warp_shuffle_kernel<<<grid, kBlockSize, 0, stream>>>(current, current_n, next);
      break;
  }
}

void reduce_block_kernel(BlockKernelKind kind,
                         const float* d_input,
                         std::size_t n,
                         float* d_output,
                         float* d_tmp1,
                         float* d_tmp2,
                         std::size_t tmp_elements,
                         cudaStream_t stream) {
  if (n == 1) {
    copy_single_async(d_input, d_output, stream);
    return;
  }

  const float* current = d_input;
  std::size_t current_n = n;
  float* next = d_tmp1;

  while (current_n > 1) {
    const std::size_t partials = div_up(current_n, kBlockSize * 2);
    ensure_workspace(partials, tmp_elements);

    const int grid = checked_grid_size(partials, "block reduction");
    launch_block_kernel(kind, current, current_n, next, grid, stream);
    CUDA_CHECK(cudaGetLastError());

    current = next;
    current_n = partials;
    next = (next == d_tmp1) ? d_tmp2 : d_tmp1;
  }

  copy_single_async(current, d_output, stream);
}

void reduce_vectorized_float4(const float* d_input,
                              std::size_t n,
                              float* d_output,
                              float* d_tmp1,
                              float* d_tmp2,
                              std::size_t tmp_elements,
                              cudaStream_t stream) {
  if (n < 4) {
    reduce_block_kernel(BlockKernelKind::kWarpShuffle, d_input, n, d_output, d_tmp1, d_tmp2, tmp_elements, stream);
    return;
  }

  const std::size_t vector_chunks = n / 4;
  const std::size_t blocks_from_data = div_up(vector_chunks, kBlockSize);
  const auto capped_blocks = std::min(static_cast<std::size_t>(kMaxVectorizedBlocks), blocks_from_data);
  const int grid = checked_grid_size(std::max<std::size_t>(1, capped_blocks), "vectorized_float4");
  ensure_workspace(static_cast<std::size_t>(grid), tmp_elements);

  vectorized_float4_kernel<<<grid, kBlockSize, 0, stream>>>(d_input, n, d_tmp1);
  CUDA_CHECK(cudaGetLastError());

  reduce_block_kernel(BlockKernelKind::kWarpShuffle,
                      d_tmp1,
                      static_cast<std::size_t>(grid),
                      d_output,
                      d_tmp2,
                      d_tmp1,
                      tmp_elements,
                      stream);
}

}  // namespace

std::vector<std::string> variant_names() {
  return {
      "naive_global",
      "shared_interleaved",
      "shared_sequential",
      "warp_shuffle",
      "vectorized_float4",
      "cub_baseline",
  };
}

const char* variant_name(Variant variant) {
  switch (variant) {
    case Variant::kNaiveGlobal:
      return "naive_global";
    case Variant::kSharedInterleaved:
      return "shared_interleaved";
    case Variant::kSharedSequential:
      return "shared_sequential";
    case Variant::kWarpShuffle:
      return "warp_shuffle";
    case Variant::kVectorizedFloat4:
      return "vectorized_float4";
    case Variant::kCubBaseline:
      return "cub_baseline";
  }
  return "unknown";
}

bool parse_variant(const std::string& name, Variant* variant) {
  if (name == "naive_global") {
    *variant = Variant::kNaiveGlobal;
  } else if (name == "shared_interleaved") {
    *variant = Variant::kSharedInterleaved;
  } else if (name == "shared_sequential") {
    *variant = Variant::kSharedSequential;
  } else if (name == "warp_shuffle") {
    *variant = Variant::kWarpShuffle;
  } else if (name == "vectorized_float4") {
    *variant = Variant::kVectorizedFloat4;
  } else if (name == "cub_baseline") {
    *variant = Variant::kCubBaseline;
  } else {
    return false;
  }
  return true;
}

std::size_t required_scratch_elements(std::size_t n) {
  return std::max<std::size_t>(1, div_up(n, 2));
}

std::size_t cub_temp_storage_bytes(const float* d_input, float* d_output, std::size_t n) {
  void* temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, d_input, d_output, n));
  return temp_storage_bytes;
}

void reduce(Variant variant,
            const float* d_input,
            std::size_t n,
            float* d_output,
            float* d_tmp1,
            float* d_tmp2,
            std::size_t tmp_elements,
            void* d_cub_temp,
            std::size_t cub_temp_bytes,
            cudaStream_t stream) {
  if (n == 0) {
    throw std::runtime_error("reduction requires n > 0");
  }

  switch (variant) {
    case Variant::kNaiveGlobal:
      reduce_pairwise(d_input, n, d_output, d_tmp1, d_tmp2, tmp_elements, stream);
      return;
    case Variant::kSharedInterleaved:
      reduce_block_kernel(BlockKernelKind::kSharedInterleaved, d_input, n, d_output, d_tmp1, d_tmp2, tmp_elements, stream);
      return;
    case Variant::kSharedSequential:
      reduce_block_kernel(BlockKernelKind::kSharedSequential, d_input, n, d_output, d_tmp1, d_tmp2, tmp_elements, stream);
      return;
    case Variant::kWarpShuffle:
      reduce_block_kernel(BlockKernelKind::kWarpShuffle, d_input, n, d_output, d_tmp1, d_tmp2, tmp_elements, stream);
      return;
    case Variant::kVectorizedFloat4:
      reduce_vectorized_float4(d_input, n, d_output, d_tmp1, d_tmp2, tmp_elements, stream);
      return;
    case Variant::kCubBaseline:
      CUDA_CHECK(cub::DeviceReduce::Sum(d_cub_temp, cub_temp_bytes, d_input, d_output, n, stream));
      return;
  }
}

}  // namespace reduction
