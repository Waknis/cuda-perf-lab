#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

namespace reduction {

enum class Variant {
  kNaiveGlobal,
  kSharedInterleaved,
  kSharedSequential,
  kWarpShuffle,
  kVectorizedFloat4,
  kCubBaseline
};

std::vector<std::string> variant_names();
const char* variant_name(Variant variant);
bool parse_variant(const std::string& name, Variant* variant);

std::size_t required_scratch_elements(std::size_t n);
std::size_t cub_temp_storage_bytes(const float* d_input, float* d_output, std::size_t n);

void reduce(Variant variant,
            const float* d_input,
            std::size_t n,
            float* d_output,
            float* d_tmp1,
            float* d_tmp2,
            std::size_t tmp_elements,
            void* d_cub_temp,
            std::size_t cub_temp_bytes,
            cudaStream_t stream);

}  // namespace reduction
