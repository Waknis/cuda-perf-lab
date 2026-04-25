#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

namespace softmax {

enum class Variant {
  kNaive,
  kStableTwoPass,
  kBlockReduce,
  kWarpSmallRow
};

std::vector<std::string> variant_names();
const char* variant_name(Variant variant);
bool parse_variant(const std::string& name, Variant* variant);

void launch(Variant variant,
            const float* d_input,
            float* d_output,
            std::size_t rows,
            std::size_t cols,
            cudaStream_t stream);

}  // namespace softmax
