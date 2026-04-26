#pragma once

#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace monte_carlo {

enum class Variant {
  kCpuBaseline,
  kGpuNaiveCurand,
  kGpuPhilox,
  kGpuPhiloxAntithetic
};

struct Params {
  float spot = 100.0f;
  float strike = 100.0f;
  float rate = 0.05f;
  float vol = 0.2f;
  float maturity = 1.0f;
  int steps = 1;
};

std::vector<std::string> variant_names();
const char* variant_name(Variant variant);
bool parse_variant(const std::string& name, Variant* variant);

void initialize_xorwow(curandState* states,
                       std::size_t paths,
                       std::uint64_t seed,
                       cudaStream_t stream);

void initialize_philox(curandStatePhilox4_32_10_t* states,
                       std::size_t paths,
                       std::uint64_t seed,
                       cudaStream_t stream);

void simulate(Variant variant,
              const Params& params,
              std::size_t paths,
              curandState* xorwow_states,
              curandStatePhilox4_32_10_t* philox_states,
              float* payoffs,
              float* payoff_squares,
              cudaStream_t stream);

}  // namespace monte_carlo
