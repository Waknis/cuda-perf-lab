#include "monte_carlo/monte_carlo_kernels.cuh"

#include "common/cuda_check.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace monte_carlo {
namespace {

constexpr int kBlockSize = 256;

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

__global__ void init_xorwow_kernel(curandState* states, std::size_t paths, unsigned long long seed) {
  const std::size_t path = blockIdx.x * blockDim.x + threadIdx.x;
  if (path >= paths) {
    return;
  }
  curand_init(seed, static_cast<unsigned long long>(path), 0ULL, &states[path]);
}

__global__ void init_philox_kernel(curandStatePhilox4_32_10_t* states,
                                   std::size_t paths,
                                   unsigned long long seed) {
  const std::size_t path = blockIdx.x * blockDim.x + threadIdx.x;
  if (path >= paths) {
    return;
  }
  curand_init(seed, static_cast<unsigned long long>(path), 0ULL, &states[path]);
}

template <typename State>
__device__ float simulate_one_path(State* state, Params params) {
  if (params.steps == 1) {
    const float z = curand_normal(state);
    const float drift = (params.rate - 0.5f * params.vol * params.vol) * params.maturity;
    const float diffusion = params.vol * sqrtf(params.maturity) * z;
    const float terminal = params.spot * expf(drift + diffusion);
    return fmaxf(terminal - params.strike, 0.0f);
  }

  const float dt = params.maturity / static_cast<float>(params.steps);
  const float drift_step = (params.rate - 0.5f * params.vol * params.vol) * dt;
  const float diffusion_step = params.vol * sqrtf(dt);
  float log_s = logf(params.spot);

  for (int step = 0; step < params.steps; ++step) {
    const float z = curand_normal(state);
    log_s += drift_step + diffusion_step * z;
  }

  const float terminal = expf(log_s);
  return fmaxf(terminal - params.strike, 0.0f);
}

template <typename State>
__device__ float simulate_antithetic_path(State* state, Params params) {
  if (params.steps == 1) {
    const float z = curand_normal(state);
    const float drift = (params.rate - 0.5f * params.vol * params.vol) * params.maturity;
    const float diffusion_scale = params.vol * sqrtf(params.maturity);
    const float terminal_a = params.spot * expf(drift + diffusion_scale * z);
    const float terminal_b = params.spot * expf(drift - diffusion_scale * z);
    const float payoff_a = fmaxf(terminal_a - params.strike, 0.0f);
    const float payoff_b = fmaxf(terminal_b - params.strike, 0.0f);
    return 0.5f * (payoff_a + payoff_b);
  }

  const float dt = params.maturity / static_cast<float>(params.steps);
  const float drift_step = (params.rate - 0.5f * params.vol * params.vol) * dt;
  const float diffusion_step = params.vol * sqrtf(dt);
  float log_s_a = logf(params.spot);
  float log_s_b = logf(params.spot);

  for (int step = 0; step < params.steps; ++step) {
    const float z = curand_normal(state);
    log_s_a += drift_step + diffusion_step * z;
    log_s_b += drift_step - diffusion_step * z;
  }

  const float payoff_a = fmaxf(expf(log_s_a) - params.strike, 0.0f);
  const float payoff_b = fmaxf(expf(log_s_b) - params.strike, 0.0f);
  return 0.5f * (payoff_a + payoff_b);
}

__global__ void simulate_xorwow_kernel(Params params,
                                       std::size_t paths,
                                       curandState* states,
                                       float* payoffs,
                                       float* payoff_squares) {
  const std::size_t path = blockIdx.x * blockDim.x + threadIdx.x;
  if (path >= paths) {
    return;
  }

  curandState state = states[path];
  const float payoff = simulate_one_path(&state, params);
  states[path] = state;
  payoffs[path] = payoff;
  payoff_squares[path] = payoff * payoff;
}

__global__ void simulate_philox_kernel(Params params,
                                       std::size_t paths,
                                       curandStatePhilox4_32_10_t* states,
                                       float* payoffs,
                                       float* payoff_squares) {
  const std::size_t path = blockIdx.x * blockDim.x + threadIdx.x;
  if (path >= paths) {
    return;
  }

  curandStatePhilox4_32_10_t state = states[path];
  const float payoff = simulate_one_path(&state, params);
  states[path] = state;
  payoffs[path] = payoff;
  payoff_squares[path] = payoff * payoff;
}

__global__ void simulate_philox_antithetic_kernel(Params params,
                                                  std::size_t paths,
                                                  curandStatePhilox4_32_10_t* states,
                                                  float* payoffs,
                                                  float* payoff_squares) {
  const std::size_t path = blockIdx.x * blockDim.x + threadIdx.x;
  if (path >= paths) {
    return;
  }

  curandStatePhilox4_32_10_t state = states[path];
  const float payoff = simulate_antithetic_path(&state, params);
  states[path] = state;
  payoffs[path] = payoff;
  payoff_squares[path] = payoff * payoff;
}

}  // namespace

std::vector<std::string> variant_names() {
  return {
      "cpu_baseline",
      "gpu_naive_curand",
      "gpu_philox",
      "gpu_philox_antithetic",
  };
}

const char* variant_name(Variant variant) {
  switch (variant) {
    case Variant::kCpuBaseline:
      return "cpu_baseline";
    case Variant::kGpuNaiveCurand:
      return "gpu_naive_curand";
    case Variant::kGpuPhilox:
      return "gpu_philox";
    case Variant::kGpuPhiloxAntithetic:
      return "gpu_philox_antithetic";
  }
  return "unknown";
}

bool parse_variant(const std::string& name, Variant* variant) {
  if (name == "cpu_baseline") {
    *variant = Variant::kCpuBaseline;
  } else if (name == "gpu_naive_curand") {
    *variant = Variant::kGpuNaiveCurand;
  } else if (name == "gpu_philox") {
    *variant = Variant::kGpuPhilox;
  } else if (name == "gpu_philox_antithetic") {
    *variant = Variant::kGpuPhiloxAntithetic;
  } else {
    return false;
  }
  return true;
}

void initialize_xorwow(curandState* states,
                       std::size_t paths,
                       std::uint64_t seed,
                       cudaStream_t stream) {
  const int grid = checked_grid_size(div_up(paths, kBlockSize), "xorwow state init");
  init_xorwow_kernel<<<grid, kBlockSize, 0, stream>>>(states, paths, static_cast<unsigned long long>(seed));
  CUDA_CHECK(cudaGetLastError());
}

void initialize_philox(curandStatePhilox4_32_10_t* states,
                       std::size_t paths,
                       std::uint64_t seed,
                       cudaStream_t stream) {
  const int grid = checked_grid_size(div_up(paths, kBlockSize), "philox state init");
  init_philox_kernel<<<grid, kBlockSize, 0, stream>>>(states, paths, static_cast<unsigned long long>(seed));
  CUDA_CHECK(cudaGetLastError());
}

void simulate(Variant variant,
              const Params& params,
              std::size_t paths,
              curandState* xorwow_states,
              curandStatePhilox4_32_10_t* philox_states,
              float* payoffs,
              float* payoff_squares,
              cudaStream_t stream) {
  const int grid = checked_grid_size(div_up(paths, kBlockSize), variant_name(variant));
  switch (variant) {
    case Variant::kGpuNaiveCurand:
      simulate_xorwow_kernel<<<grid, kBlockSize, 0, stream>>>(params, paths, xorwow_states, payoffs, payoff_squares);
      CUDA_CHECK(cudaGetLastError());
      return;
    case Variant::kGpuPhilox:
      simulate_philox_kernel<<<grid, kBlockSize, 0, stream>>>(params, paths, philox_states, payoffs, payoff_squares);
      CUDA_CHECK(cudaGetLastError());
      return;
    case Variant::kGpuPhiloxAntithetic:
      simulate_philox_antithetic_kernel<<<grid, kBlockSize, 0, stream>>>(
          params, paths, philox_states, payoffs, payoff_squares);
      CUDA_CHECK(cudaGetLastError());
      return;
    case Variant::kCpuBaseline:
      throw std::runtime_error("cpu_baseline does not launch a GPU simulation kernel");
  }
}

}  // namespace monte_carlo
