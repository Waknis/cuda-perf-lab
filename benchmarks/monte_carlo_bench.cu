#include "common/cuda_check.hpp"
#include "common/device_info.hpp"
#include "common/stats.hpp"
#include "common/timer.hpp"
#include "monte_carlo/monte_carlo_kernels.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CUDA_PERF_LAB_COMMIT_HASH
#define CUDA_PERF_LAB_COMMIT_HASH "unknown"
#endif

namespace {

constexpr const char* kKernelName = "monte_carlo";
constexpr double kConfidenceZ95 = 1.96;
constexpr std::size_t kLargePathCount = 1048576;
constexpr double kExtremeRelError = 0.25;

struct Options {
  std::string variant = "all";
  std::size_t paths = 1 << 20;
  int steps = 1;
  int iters = 20;
  int warmup = 5;
  std::uint64_t seed = 1234;
  double spot = 100.0;
  double strike = 100.0;
  double rate = 0.05;
  double vol = 0.2;
  double maturity = 1.0;
  bool csv = false;
  bool json = false;
  bool help = false;
};

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ > 0) {
      CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  T* get() const { return ptr_; }
  std::size_t count() const { return count_; }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

class DeviceBytes {
 public:
  explicit DeviceBytes(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ > 0) {
      CUDA_CHECK(cudaMalloc(&ptr_, bytes_));
    }
  }

  DeviceBytes(const DeviceBytes&) = delete;
  DeviceBytes& operator=(const DeviceBytes&) = delete;

  ~DeviceBytes() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  void* get() const { return ptr_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

struct ResultRow {
  std::string variant;
  std::size_t paths = 0;
  int steps = 0;
  double latency_median_us = std::numeric_limits<double>::quiet_NaN();
  double latency_p95_us = std::numeric_limits<double>::quiet_NaN();
  double paths_per_second = std::numeric_limits<double>::quiet_NaN();
  double estimated_price = std::numeric_limits<double>::quiet_NaN();
  double analytical_price = std::numeric_limits<double>::quiet_NaN();
  double abs_error = std::numeric_limits<double>::quiet_NaN();
  double rel_error = std::numeric_limits<double>::quiet_NaN();
  double standard_error = std::numeric_limits<double>::quiet_NaN();
  double confidence_interval_95_low = std::numeric_limits<double>::quiet_NaN();
  double confidence_interval_95_high = std::numeric_limits<double>::quiet_NaN();
  double baseline_ratio = std::numeric_limits<double>::quiet_NaN();
  std::string gpu_name;
  std::string cuda_runtime_version;
  std::string commit_hash;
};

std::string usage(const char* program) {
  std::ostringstream out;
  out << "Usage: " << program << " [options]\n"
      << "\n"
      << "Options:\n"
      << "  --variant <name>     cpu_baseline, gpu_naive_curand, gpu_philox,\n"
      << "                       gpu_philox_antithetic, all\n"
      << "  --paths <count>      Number of Monte Carlo paths/samples\n"
      << "  --steps <count>      Time steps per path; 1 uses direct terminal GBM\n"
      << "  --iters <count>      Timed iterations\n"
      << "  --warmup <count>     Warmup iterations\n"
      << "  --seed <value>       Deterministic cuRAND seed\n"
      << "  --spot <value>       Spot price, default 100\n"
      << "  --strike <value>     Strike price, default 100\n"
      << "  --rate <value>       Risk-free rate, default 0.05\n"
      << "  --vol <value>        Volatility, default 0.2\n"
      << "  --maturity <value>   Time to maturity, default 1.0\n"
      << "  --csv                Print CSV only\n"
      << "  --json               Print JSON only\n"
      << "  --help               Show this message\n";
  return out.str();
}

std::string next_value(int& i, int argc, char** argv, const std::string& flag) {
  if (i + 1 >= argc) {
    throw std::runtime_error("missing value for " + flag);
  }
  ++i;
  return argv[i];
}

Options parse_cli(int argc, char** argv) {
  Options opts;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    std::string value;
    const auto equals = arg.find('=');
    if (equals != std::string::npos) {
      value = arg.substr(equals + 1);
      arg = arg.substr(0, equals);
    }

    auto get_value = [&]() {
      if (!value.empty()) {
        return value;
      }
      return next_value(i, argc, argv, arg);
    };

    if (arg == "--variant") {
      opts.variant = get_value();
    } else if (arg == "--paths") {
      opts.paths = static_cast<std::size_t>(std::stoull(get_value()));
    } else if (arg == "--steps") {
      opts.steps = std::stoi(get_value());
    } else if (arg == "--iters") {
      opts.iters = std::stoi(get_value());
    } else if (arg == "--warmup") {
      opts.warmup = std::stoi(get_value());
    } else if (arg == "--seed") {
      opts.seed = static_cast<std::uint64_t>(std::stoull(get_value()));
    } else if (arg == "--spot") {
      opts.spot = std::stod(get_value());
    } else if (arg == "--strike") {
      opts.strike = std::stod(get_value());
    } else if (arg == "--rate") {
      opts.rate = std::stod(get_value());
    } else if (arg == "--vol") {
      opts.vol = std::stod(get_value());
    } else if (arg == "--maturity") {
      opts.maturity = std::stod(get_value());
    } else if (arg == "--csv") {
      opts.csv = true;
    } else if (arg == "--json") {
      opts.json = true;
    } else if (arg == "--help" || arg == "-h") {
      opts.help = true;
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }

  if (opts.csv && opts.json) {
    throw std::runtime_error("--csv and --json are mutually exclusive");
  }
  if (opts.paths < 2) {
    throw std::runtime_error("--paths must be greater than one");
  }
  if (opts.steps <= 0) {
    throw std::runtime_error("--steps must be greater than zero");
  }
  if (opts.iters <= 0) {
    throw std::runtime_error("--iters must be greater than zero");
  }
  if (opts.warmup < 0) {
    throw std::runtime_error("--warmup must be non-negative");
  }
  if (!(opts.spot > 0.0) || !(opts.strike > 0.0)) {
    throw std::runtime_error("--spot and --strike must be greater than zero");
  }
  if (!(opts.vol >= 0.0) || !(opts.maturity > 0.0)) {
    throw std::runtime_error("--vol must be non-negative and --maturity must be greater than zero");
  }

  return opts;
}

std::vector<monte_carlo::Variant> variants_to_run(const std::string& requested) {
  if (requested == "all") {
    return {
        monte_carlo::Variant::kCpuBaseline,
        monte_carlo::Variant::kGpuNaiveCurand,
        monte_carlo::Variant::kGpuPhilox,
        monte_carlo::Variant::kGpuPhiloxAntithetic,
    };
  }

  monte_carlo::Variant variant{};
  if (!monte_carlo::parse_variant(requested, &variant)) {
    std::ostringstream message;
    message << "unsupported variant '" << requested << "'. Supported variants: all";
    for (const auto& name : monte_carlo::variant_names()) {
      message << ", " << name;
    }
    throw std::runtime_error(message.str());
  }
  return {variant};
}

bool is_gpu_variant(monte_carlo::Variant variant) {
  return variant != monte_carlo::Variant::kCpuBaseline;
}

bool uses_xorwow(monte_carlo::Variant variant) {
  return variant == monte_carlo::Variant::kGpuNaiveCurand;
}

bool uses_philox(monte_carlo::Variant variant) {
  return variant == monte_carlo::Variant::kGpuPhilox ||
         variant == monte_carlo::Variant::kGpuPhiloxAntithetic;
}

monte_carlo::Params make_gpu_params(const Options& opts) {
  monte_carlo::Params params;
  params.spot = static_cast<float>(opts.spot);
  params.strike = static_cast<float>(opts.strike);
  params.rate = static_cast<float>(opts.rate);
  params.vol = static_cast<float>(opts.vol);
  params.maturity = static_cast<float>(opts.maturity);
  params.steps = opts.steps;
  return params;
}

double normal_cdf(double x) {
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double black_scholes_call_price(const Options& opts) {
  if (opts.vol == 0.0) {
    const double forward_terminal = opts.spot * std::exp(opts.rate * opts.maturity);
    return std::exp(-opts.rate * opts.maturity) * std::max(forward_terminal - opts.strike, 0.0);
  }

  const double sqrt_t = std::sqrt(opts.maturity);
  const double sigma_sqrt_t = opts.vol * sqrt_t;
  const double d1 = (std::log(opts.spot / opts.strike) +
                     (opts.rate + 0.5 * opts.vol * opts.vol) * opts.maturity) /
                    sigma_sqrt_t;
  const double d2 = d1 - sigma_sqrt_t;
  return opts.spot * normal_cdf(d1) -
         opts.strike * std::exp(-opts.rate * opts.maturity) * normal_cdf(d2);
}

int checked_cub_item_count(std::size_t paths) {
  if (paths > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("CUB DeviceReduce item count exceeds int range for this benchmark");
  }
  return static_cast<int>(paths);
}

std::size_t cub_temp_storage_bytes(const float* d_input, float* d_output, int items) {
  void* temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, d_input, d_output, items));
  return temp_storage_bytes;
}

void reduce_sum(const float* d_input,
                float* d_output,
                int items,
                void* d_temp_storage,
                std::size_t temp_storage_bytes,
                cudaStream_t stream) {
  CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, items, stream));
}

ResultRow benchmark_cpu_baseline(const Options& opts,
                                 double analytical_price,
                                 const std::string& gpu_name,
                                 const std::string& runtime_version) {
  for (int i = 0; i < opts.warmup; ++i) {
    volatile double sink = black_scholes_call_price(opts);
    (void)sink;
  }

  std::vector<double> times_us;
  times_us.reserve(static_cast<std::size_t>(opts.iters));
  double last_price = analytical_price;

  for (int i = 0; i < opts.iters; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    last_price = black_scholes_call_price(opts);
    const auto stop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::micro> elapsed = stop - start;
    times_us.push_back(elapsed.count());
  }

  ResultRow row;
  row.variant = monte_carlo::variant_name(monte_carlo::Variant::kCpuBaseline);
  row.paths = opts.paths;
  row.steps = opts.steps;
  row.latency_median_us = common::median(times_us);
  row.latency_p95_us = common::percentile(times_us, 0.95);
  row.estimated_price = last_price;
  row.analytical_price = analytical_price;
  row.abs_error = std::abs(last_price - analytical_price);
  row.rel_error = row.abs_error / std::max(1.0, std::abs(analytical_price));
  row.gpu_name = gpu_name;
  row.cuda_runtime_version = runtime_version;
  row.commit_hash = CUDA_PERF_LAB_COMMIT_HASH;
  return row;
}

ResultRow benchmark_gpu_variant(monte_carlo::Variant variant,
                                const Options& opts,
                                double analytical_price,
                                const std::string& gpu_name,
                                const std::string& runtime_version,
                                curandState* d_xorwow_states,
                                curandStatePhilox4_32_10_t* d_philox_states,
                                float* d_payoffs,
                                float* d_payoff_squares,
                                float* d_sum_payoff,
                                float* d_sum_payoff_square,
                                int cub_items,
                                void* d_cub_temp,
                                std::size_t cub_temp_bytes,
                                cudaStream_t stream) {
  const monte_carlo::Params params = make_gpu_params(opts);

  for (int i = 0; i < opts.warmup; ++i) {
    monte_carlo::simulate(variant,
                          params,
                          opts.paths,
                          d_xorwow_states,
                          d_philox_states,
                          d_payoffs,
                          d_payoff_squares,
                          stream);
    reduce_sum(d_payoffs, d_sum_payoff, cub_items, d_cub_temp, cub_temp_bytes, stream);
    reduce_sum(d_payoff_squares, d_sum_payoff_square, cub_items, d_cub_temp, cub_temp_bytes, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> times_us;
  times_us.reserve(static_cast<std::size_t>(opts.iters));
  common::GpuTimer timer;

  for (int i = 0; i < opts.iters; ++i) {
    timer.start(stream);
    monte_carlo::simulate(variant,
                          params,
                          opts.paths,
                          d_xorwow_states,
                          d_philox_states,
                          d_payoffs,
                          d_payoff_squares,
                          stream);
    reduce_sum(d_payoffs, d_sum_payoff, cub_items, d_cub_temp, cub_temp_bytes, stream);
    reduce_sum(d_payoff_squares, d_sum_payoff_square, cub_items, d_cub_temp, cub_temp_bytes, stream);
    const float elapsed_ms = timer.stop_ms(stream);
    times_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
  }

  float sum_payoff = 0.0f;
  float sum_payoff_square = 0.0f;
  CUDA_CHECK(cudaMemcpy(&sum_payoff, d_sum_payoff, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&sum_payoff_square, d_sum_payoff_square, sizeof(float), cudaMemcpyDeviceToHost));

  const double median_us = common::median(times_us);
  const double p95_us = common::percentile(times_us, 0.95);
  const double seconds = median_us * 1.0e-6;
  const double paths_per_second = seconds > 0.0 ? static_cast<double>(opts.paths) / seconds : 0.0;
  const double sample_count = static_cast<double>(opts.paths);
  const double mean_payoff = static_cast<double>(sum_payoff) / sample_count;
  const double raw_variance =
      (static_cast<double>(sum_payoff_square) -
       static_cast<double>(sum_payoff) * static_cast<double>(sum_payoff) / sample_count) /
      (sample_count - 1.0);
  const double sample_variance = std::max(0.0, raw_variance);
  const double discount = std::exp(-opts.rate * opts.maturity);
  const double estimated_price = discount * mean_payoff;
  const double standard_error = discount * std::sqrt(sample_variance / sample_count);
  const double ci_delta = kConfidenceZ95 * standard_error;
  const double abs_error = std::abs(estimated_price - analytical_price);
  const double rel_error = abs_error / std::max(1.0, std::abs(analytical_price));

  ResultRow row;
  row.variant = monte_carlo::variant_name(variant);
  row.paths = opts.paths;
  row.steps = opts.steps;
  row.latency_median_us = median_us;
  row.latency_p95_us = p95_us;
  row.paths_per_second = paths_per_second;
  row.estimated_price = estimated_price;
  row.analytical_price = analytical_price;
  row.abs_error = abs_error;
  row.rel_error = rel_error;
  row.standard_error = standard_error;
  row.confidence_interval_95_low = estimated_price - ci_delta;
  row.confidence_interval_95_high = estimated_price + ci_delta;
  row.gpu_name = gpu_name;
  row.cuda_runtime_version = runtime_version;
  row.commit_hash = CUDA_PERF_LAB_COMMIT_HASH;
  return row;
}

void assign_baseline_ratios(std::vector<ResultRow>& rows) {
  double philox_latency = std::numeric_limits<double>::quiet_NaN();
  for (const auto& row : rows) {
    if (row.variant == "gpu_philox") {
      philox_latency = row.latency_median_us;
      break;
    }
  }

  if (!std::isfinite(philox_latency) || philox_latency <= 0.0) {
    return;
  }

  for (auto& row : rows) {
    if (row.variant == "cpu_baseline") {
      row.baseline_ratio = std::numeric_limits<double>::quiet_NaN();
    } else if (row.variant == "gpu_philox") {
      row.baseline_ratio = 1.0;
    } else {
      row.baseline_ratio = row.latency_median_us / philox_latency;
    }
  }
}

void validate_row(const ResultRow& row) {
  if (row.variant == "cpu_baseline") {
    if (!std::isfinite(row.estimated_price)) {
      throw std::runtime_error("cpu_baseline produced a non-finite analytical price");
    }
    return;
  }

  if (!std::isfinite(row.estimated_price)) {
    throw std::runtime_error(row.variant + " produced a non-finite Monte Carlo price");
  }
  if (!std::isfinite(row.standard_error)) {
    throw std::runtime_error(row.variant + " produced a non-finite standard error");
  }
  if (row.paths >= kLargePathCount && row.rel_error > kExtremeRelError) {
    std::ostringstream message;
    message << row.variant << " relative error " << row.rel_error
            << " is extreme for paths=" << row.paths;
    throw std::runtime_error(message.str());
  }

  if (std::isfinite(row.confidence_interval_95_low) &&
      std::isfinite(row.confidence_interval_95_high) &&
      (row.analytical_price < row.confidence_interval_95_low ||
       row.analytical_price > row.confidence_interval_95_high)) {
    std::cerr << "warning: analytical Black-Scholes price is outside the 95% confidence interval for "
              << row.variant << "\n";
  }
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (char ch : value) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

void print_csv_double(std::ostream& out, double value) {
  if (std::isfinite(value)) {
    out << value;
  }
}

void print_json_double_or_null(std::ostream& out, double value) {
  if (std::isfinite(value)) {
    out << value;
  } else {
    out << "null";
  }
}

void print_csv(const std::vector<ResultRow>& rows) {
  std::cout << "kernel,variant,paths,steps,latency_median_us,latency_p95_us,paths_per_second,estimated_price,analytical_price,abs_error,rel_error,standard_error,confidence_interval_95_low,confidence_interval_95_high,baseline_ratio,gpu_name,cuda_runtime_version,commit_hash\n";
  std::cout << std::setprecision(10);
  for (const auto& row : rows) {
    std::cout << kKernelName << ','
              << row.variant << ','
              << row.paths << ','
              << row.steps << ',';
    print_csv_double(std::cout, row.latency_median_us);
    std::cout << ',';
    print_csv_double(std::cout, row.latency_p95_us);
    std::cout << ',';
    print_csv_double(std::cout, row.paths_per_second);
    std::cout << ',';
    print_csv_double(std::cout, row.estimated_price);
    std::cout << ',';
    print_csv_double(std::cout, row.analytical_price);
    std::cout << ',';
    print_csv_double(std::cout, row.abs_error);
    std::cout << ',';
    print_csv_double(std::cout, row.rel_error);
    std::cout << ',';
    print_csv_double(std::cout, row.standard_error);
    std::cout << ',';
    print_csv_double(std::cout, row.confidence_interval_95_low);
    std::cout << ',';
    print_csv_double(std::cout, row.confidence_interval_95_high);
    std::cout << ',';
    print_csv_double(std::cout, row.baseline_ratio);
    std::cout << ','
              << row.gpu_name << ','
              << row.cuda_runtime_version << ','
              << row.commit_hash << '\n';
  }
}

void print_json(const std::vector<ResultRow>& rows) {
  std::cout << std::setprecision(10);
  std::cout << "[\n";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& row = rows[i];
    std::cout << "  {"
              << "\"kernel\":\"" << kKernelName << "\","
              << "\"variant\":\"" << json_escape(row.variant) << "\","
              << "\"paths\":" << row.paths << ","
              << "\"steps\":" << row.steps << ","
              << "\"latency_median_us\":";
    print_json_double_or_null(std::cout, row.latency_median_us);
    std::cout << ",\"latency_p95_us\":";
    print_json_double_or_null(std::cout, row.latency_p95_us);
    std::cout << ",\"paths_per_second\":";
    print_json_double_or_null(std::cout, row.paths_per_second);
    std::cout << ",\"estimated_price\":";
    print_json_double_or_null(std::cout, row.estimated_price);
    std::cout << ",\"analytical_price\":";
    print_json_double_or_null(std::cout, row.analytical_price);
    std::cout << ",\"abs_error\":";
    print_json_double_or_null(std::cout, row.abs_error);
    std::cout << ",\"rel_error\":";
    print_json_double_or_null(std::cout, row.rel_error);
    std::cout << ",\"standard_error\":";
    print_json_double_or_null(std::cout, row.standard_error);
    std::cout << ",\"confidence_interval_95_low\":";
    print_json_double_or_null(std::cout, row.confidence_interval_95_low);
    std::cout << ",\"confidence_interval_95_high\":";
    print_json_double_or_null(std::cout, row.confidence_interval_95_high);
    std::cout << ",\"baseline_ratio\":";
    print_json_double_or_null(std::cout, row.baseline_ratio);
    std::cout << ",\"gpu_name\":\"" << json_escape(row.gpu_name) << "\","
              << "\"cuda_runtime_version\":\"" << json_escape(row.cuda_runtime_version) << "\","
              << "\"commit_hash\":\"" << json_escape(row.commit_hash) << "\""
              << "}";
    if (i + 1 != rows.size()) {
      std::cout << ",";
    }
    std::cout << "\n";
  }
  std::cout << "]\n";
}

void print_value_or_na(double value, int width) {
  if (std::isfinite(value)) {
    std::cout << std::setw(width) << value;
  } else {
    std::cout << std::setw(width) << "NA";
  }
}

void print_human(const std::vector<ResultRow>& rows, const Options& opts) {
  std::cout << "CUDA Perf Lab: Monte Carlo Black-Scholes benchmark\n";
  std::cout << "GPU: " << rows.front().gpu_name << "\n";
  std::cout << "CUDA runtime: " << rows.front().cuda_runtime_version << "\n";
  std::cout << "Commit: " << rows.front().commit_hash << "\n";
  std::cout << "paths: " << opts.paths << " steps: " << opts.steps << "\n";
  std::cout << "spot: " << opts.spot << " strike: " << opts.strike
            << " rate: " << opts.rate << " vol: " << opts.vol
            << " maturity: " << opts.maturity << "\n";
  std::cout << "GPU timing excludes allocation, RNG state initialization, CUB temp allocation, and formatting.\n";
  std::cout << "baseline_ratio = variant median latency / gpu_philox median latency; lower is better.\n\n";

  std::cout << std::left
            << std::setw(24) << "variant"
            << std::right
            << std::setw(14) << "median_us"
            << std::setw(14) << "p95_us"
            << std::setw(18) << "paths/s"
            << std::setw(14) << "price"
            << std::setw(14) << "abs_error"
            << std::setw(14) << "std_error"
            << std::setw(14) << "base_ratio"
            << "\n";
  std::cout << std::string(126, '-') << "\n";
  std::cout << std::setprecision(6) << std::fixed;

  for (const auto& row : rows) {
    std::cout << std::left << std::setw(24) << row.variant << std::right;
    print_value_or_na(row.latency_median_us, 14);
    print_value_or_na(row.latency_p95_us, 14);
    print_value_or_na(row.paths_per_second, 18);
    print_value_or_na(row.estimated_price, 14);
    print_value_or_na(row.abs_error, 14);
    print_value_or_na(row.standard_error, 14);
    print_value_or_na(row.baseline_ratio, 14);
    std::cout << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opts = parse_cli(argc, argv);
    if (opts.help) {
      std::cout << usage(argv[0]);
      return EXIT_SUCCESS;
    }

    CUDA_CHECK(cudaSetDevice(0));

    const auto variants = variants_to_run(opts.variant);
    const std::string gpu_name = common::gpu_name(0);
    const std::string runtime_version = common::cuda_runtime_version();
    const double analytical_price = black_scholes_call_price(opts);

    bool needs_gpu = false;
    bool needs_xorwow = false;
    bool needs_philox = false;
    for (auto variant : variants) {
      needs_gpu = needs_gpu || is_gpu_variant(variant);
      needs_xorwow = needs_xorwow || uses_xorwow(variant);
      needs_philox = needs_philox || uses_philox(variant);
    }

    std::vector<ResultRow> rows;
    rows.reserve(variants.size());
    cudaStream_t stream = nullptr;

    if (!needs_gpu) {
      ResultRow row = benchmark_cpu_baseline(opts, analytical_price, gpu_name, runtime_version);
      validate_row(row);
      rows.push_back(row);
    } else {
      DeviceBuffer<float> d_payoffs(opts.paths);
      DeviceBuffer<float> d_payoff_squares(opts.paths);
      DeviceBuffer<float> d_sum_payoff(1);
      DeviceBuffer<float> d_sum_payoff_square(1);
      DeviceBuffer<curandState> d_xorwow_states(needs_xorwow ? opts.paths : 0);
      DeviceBuffer<curandStatePhilox4_32_10_t> d_philox_states(needs_philox ? opts.paths : 0);

      const int cub_items = checked_cub_item_count(opts.paths);
      const std::size_t cub_payoff_bytes = cub_temp_storage_bytes(d_payoffs.get(), d_sum_payoff.get(), cub_items);
      const std::size_t cub_square_bytes =
          cub_temp_storage_bytes(d_payoff_squares.get(), d_sum_payoff_square.get(), cub_items);
      DeviceBytes d_cub_temp(std::max(cub_payoff_bytes, cub_square_bytes));

      for (auto variant : variants) {
        if (variant == monte_carlo::Variant::kCpuBaseline) {
          ResultRow row = benchmark_cpu_baseline(opts, analytical_price, gpu_name, runtime_version);
          validate_row(row);
          rows.push_back(row);
          continue;
        }

        if (uses_xorwow(variant)) {
          monte_carlo::initialize_xorwow(d_xorwow_states.get(), opts.paths, opts.seed, stream);
          CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        if (uses_philox(variant)) {
          monte_carlo::initialize_philox(d_philox_states.get(), opts.paths, opts.seed, stream);
          CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        ResultRow row = benchmark_gpu_variant(variant,
                                              opts,
                                              analytical_price,
                                              gpu_name,
                                              runtime_version,
                                              d_xorwow_states.get(),
                                              d_philox_states.get(),
                                              d_payoffs.get(),
                                              d_payoff_squares.get(),
                                              d_sum_payoff.get(),
                                              d_sum_payoff_square.get(),
                                              cub_items,
                                              d_cub_temp.get(),
                                              d_cub_temp.bytes(),
                                              stream);
        validate_row(row);
        rows.push_back(row);
      }
    }

    assign_baseline_ratios(rows);

    if (opts.csv) {
      print_csv(rows);
    } else if (opts.json) {
      print_json(rows);
    } else {
      print_human(rows, opts);
    }

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
