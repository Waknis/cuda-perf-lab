#include "common/cuda_check.hpp"
#include "common/device_info.hpp"
#include "common/stats.hpp"
#include "common/timer.hpp"
#include "softmax/softmax_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CUDA_PERF_LAB_COMMIT_HASH
#define CUDA_PERF_LAB_COMMIT_HASH "unknown"
#endif

namespace {

constexpr const char* kKernelName = "softmax";
constexpr double kAbsTolerance = 1.0e-4;
constexpr double kRelTolerance = 1.0e-2;
constexpr double kRowSumTolerance = 1.0e-4;

struct Options {
  std::string variant = "all";
  std::size_t rows = 4096;
  std::size_t cols = 1024;
  int iters = 20;
  int warmup = 5;
  bool csv = false;
  bool json = false;
  std::uint64_t seed = 1234;
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

struct ResultRow {
  std::string variant;
  std::size_t rows = 0;
  std::size_t cols = 0;
  double latency_median_us = 0.0;
  double latency_p95_us = 0.0;
  double bandwidth_gb_s = 0.0;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  double max_row_sum_error = 0.0;
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
      << "  --variant <name>   naive, stable_two_pass, block_reduce, warp_small_row, all\n"
      << "  --rows <count>     Matrix rows\n"
      << "  --cols <count>     Matrix columns\n"
      << "  --iters <count>    Timed iterations\n"
      << "  --warmup <count>   Warmup iterations\n"
      << "  --csv              Print CSV only\n"
      << "  --json             Print JSON only\n"
      << "  --seed <value>     Deterministic input seed\n"
      << "  --help             Show this message\n";
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
    } else if (arg == "--rows") {
      opts.rows = static_cast<std::size_t>(std::stoull(get_value()));
    } else if (arg == "--cols") {
      opts.cols = static_cast<std::size_t>(std::stoull(get_value()));
    } else if (arg == "--iters") {
      opts.iters = std::stoi(get_value());
    } else if (arg == "--warmup") {
      opts.warmup = std::stoi(get_value());
    } else if (arg == "--csv") {
      opts.csv = true;
    } else if (arg == "--json") {
      opts.json = true;
    } else if (arg == "--seed") {
      opts.seed = static_cast<std::uint64_t>(std::stoull(get_value()));
    } else if (arg == "--help" || arg == "-h") {
      opts.help = true;
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }

  if (opts.csv && opts.json) {
    throw std::runtime_error("--csv and --json are mutually exclusive");
  }
  if (opts.rows == 0 || opts.cols == 0) {
    throw std::runtime_error("--rows and --cols must be greater than zero");
  }
  if (opts.iters <= 0) {
    throw std::runtime_error("--iters must be greater than zero");
  }
  if (opts.warmup < 0) {
    throw std::runtime_error("--warmup must be non-negative");
  }

  return opts;
}

std::vector<float> make_input(std::size_t elements, std::uint64_t seed) {
  std::vector<float> values(elements);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

  for (float& value : values) {
    value = dist(rng);
  }

  return values;
}

std::vector<double> cpu_softmax_reference(const std::vector<float>& input, std::size_t rows, std::size_t cols) {
  std::vector<double> reference(rows * cols);

  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t base = row * cols;
    double row_max = -std::numeric_limits<double>::infinity();
    for (std::size_t col = 0; col < cols; ++col) {
      row_max = std::max(row_max, static_cast<double>(input[base + col]));
    }

    double sum = 0.0;
    for (std::size_t col = 0; col < cols; ++col) {
      sum += std::exp(static_cast<double>(input[base + col]) - row_max);
    }

    for (std::size_t col = 0; col < cols; ++col) {
      reference[base + col] = std::exp(static_cast<double>(input[base + col]) - row_max) / sum;
    }
  }

  return reference;
}

std::vector<softmax::Variant> variants_to_run(const std::string& requested) {
  if (requested == "all") {
    return {
        softmax::Variant::kNaive,
        softmax::Variant::kStableTwoPass,
        softmax::Variant::kBlockReduce,
        softmax::Variant::kWarpSmallRow,
    };
  }

  softmax::Variant variant{};
  if (!softmax::parse_variant(requested, &variant)) {
    std::ostringstream message;
    message << "unsupported variant '" << requested << "'. Supported variants: all";
    for (const auto& name : softmax::variant_names()) {
      message << ", " << name;
    }
    throw std::runtime_error(message.str());
  }
  return {variant};
}

double estimated_bytes_moved(softmax::Variant variant, std::size_t rows, std::size_t cols) {
  const double elements = static_cast<double>(rows) * static_cast<double>(cols);
  const double bytes_per_matrix = elements * sizeof(float);
  switch (variant) {
    case softmax::Variant::kNaive:
      return 3.0 * bytes_per_matrix;
    case softmax::Variant::kStableTwoPass:
    case softmax::Variant::kBlockReduce:
    case softmax::Variant::kWarpSmallRow:
      return 4.0 * bytes_per_matrix;
  }
  return 0.0;
}

void compute_errors(const std::vector<float>& output,
                    const std::vector<double>& reference,
                    std::size_t rows,
                    std::size_t cols,
                    double* max_abs_error,
                    double* max_rel_error,
                    double* max_row_sum_error) {
  double abs_error = 0.0;
  double rel_error = 0.0;
  double row_sum_error = 0.0;

  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t base = row * cols;
    double row_sum = 0.0;
    for (std::size_t col = 0; col < cols; ++col) {
      const double actual = static_cast<double>(output[base + col]);
      const double expected = reference[base + col];
      const double diff = std::abs(actual - expected);
      abs_error = std::max(abs_error, diff);
      rel_error = std::max(rel_error, diff / std::max(1.0e-12, std::abs(expected)));
      row_sum += actual;
    }
    row_sum_error = std::max(row_sum_error, std::abs(row_sum - 1.0));
  }

  *max_abs_error = abs_error;
  *max_rel_error = rel_error;
  *max_row_sum_error = row_sum_error;
}

ResultRow benchmark_variant(softmax::Variant variant,
                            const Options& opts,
                            const std::vector<double>& reference,
                            const std::string& gpu_name,
                            const std::string& runtime_version,
                            const float* d_input,
                            float* d_output,
                            cudaStream_t stream) {
  for (int i = 0; i < opts.warmup; ++i) {
    softmax::launch(variant, d_input, d_output, opts.rows, opts.cols, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> times_us;
  times_us.reserve(static_cast<std::size_t>(opts.iters));
  common::GpuTimer timer;

  for (int i = 0; i < opts.iters; ++i) {
    timer.start(stream);
    softmax::launch(variant, d_input, d_output, opts.rows, opts.cols, stream);
    const float elapsed_ms = timer.stop_ms(stream);
    times_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
  }

  std::vector<float> output(opts.rows * opts.cols);
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  double max_row_sum_error = 0.0;
  compute_errors(output, reference, opts.rows, opts.cols, &max_abs_error, &max_rel_error, &max_row_sum_error);

  const double median_us = common::median(times_us);
  const double p95_us = common::percentile(times_us, 0.95);
  const double seconds = median_us * 1.0e-6;
  const double bandwidth_gb_s = seconds > 0.0 ? estimated_bytes_moved(variant, opts.rows, opts.cols) / seconds / 1.0e9 : 0.0;

  ResultRow row;
  row.variant = softmax::variant_name(variant);
  row.rows = opts.rows;
  row.cols = opts.cols;
  row.latency_median_us = median_us;
  row.latency_p95_us = p95_us;
  row.bandwidth_gb_s = bandwidth_gb_s;
  row.max_abs_error = max_abs_error;
  row.max_rel_error = max_rel_error;
  row.max_row_sum_error = max_row_sum_error;
  row.gpu_name = gpu_name;
  row.cuda_runtime_version = runtime_version;
  row.commit_hash = CUDA_PERF_LAB_COMMIT_HASH;
  return row;
}

void assign_baseline_ratios(std::vector<ResultRow>& rows) {
  double baseline_latency = std::numeric_limits<double>::quiet_NaN();
  for (const auto& row : rows) {
    if (row.variant == "stable_two_pass") {
      baseline_latency = row.latency_median_us;
      break;
    }
  }

  if (!std::isfinite(baseline_latency) || baseline_latency <= 0.0) {
    return;
  }

  for (auto& row : rows) {
    row.baseline_ratio = row.latency_median_us / baseline_latency;
  }
}

void check_row(const ResultRow& row) {
  if (row.max_abs_error <= kAbsTolerance &&
      row.max_rel_error <= kRelTolerance &&
      row.max_row_sum_error <= kRowSumTolerance) {
    return;
  }

  std::ostringstream message;
  message << "correctness failure for " << row.variant
          << " rows=" << row.rows
          << " cols=" << row.cols
          << " max_abs_error=" << row.max_abs_error
          << " max_rel_error=" << row.max_rel_error
          << " max_row_sum_error=" << row.max_row_sum_error
          << " tolerances(abs=" << kAbsTolerance
          << ", rel=" << kRelTolerance
          << ", row_sum=" << kRowSumTolerance << ")";
  throw std::runtime_error(message.str());
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

void print_double_or_null(std::ostream& out, double value) {
  if (std::isfinite(value)) {
    out << value;
  } else {
    out << "null";
  }
}

void print_csv(const std::vector<ResultRow>& rows) {
  std::cout << "kernel,variant,rows,cols,latency_median_us,latency_p95_us,bandwidth_gb_s,max_abs_error,max_rel_error,max_row_sum_error,baseline_ratio,gpu_name,cuda_runtime_version,commit_hash\n";
  std::cout << std::setprecision(10);
  for (const auto& row : rows) {
    std::cout << kKernelName << ','
              << row.variant << ','
              << row.rows << ','
              << row.cols << ','
              << row.latency_median_us << ','
              << row.latency_p95_us << ','
              << row.bandwidth_gb_s << ','
              << row.max_abs_error << ','
              << row.max_rel_error << ','
              << row.max_row_sum_error << ',';
    if (std::isfinite(row.baseline_ratio)) {
      std::cout << row.baseline_ratio;
    } else {
      std::cout << "nan";
    }
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
              << "\"rows\":" << row.rows << ","
              << "\"cols\":" << row.cols << ","
              << "\"latency_median_us\":" << row.latency_median_us << ","
              << "\"latency_p95_us\":" << row.latency_p95_us << ","
              << "\"bandwidth_gb_s\":" << row.bandwidth_gb_s << ","
              << "\"max_abs_error\":" << row.max_abs_error << ","
              << "\"max_rel_error\":" << row.max_rel_error << ","
              << "\"max_row_sum_error\":" << row.max_row_sum_error << ","
              << "\"baseline_ratio\":";
    print_double_or_null(std::cout, row.baseline_ratio);
    std::cout << ","
              << "\"gpu_name\":\"" << json_escape(row.gpu_name) << "\","
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

void print_human(const std::vector<ResultRow>& rows) {
  std::cout << "CUDA Perf Lab: softmax benchmark\n";
  std::cout << "GPU: " << rows.front().gpu_name << "\n";
  std::cout << "CUDA runtime: " << rows.front().cuda_runtime_version << "\n";
  std::cout << "Commit: " << rows.front().commit_hash << "\n";
  std::cout << "shape: " << rows.front().rows << " x " << rows.front().cols << "\n";
  std::cout << "bandwidth uses estimated bytes moved for each variant; no GFLOP/s estimate is reported in v1.\n";
  std::cout << "baseline_ratio = variant median latency / stable_two_pass median latency; lower is better.\n\n";

  std::cout << std::left
            << std::setw(18) << "variant"
            << std::right
            << std::setw(16) << "median_us"
            << std::setw(16) << "p95_us"
            << std::setw(16) << "GB/s"
            << std::setw(16) << "max_abs"
            << std::setw(16) << "max_rel"
            << std::setw(16) << "row_sum"
            << std::setw(16) << "base_ratio"
            << "\n";

  std::cout << std::string(130, '-') << "\n";
  std::cout << std::setprecision(6) << std::fixed;

  for (const auto& row : rows) {
    std::cout << std::left << std::setw(18) << row.variant
              << std::right
              << std::setw(16) << row.latency_median_us
              << std::setw(16) << row.latency_p95_us
              << std::setw(16) << row.bandwidth_gb_s
              << std::setw(16) << row.max_abs_error
              << std::setw(16) << row.max_rel_error
              << std::setw(16) << row.max_row_sum_error;
    if (std::isfinite(row.baseline_ratio)) {
      std::cout << std::setw(16) << row.baseline_ratio;
    } else {
      std::cout << std::setw(16) << "NA";
    }
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

    if (opts.rows > std::numeric_limits<std::size_t>::max() / opts.cols) {
      throw std::runtime_error("rows * cols overflowed size_t");
    }
    const std::size_t elements = opts.rows * opts.cols;

    const auto variants = variants_to_run(opts.variant);
    const std::string gpu_name = common::gpu_name(0);
    const std::string runtime_version = common::cuda_runtime_version();

    const std::vector<float> input = make_input(elements, opts.seed);
    const std::vector<double> reference = cpu_softmax_reference(input, opts.rows, opts.cols);

    DeviceBuffer<float> d_input(elements);
    DeviceBuffer<float> d_output(elements);
    CUDA_CHECK(cudaMemcpy(d_input.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<ResultRow> rows;
    rows.reserve(variants.size());
    cudaStream_t stream = nullptr;

    for (auto variant : variants) {
      ResultRow row = benchmark_variant(variant,
                                        opts,
                                        reference,
                                        gpu_name,
                                        runtime_version,
                                        d_input.get(),
                                        d_output.get(),
                                        stream);
      check_row(row);
      rows.push_back(row);
    }

    assign_baseline_ratios(rows);

    if (opts.csv) {
      print_csv(rows);
    } else if (opts.json) {
      print_json(rows);
    } else {
      print_human(rows);
    }

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
