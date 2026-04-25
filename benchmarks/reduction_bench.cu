#include "common/cli.hpp"
#include "common/cuda_check.hpp"
#include "common/device_info.hpp"
#include "common/stats.hpp"
#include "common/timer.hpp"
#include "reduction/reduction_kernels.cuh"

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

constexpr const char* kKernelName = "reduction";
constexpr std::size_t kReductionBlockSize = 256;
constexpr std::size_t kBlockReductionFanIn = kReductionBlockSize * 2;
constexpr std::size_t kMaxVectorizedBlocks = 4096;

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
  std::size_t n = 0;
  double latency_median_us = 0.0;
  double latency_p95_us = 0.0;
  double bandwidth_gb_s = 0.0;
  double abs_error = 0.0;
  double rel_error = 0.0;
  double baseline_ratio = std::numeric_limits<double>::quiet_NaN();
  std::string gpu_name;
  std::string cuda_runtime_version;
  std::string commit_hash;
};

std::vector<float> make_input(std::size_t n, std::uint64_t seed) {
  std::vector<float> values(n);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (float& value : values) {
    value = dist(rng);
  }

  return values;
}

double cpu_reference_sum(const std::vector<float>& values) {
  double sum = 0.0;
  for (float value : values) {
    sum += static_cast<double>(value);
  }
  return sum;
}

std::size_t div_up(std::size_t x, std::size_t y) {
  return (x + y - 1) / y;
}

double staged_reduction_global_bytes(std::size_t n, std::size_t fan_in) {
  double bytes = 0.0;
  std::size_t current_n = n;

  while (current_n > 1) {
    const std::size_t partials = div_up(current_n, fan_in);
    bytes += static_cast<double>(current_n + partials) * sizeof(float);
    current_n = partials;
  }

  // The custom staged kernels copy the final device partial into d_output.
  bytes += 2.0 * sizeof(float);
  return bytes;
}

double estimated_global_bytes_moved(reduction::Variant variant, std::size_t n) {
  switch (variant) {
    case reduction::Variant::kNaiveGlobal:
      return staged_reduction_global_bytes(n, 2);
    case reduction::Variant::kSharedInterleaved:
    case reduction::Variant::kSharedSequential:
    case reduction::Variant::kWarpShuffle:
      return staged_reduction_global_bytes(n, kBlockReductionFanIn);
    case reduction::Variant::kVectorizedFloat4: {
      if (n < 4) {
        return staged_reduction_global_bytes(n, kBlockReductionFanIn);
      }

      const std::size_t vector_chunks = n / 4;
      const std::size_t blocks_from_data = div_up(vector_chunks, kReductionBlockSize);
      const std::size_t first_pass_partials = std::max<std::size_t>(
          1, std::min(kMaxVectorizedBlocks, blocks_from_data));
      const double first_pass_bytes = static_cast<double>(n + first_pass_partials) * sizeof(float);
      return first_pass_bytes + staged_reduction_global_bytes(first_pass_partials, kBlockReductionFanIn);
    }
    case reduction::Variant::kCubBaseline:
      // CUB's internal traffic is implementation-dependent. This is the minimum
      // global traffic needed to read the input and write one output value.
      return static_cast<double>(n + 1) * sizeof(float);
  }

  return static_cast<double>(n) * sizeof(float);
}

std::vector<reduction::Variant> variants_to_run(const std::string& requested) {
  if (requested == "all") {
    return {
        reduction::Variant::kNaiveGlobal,
        reduction::Variant::kSharedInterleaved,
        reduction::Variant::kSharedSequential,
        reduction::Variant::kWarpShuffle,
        reduction::Variant::kVectorizedFloat4,
        reduction::Variant::kCubBaseline,
    };
  }

  reduction::Variant variant{};
  if (!reduction::parse_variant(requested, &variant)) {
    std::ostringstream message;
    message << "unsupported variant '" << requested << "'. Supported variants: all";
    for (const auto& name : reduction::variant_names()) {
      message << ", " << name;
    }
    throw std::runtime_error(message.str());
  }
  return {variant};
}

ResultRow benchmark_variant(reduction::Variant variant,
                            std::size_t n,
                            int iters,
                            int warmup,
                            double reference,
                            const std::string& gpu_name,
                            const std::string& runtime_version,
                            const float* d_input,
                            float* d_output,
                            float* d_tmp1,
                            float* d_tmp2,
                            std::size_t tmp_elements,
                            void* d_cub_temp,
                            std::size_t cub_temp_bytes,
                            cudaStream_t stream) {
  for (int i = 0; i < warmup; ++i) {
    reduction::reduce(variant,
                      d_input,
                      n,
                      d_output,
                      d_tmp1,
                      d_tmp2,
                      tmp_elements,
                      d_cub_temp,
                      cub_temp_bytes,
                      stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> times_us;
  times_us.reserve(static_cast<std::size_t>(iters));
  common::GpuTimer timer;

  for (int i = 0; i < iters; ++i) {
    timer.start(stream);
    reduction::reduce(variant,
                      d_input,
                      n,
                      d_output,
                      d_tmp1,
                      d_tmp2,
                      tmp_elements,
                      d_cub_temp,
                      cub_temp_bytes,
                      stream);
    const float elapsed_ms = timer.stop_ms(stream);
    times_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
  }

  float gpu_sum = 0.0f;
  CUDA_CHECK(cudaMemcpy(&gpu_sum, d_output, sizeof(float), cudaMemcpyDeviceToHost));

  const double median_us = common::median(times_us);
  const double p95_us = common::percentile(times_us, 0.95);
  const double estimated_bytes = estimated_global_bytes_moved(variant, n);
  const double seconds = median_us * 1.0e-6;
  const double bandwidth_gb_s = seconds > 0.0 ? (estimated_bytes / seconds) / 1.0e9 : 0.0;
  const double abs_error = std::abs(static_cast<double>(gpu_sum) - reference);
  const double rel_error = abs_error / std::max(1.0, std::abs(reference));

  ResultRow row;
  row.variant = reduction::variant_name(variant);
  row.n = n;
  row.latency_median_us = median_us;
  row.latency_p95_us = p95_us;
  row.bandwidth_gb_s = bandwidth_gb_s;
  row.abs_error = abs_error;
  row.rel_error = rel_error;
  row.gpu_name = gpu_name;
  row.cuda_runtime_version = runtime_version;
  row.commit_hash = CUDA_PERF_LAB_COMMIT_HASH;
  return row;
}

void assign_baseline_ratios(std::vector<ResultRow>& rows) {
  double cub_latency = std::numeric_limits<double>::quiet_NaN();
  for (const auto& row : rows) {
    if (row.variant == "cub_baseline") {
      cub_latency = row.latency_median_us;
      break;
    }
  }

  if (!std::isfinite(cub_latency) || cub_latency <= 0.0) {
    return;
  }

  for (auto& row : rows) {
    row.baseline_ratio = row.latency_median_us / cub_latency;
  }
}

double tolerance_for(double reference) {
  return std::max(1.0e-2, 1.0e-4 * std::abs(reference));
}

void check_row(const ResultRow& row, double tolerance, double reference) {
  if (row.abs_error <= tolerance) {
    return;
  }

  std::ostringstream message;
  message << "correctness failure for " << row.variant << " at n=" << row.n
          << ": abs_error=" << row.abs_error << " rel_error=" << row.rel_error
          << " tolerance=" << tolerance << " cpu_reference=" << reference;
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
  std::cout << "kernel,variant,n,latency_median_us,latency_p95_us,bandwidth_gb_s,abs_error,rel_error,baseline_ratio,gpu_name,cuda_runtime_version,commit_hash\n";
  std::cout << std::setprecision(10);
  for (const auto& row : rows) {
    std::cout << kKernelName << ','
              << row.variant << ','
              << row.n << ','
              << row.latency_median_us << ','
              << row.latency_p95_us << ','
              << row.bandwidth_gb_s << ','
              << row.abs_error << ','
              << row.rel_error << ',';
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
              << "\"n\":" << row.n << ","
              << "\"latency_median_us\":" << row.latency_median_us << ","
              << "\"latency_p95_us\":" << row.latency_p95_us << ","
              << "\"bandwidth_gb_s\":" << row.bandwidth_gb_s << ","
              << "\"abs_error\":" << row.abs_error << ","
              << "\"rel_error\":" << row.rel_error << ","
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

void print_human(const std::vector<ResultRow>& rows, double reference, double tolerance) {
  std::cout << "CUDA Perf Lab: reduction benchmark\n";
  std::cout << "GPU: " << rows.front().gpu_name << "\n";
  std::cout << "CUDA runtime: " << rows.front().cuda_runtime_version << "\n";
  std::cout << "Commit: " << rows.front().commit_hash << "\n";
  std::cout << "n: " << rows.front().n << "\n";
  std::cout << "CPU double reference: " << std::setprecision(12) << reference << "\n";
  std::cout << "Correctness tolerance: " << tolerance << "\n";
  std::cout << "bandwidth uses estimated global bytes moved for each variant.\n";
  std::cout << "baseline_ratio = variant median latency / CUB median latency; lower is better.\n\n";

  std::cout << std::left
            << std::setw(22) << "variant"
            << std::right
            << std::setw(16) << "median_us"
            << std::setw(16) << "p95_us"
            << std::setw(16) << "GB/s"
            << std::setw(16) << "abs_error"
            << std::setw(16) << "rel_error"
            << std::setw(16) << "cub_ratio"
            << "\n";

  std::cout << std::string(118, '-') << "\n";
  std::cout << std::setprecision(6) << std::fixed;

  for (const auto& row : rows) {
    std::cout << std::left << std::setw(22) << row.variant
              << std::right
              << std::setw(16) << row.latency_median_us
              << std::setw(16) << row.latency_p95_us
              << std::setw(16) << row.bandwidth_gb_s
              << std::setw(16) << row.abs_error
              << std::setw(16) << row.rel_error;
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
    const common::CliOptions opts = common::parse_cli(argc, argv);
    if (opts.help) {
      std::cout << common::usage(argv[0]);
      return EXIT_SUCCESS;
    }

    CUDA_CHECK(cudaSetDevice(0));

    const auto variants = variants_to_run(opts.variant);
    const std::string name = common::gpu_name(0);
    const std::string runtime_version = common::cuda_runtime_version();

    const std::vector<float> input = make_input(opts.n, opts.seed);
    const double reference = cpu_reference_sum(input);
    const double tolerance = tolerance_for(reference);

    DeviceBuffer<float> d_input(opts.n);
    DeviceBuffer<float> d_output(1);
    const std::size_t scratch_elements = reduction::required_scratch_elements(opts.n);
    DeviceBuffer<float> d_tmp1(scratch_elements);
    DeviceBuffer<float> d_tmp2(scratch_elements);

    CUDA_CHECK(cudaMemcpy(d_input.get(), input.data(), opts.n * sizeof(float), cudaMemcpyHostToDevice));

    const std::size_t cub_bytes = reduction::cub_temp_storage_bytes(d_input.get(), d_output.get(), opts.n);
    DeviceBytes d_cub_temp(cub_bytes);

    std::vector<ResultRow> rows;
    rows.reserve(variants.size());
    cudaStream_t stream = nullptr;

    for (auto variant : variants) {
      ResultRow row = benchmark_variant(variant,
                                        opts.n,
                                        opts.iters,
                                        opts.warmup,
                                        reference,
                                        name,
                                        runtime_version,
                                        d_input.get(),
                                        d_output.get(),
                                        d_tmp1.get(),
                                        d_tmp2.get(),
                                        d_tmp1.count(),
                                        d_cub_temp.get(),
                                        d_cub_temp.bytes(),
                                        stream);
      check_row(row, tolerance, reference);
      rows.push_back(row);
    }

    assign_baseline_ratios(rows);

    if (opts.csv) {
      print_csv(rows);
    } else if (opts.json) {
      print_json(rows);
    } else {
      print_human(rows, reference, tolerance);
    }

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
