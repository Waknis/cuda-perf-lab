#pragma once

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace common {

struct CliOptions {
  std::string variant = "all";
  std::size_t n = 1 << 20;
  int iters = 20;
  int warmup = 5;
  bool csv = false;
  bool json = false;
  std::uint64_t seed = 1234;
  bool help = false;
};

inline std::string usage(const char* program) {
  std::ostringstream out;
  out << "Usage: " << program << " [options]\n"
      << "\n"
      << "Options:\n"
      << "  --variant <name>   naive_global, shared_interleaved, shared_sequential,\n"
      << "                     warp_shuffle, vectorized_float4, cub_baseline, all\n"
      << "  --n <count>        Number of float32 values to reduce\n"
      << "  --iters <count>    Timed iterations\n"
      << "  --warmup <count>   Warmup iterations\n"
      << "  --csv              Print CSV only\n"
      << "  --json             Print JSON only\n"
      << "  --seed <value>     Deterministic input seed\n"
      << "  --help             Show this message\n";
  return out.str();
}

inline std::string next_value(int& i, int argc, char** argv, const std::string& flag) {
  if (i + 1 >= argc) {
    throw std::runtime_error("missing value for " + flag);
  }
  ++i;
  return argv[i];
}

inline CliOptions parse_cli(int argc, char** argv) {
  CliOptions opts;

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
    } else if (arg == "--n") {
      opts.n = static_cast<std::size_t>(std::stoull(get_value()));
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
  if (opts.n == 0) {
    throw std::runtime_error("--n must be greater than zero");
  }
  if (opts.iters <= 0) {
    throw std::runtime_error("--iters must be greater than zero");
  }
  if (opts.warmup < 0) {
    throw std::runtime_error("--warmup must be non-negative");
  }

  return opts;
}

}  // namespace common
