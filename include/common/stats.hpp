#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace common {

inline double median(std::vector<double> values) {
  if (values.empty()) {
    throw std::runtime_error("median requires at least one sample");
  }

  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  if (values.size() % 2 == 0) {
    return 0.5 * (values[mid - 1] + values[mid]);
  }
  return values[mid];
}

inline double percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    throw std::runtime_error("percentile requires at least one sample");
  }

  std::sort(values.begin(), values.end());
  const double clamped = std::max(0.0, std::min(1.0, p));
  const auto raw_index = static_cast<std::size_t>(std::ceil(clamped * values.size()));
  const std::size_t index = raw_index == 0 ? 0 : raw_index - 1;
  return values[std::min(index, values.size() - 1)];
}

}  // namespace common
