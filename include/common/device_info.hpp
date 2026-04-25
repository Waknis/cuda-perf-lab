#pragma once

#include "common/cuda_check.hpp"

#include <cuda_runtime.h>

#include <sstream>
#include <string>

namespace common {

inline std::string gpu_name(int device = 0) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return std::string(prop.name);
}

inline std::string cuda_runtime_version() {
  int version = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&version));

  const int major = version / 1000;
  const int minor = (version % 1000) / 10;

  std::ostringstream out;
  out << major << "." << minor;
  return out.str();
}

}  // namespace common
