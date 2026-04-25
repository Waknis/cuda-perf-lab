#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace common {

inline void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
  if (status == cudaSuccess) {
    return;
  }

  std::ostringstream message;
  message << "CUDA error at " << file << ":" << line << " while calling " << expr
          << ": " << cudaGetErrorString(status);
  throw std::runtime_error(message.str());
}

}  // namespace common

#define CUDA_CHECK(expr) ::common::cuda_check((expr), #expr, __FILE__, __LINE__)
