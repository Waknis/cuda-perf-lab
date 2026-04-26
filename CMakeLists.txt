cmake_minimum_required(VERSION 3.24)

project(cuda_perf_lab VERSION 0.1.0 LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures to build for" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

execute_process(
  COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  RESULT_VARIABLE GIT_RESULT
  OUTPUT_VARIABLE GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

if(NOT GIT_RESULT EQUAL 0 OR GIT_HASH STREQUAL "")
  set(GIT_HASH "unknown")
endif()

add_executable(reduction_bench
  benchmarks/reduction_bench.cu
  kernels/reduction/reduction_kernels.cu
)

add_executable(softmax_bench
  benchmarks/softmax_bench.cu
  kernels/softmax/softmax_kernels.cu
)

add_executable(monte_carlo_bench
  benchmarks/monte_carlo_bench.cu
  kernels/monte_carlo/monte_carlo_kernels.cu
)

target_include_directories(reduction_bench
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
    "${CUDAToolkit_INCLUDE_DIRS}"
)

target_include_directories(softmax_bench
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
    "${CUDAToolkit_INCLUDE_DIRS}"
)

target_include_directories(monte_carlo_bench
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
    "${CUDAToolkit_INCLUDE_DIRS}"
)

if(EXISTS "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  target_include_directories(reduction_bench PRIVATE "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  target_include_directories(softmax_bench PRIVATE "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  target_include_directories(monte_carlo_bench PRIVATE "${CUDAToolkit_INCLUDE_DIRS}/cccl")
endif()

target_compile_definitions(reduction_bench
  PRIVATE
    CUDA_PERF_LAB_COMMIT_HASH="${GIT_HASH}"
)

target_compile_definitions(softmax_bench
  PRIVATE
    CUDA_PERF_LAB_COMMIT_HASH="${GIT_HASH}"
)

target_compile_definitions(monte_carlo_bench
  PRIVATE
    CUDA_PERF_LAB_COMMIT_HASH="${GIT_HASH}"
)

target_compile_options(reduction_bench
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_compile_options(softmax_bench
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_compile_options(monte_carlo_bench
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(reduction_bench PRIVATE CUDA::cudart)
target_link_libraries(softmax_bench PRIVATE CUDA::cudart)
target_link_libraries(monte_carlo_bench PRIVATE CUDA::cudart CUDA::curand)

set_target_properties(reduction_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
set_target_properties(softmax_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
set_target_properties(monte_carlo_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
