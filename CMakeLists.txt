cmake_minimum_required(VERSION 3.18)
project(dyn_quant LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -O3 --expt-relaxed-constexpr")

# Collect kernel sources
set(KERNEL_SOURCES
    src/int4_gemv.cu
    src/split_gemv.cu
    src/reference_gemv.cu
)

add_library(dyn_quant_kernels STATIC ${KERNEL_SOURCES})
target_include_directories(dyn_quant_kernels PUBLIC src)
set_target_properties(dyn_quant_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(dyn_quant_kernels PUBLIC CUDA::cudart)

# Correctness test
add_executable(test_correctness tests/test_correctness.cu)
target_include_directories(test_correctness PRIVATE src)
target_link_libraries(test_correctness PRIVATE dyn_quant_kernels CUDA::cudart)
set_target_properties(test_correctness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Latency benchmark
add_executable(bench_latency benchmarks/bench_latency.cu)
target_include_directories(bench_latency PRIVATE src)
target_link_libraries(bench_latency PRIVATE dyn_quant_kernels CUDA::cudart)
set_target_properties(bench_latency PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Accuracy sweep benchmark
add_executable(bench_accuracy benchmarks/bench_accuracy.cu)
target_include_directories(bench_accuracy PRIVATE src)
target_link_libraries(bench_accuracy PRIVATE dyn_quant_kernels CUDA::cudart)
set_target_properties(bench_accuracy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

enable_testing()
add_test(NAME correctness COMMAND test_correctness)
