#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "quant_format.cuh"
#include "int4_gemv.cuh"
#include "split_gemv.cuh"
#include "reference_gemv.cuh"

static constexpr int WARMUP_ITERS = 10;
static constexpr int BENCH_ITERS  = 100;

// Returns average latency in microseconds over BENCH_ITERS iterations.
template<typename Fn>
float bench_us(Fn&& fn) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERS; ++i) fn();
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; ++i) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms * 1000.f / BENCH_ITERS;  // us per iteration
}

struct Dim { const char* name; int d_out; int d_in; };

void bench_dim(const Dim& dim) {
    const int d_out = dim.d_out;
    const int d_in  = dim.d_in;

    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.f, 1.f);

    std::vector<__half> h_W(static_cast<size_t>(d_out) * d_in);
    std::vector<__half> h_x(d_in);
    for (auto& v : h_W) v = __float2half(dist(rng) * 0.02f);
    for (auto& v : h_x) v = __float2half(dist(rng));

    __half *d_W, *d_x, *d_y;
    cudaMalloc(&d_W, static_cast<size_t>(d_out) * d_in * sizeof(__half));
    cudaMalloc(&d_x, d_in  * sizeof(__half));
    cudaMalloc(&d_y, d_out * sizeof(__half));
    cudaMemcpy(d_W, h_W.data(), static_cast<size_t>(d_out) * d_in * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), d_in * sizeof(__half), cudaMemcpyHostToDevice);

    QuantizedWeights qW = quantize_weights_host(h_W.data(), d_out, d_in);

    // Memory bandwidth: FP16 reference reads d_out*d_in + d_in halfs
    double weight_bytes_fp16 = static_cast<double>(d_out) * d_in * sizeof(__half);
    double weight_bytes_int4 = static_cast<double>(d_out) * d_in / 2.0;  // 4 bits per element
    double act_bytes         = d_in * sizeof(__half);

    float us_ref   = bench_us([&]{ launch_reference_gemv(d_W, d_x, d_y, d_out, d_in); });
    float us_int4  = bench_us([&]{ launch_int4_gemv(qW, d_x, d_y, d_out, d_in); });
    float us_split = bench_us([&]{ launch_split_gemv(qW, d_x, d_y, d_out, d_in, 3.0f); });

    double bw_ref   = (weight_bytes_fp16 + act_bytes) / (us_ref   * 1e-6) / 1e9;
    double bw_int4  = (weight_bytes_int4 + act_bytes) / (us_int4  * 1e-6) / 1e9;
    double bw_split = (weight_bytes_int4 + act_bytes) / (us_split * 1e-6) / 1e9;

    printf("%-25s | ref=%7.1f us  int4=%7.1f us (%.2fx)  split=%7.1f us (%.2fx vs fp16, %.2fx vs int4)\n"
           "                          | bw: ref=%.0f GB/s  int4=%.0f GB/s  split=%.0f GB/s\n",
           dim.name,
           us_ref, us_int4, us_ref / us_int4,
           us_split, us_ref / us_split, us_int4 / us_split,
           bw_ref, bw_int4, bw_split);

    free_quantized_weights(qW);
    cudaFree(d_W);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (sm_%d%d, %.0f GB/s peak BW)\n\n",
           prop.name, prop.major, prop.minor,
           2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9);

    printf("%-25s | %s\n", "Layer", "Latency (us) and Bandwidth");
    printf("%.80s\n", "----------------------------------------------------------------------");

    Dim dims[] = {
        {"Llama-2-7B  (d=4096)", 4096,  4096},
        {"Llama-2-7B  (ff=11008)", 11008, 4096},
        {"Llama-2-70B (d=8192)", 8192,  8192},
        {"Llama-2-70B (ff=28672)", 28672, 8192},
        {"Mixtral (d=4096)",   4096,  4096},
        {"Mixtral (ff=14336)", 14336, 4096},
    };

    for (const auto& dim : dims) {
        bench_dim(dim);
    }

    printf("\nNote: bandwidth for INT4 and split uses INT4 weight size (d_out*d_in/2 bytes).\n");
    printf("      Overhead = split latency / int4 latency; speedup = fp16 latency / variant latency.\n");
    return 0;
}
