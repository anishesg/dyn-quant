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

static constexpr int D_OUT = 4096;
static constexpr int D_IN  = 4096;
static constexpr float OUTLIER_FRAC = 0.02f;   // 2% of channels
static constexpr float OUTLIER_SIGMA = 20.0f;   // drawn from N(0, 20)

static constexpr int WARMUP_ITERS = 5;
static constexpr int BENCH_ITERS  = 50;

static float cosine_sim(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    double denom = sqrt(na * nb);
    return (denom < 1e-10) ? 1.f : static_cast<float>(dot / denom);
}

static float max_abs_err(const float* a, const float* b, int n) {
    float err = 0;
    for (int i = 0; i < n; ++i) err = fmaxf(err, fabsf(a[i] - b[i]));
    return err;
}

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
    return ms * 1000.f / BENCH_ITERS;
}

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> normal_dist(0.f, 1.f);
    std::normal_distribution<float> outlier_dist(0.f, OUTLIER_SIGMA);

    // Build weight matrix
    std::vector<__half> h_W(static_cast<size_t>(D_OUT) * D_IN);
    for (auto& v : h_W) v = __float2half(normal_dist(rng) * 0.02f);

    // Build activation vector with 2% outlier channels
    std::vector<__half> h_x(D_IN);
    int n_outliers = static_cast<int>(OUTLIER_FRAC * D_IN);
    std::vector<int> outlier_positions(D_IN);
    for (int i = 0; i < D_IN; ++i) outlier_positions[i] = i;
    // Shuffle to randomize outlier positions
    for (int i = D_IN - 1; i > 0; --i) {
        std::uniform_int_distribution<int> ud(0, i);
        int j = ud(rng);
        std::swap(outlier_positions[i], outlier_positions[j]);
    }
    std::vector<bool> is_outlier(D_IN, false);
    for (int i = 0; i < n_outliers; ++i) is_outlier[outlier_positions[i]] = true;

    for (int i = 0; i < D_IN; ++i) {
        h_x[i] = is_outlier[i]
                 ? __float2half(outlier_dist(rng))
                 : __float2half(normal_dist(rng));
    }

    // Device allocations
    __half *d_W, *d_x, *d_y_ref, *d_y_int4, *d_y_split;
    cudaMalloc(&d_W,       static_cast<size_t>(D_OUT) * D_IN * sizeof(__half));
    cudaMalloc(&d_x,       D_IN  * sizeof(__half));
    cudaMalloc(&d_y_ref,   D_OUT * sizeof(__half));
    cudaMalloc(&d_y_int4,  D_OUT * sizeof(__half));
    cudaMalloc(&d_y_split, D_OUT * sizeof(__half));
    cudaMemcpy(d_W, h_W.data(), static_cast<size_t>(D_OUT) * D_IN * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), D_IN * sizeof(__half), cudaMemcpyHostToDevice);

    QuantizedWeights qW = quantize_weights_host(h_W.data(), D_OUT, D_IN);

    // Baseline INT4 latency
    float us_int4 = bench_us([&]{ launch_int4_gemv(qW, d_x, d_y_int4, D_OUT, D_IN); });

    // Reference FP16 output for cosine/MAE metrics
    launch_reference_gemv(d_W, d_x, d_y_ref, D_OUT, D_IN);
    cudaDeviceSynchronize();
    std::vector<__half> h_ref(D_OUT);
    cudaMemcpy(h_ref.data(), d_y_ref, D_OUT * sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<float> f_ref(D_OUT);
    for (int i = 0; i < D_OUT; ++i) f_ref[i] = __half2float(h_ref[i]);

    printf("=== Accuracy vs Speed Tradeoff (outlier_frac=%.0f%%, outlier_sigma=%.0f) ===\n\n",
           OUTLIER_FRAC * 100, OUTLIER_SIGMA);
    printf("%-6s | %-12s | %-10s | %-12s | %-12s | %s\n",
           "k", "detected_%", "cos_sim", "max_abs_err", "split_us", "overhead_vs_int4");
    printf("%.80s\n", "-----------------------------------------------------------------------");

    float ks[] = {1.5f, 2.0f, 2.5f, 3.0f, 4.0f, 5.0f};
    for (float k : ks) {
        // Detect outliers on CPU to measure detected fraction
        float h_x_f[D_IN];
        double mean = 0, M2 = 0;
        int count = 0;
        for (int i = 0; i < D_IN; ++i) {
            float v = __half2float(h_x[i]);
            count++;
            double delta = v - mean;
            mean += delta / count;
            M2   += delta * (v - mean);
            h_x_f[i] = v;
        }
        float stddev = (count > 1) ? sqrtf(static_cast<float>(M2 / (count - 1))) : 0.f;
        float thresh = k * stddev;
        int detected = 0;
        for (int i = 0; i < D_IN; ++i) {
            if (fabsf(h_x_f[i] - static_cast<float>(mean)) > thresh) ++detected;
        }
        float detected_pct = 100.f * detected / D_IN;

        float us_split = bench_us([&]{ launch_split_gemv(qW, d_x, d_y_split, D_OUT, D_IN, k); });

        launch_split_gemv(qW, d_x, d_y_split, D_OUT, D_IN, k);
        cudaDeviceSynchronize();
        std::vector<__half> h_split(D_OUT);
        cudaMemcpy(h_split.data(), d_y_split, D_OUT * sizeof(__half), cudaMemcpyDeviceToHost);
        std::vector<float> f_split(D_OUT);
        for (int i = 0; i < D_OUT; ++i) f_split[i] = __half2float(h_split[i]);

        float cos_s   = cosine_sim(f_ref.data(), f_split.data(), D_OUT);
        float mae_s   = max_abs_err(f_ref.data(), f_split.data(), D_OUT);
        float overhead = us_split / us_int4;

        printf("%-6.1f | %10.2f%% | %.6f | %12.4f | %9.1f us | %.2fx\n",
               k, detected_pct, cos_s, mae_s, us_split, overhead);
    }

    printf("\nINT4-only: %.1f us (baseline)\n", us_int4);

    free_quantized_weights(qW);
    cudaFree(d_W);
    cudaFree(d_x);
    cudaFree(d_y_ref);
    cudaFree(d_y_int4);
    cudaFree(d_y_split);
    return 0;
}
