#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <vector>
#include <random>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "quant_format.cuh"
#include "int4_gemv.cuh"
#include "split_gemv.cuh"
#include "reference_gemv.cuh"

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

struct Config {
    int   d_out;
    int   d_in;
    float outlier_fraction;
    float outlier_magnitude;
};

void run_test(const Config& cfg) {
    const int d_out             = cfg.d_out;
    const int d_in              = cfg.d_in;
    const float out_frac        = cfg.outlier_fraction;
    const float out_mag         = cfg.outlier_magnitude;

    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.f, 1.f);

    // Generate FP16 weight matrix with controlled statistics
    std::vector<__half> h_W(static_cast<size_t>(d_out) * d_in);
    for (auto& v : h_W) v = __float2half(normal(rng) * 0.02f);

    // Generate FP16 activation vector with controlled outlier channels
    std::vector<__half> h_x(d_in);
    std::vector<bool>   is_outlier_ch(d_in, false);
    int n_outliers = static_cast<int>(out_frac * d_in);
    // Mark first n_outliers channels (deterministic for reproducibility)
    for (int i = 0; i < n_outliers; ++i) is_outlier_ch[i] = true;

    float normal_scale = 1.0f;
    for (int i = 0; i < d_in; ++i) {
        if (is_outlier_ch[i])
            h_x[i] = __float2half(normal(rng) * normal_scale * out_mag);
        else
            h_x[i] = __float2half(normal(rng) * normal_scale);
    }

    // Device allocations
    __half *d_W, *d_x, *d_y_ref, *d_y_int4, *d_y_split;
    cudaMalloc(&d_W,       static_cast<size_t>(d_out) * d_in * sizeof(__half));
    cudaMalloc(&d_x,       d_in  * sizeof(__half));
    cudaMalloc(&d_y_ref,   d_out * sizeof(__half));
    cudaMalloc(&d_y_int4,  d_out * sizeof(__half));
    cudaMalloc(&d_y_split, d_out * sizeof(__half));

    cudaMemcpy(d_W, h_W.data(), static_cast<size_t>(d_out) * d_in * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), d_in * sizeof(__half), cudaMemcpyHostToDevice);

    // Quantize weights to INT4
    QuantizedWeights qW = quantize_weights_host(h_W.data(), d_out, d_in);

    // Run all three kernels
    launch_reference_gemv(d_W, d_x, d_y_ref, d_out, d_in);
    launch_int4_gemv(qW, d_x, d_y_int4, d_out, d_in);
    launch_split_gemv(qW, d_x, d_y_split, d_out, d_in, 3.0f);
    cudaDeviceSynchronize();

    // Copy results to host
    std::vector<__half> h_ref(d_out), h_int4(d_out), h_split(d_out);
    cudaMemcpy(h_ref.data(),   d_y_ref,   d_out * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_int4.data(),  d_y_int4,  d_out * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_split.data(), d_y_split, d_out * sizeof(__half), cudaMemcpyDeviceToHost);

    // Convert to float for metrics
    std::vector<float> f_ref(d_out), f_int4(d_out), f_split(d_out);
    for (int i = 0; i < d_out; ++i) {
        f_ref[i]   = __half2float(h_ref[i]);
        f_int4[i]  = __half2float(h_int4[i]);
        f_split[i] = __half2float(h_split[i]);
    }

    float cos_int4  = cosine_sim(f_ref.data(), f_int4.data(), d_out);
    float cos_split = cosine_sim(f_ref.data(), f_split.data(), d_out);
    float mae_int4  = max_abs_err(f_ref.data(), f_int4.data(), d_out);
    float mae_split = max_abs_err(f_ref.data(), f_split.data(), d_out);

    printf("  d_out=%5d d_in=%5d outlier_frac=%.2f outlier_mag=%4.0fx | "
           "int4: cos=%.6f mae=%.4f | split: cos=%.6f mae=%.4f\n",
           d_out, d_in, out_frac, out_mag,
           cos_int4, mae_int4, cos_split, mae_split);

    // Split-precision must maintain high cosine similarity at all tested configurations
    if (cos_split < 0.9999f) {
        fprintf(stderr, "FAIL: split cosine %.6f < 0.9999 for d_out=%d d_in=%d frac=%.2f mag=%.0fx\n",
                cos_split, d_out, d_in, out_frac, out_mag);
        exit(1);
    }

    free_quantized_weights(qW);
    cudaFree(d_W);
    cudaFree(d_x);
    cudaFree(d_y_ref);
    cudaFree(d_y_int4);
    cudaFree(d_y_split);
}

int main() {
    printf("=== dyn-quant correctness test ===\n\n");

    int d_outs[]              = {4096, 8192};
    int d_ins[]               = {4096, 11008};
    float out_fracs[]         = {0.0f, 0.01f, 0.05f, 0.10f};
    float out_mags[]          = {5.f, 10.f, 50.f};

    int pass = 0, total = 0;

    for (int d_out : d_outs) {
        for (int d_in : d_ins) {
            for (float frac : out_fracs) {
                for (float mag : out_mags) {
                    // Skip 0 outlier fraction with varying magnitude (identical tests)
                    if (frac == 0.0f && mag > 5.f) continue;
                    Config cfg = {d_out, d_in, frac, mag};
                    run_test(cfg);
                    ++pass;
                    ++total;
                }
            }
        }
    }

    printf("\nAll %d/%d tests passed\n", pass, total);
    return 0;
}
