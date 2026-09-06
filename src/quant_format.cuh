#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include <cstdlib>
#include <cmath>
#include <cfloat>

// Group size for per-group INT4 quantization. Matches TILE_K in kernels.
static constexpr int GROUP_SIZE = 128;

struct QuantizedWeights {
    // INT4 packed weights: 2 values per byte, row-major, shape [d_out][d_in/2]
    uint8_t* packed;
    // Per-group FP16 scales: shape [d_out][d_in/GROUP_SIZE]
    __half* scales;
    // Per-group FP16 zero-points: shape [d_out][d_in/GROUP_SIZE]
    __half* zeros;
    // Original FP16 weight matrix for outlier-channel access: shape [d_out][d_in]
    __half* fp16_weights;
    int d_out;
    int d_in;
};

// Packs two 4-bit values (each in [0,15]) into one byte. lo occupies bits [3:0], hi bits [7:4].
__device__ __forceinline__ uint8_t pack_int4(uint8_t lo, uint8_t hi) {
    return (lo & 0xF) | ((hi & 0xF) << 4);
}

// Extracts the low nibble (bits [3:0]) from a packed byte.
__device__ __forceinline__ uint8_t unpack_int4_lo(uint8_t packed) {
    return packed & 0xF;
}

// Extracts the high nibble (bits [7:4]) from a packed byte.
__device__ __forceinline__ uint8_t unpack_int4_hi(uint8_t packed) {
    return (packed >> 4) & 0xF;
}

// Dequantizes a 4-bit code to FP16 using per-group scale and zero-point.
// Reconstruction: value = (code - zero) * scale
__device__ __forceinline__ __half dequantize_int4_group(uint8_t code, __half scale, __half zero) {
    float v = (__half2float(scale)) * (static_cast<float>(code) - __half2float(zero));
    return __float2half(v);
}

// Quantizes a single FP32 value given group min and scale, returning INT4 code in [0,15].
inline int quantize_val(float val, float min_val, float inv_scale) {
    int code = static_cast<int>((val - min_val) * inv_scale + 0.5f);
    if (code < 0) code = 0;
    if (code > 15) code = 15;
    return code;
}

// Converts an FP16 host matrix to the packed QuantizedWeights format.
// fp16_src: row-major [d_out][d_in] FP16 values on host.
// Returns a QuantizedWeights with all device pointers allocated and populated.
inline QuantizedWeights quantize_weights_host(const __half* fp16_src, int d_out, int d_in) {
    int num_groups = d_in / GROUP_SIZE;
    size_t packed_bytes = static_cast<size_t>(d_out) * (d_in / 2);
    size_t scale_elems  = static_cast<size_t>(d_out) * num_groups;
    size_t fp16_elems   = static_cast<size_t>(d_out) * d_in;

    uint8_t*  h_packed = new uint8_t[packed_bytes];
    __half*   h_scales = new __half[scale_elems];
    __half*   h_zeros  = new __half[scale_elems];

    for (int row = 0; row < d_out; ++row) {
        const __half* src_row = fp16_src + static_cast<size_t>(row) * d_in;
        for (int g = 0; g < num_groups; ++g) {
            int col_start = g * GROUP_SIZE;
            // Compute per-group min and max
            float gmin =  FLT_MAX;
            float gmax = -FLT_MAX;
            for (int i = 0; i < GROUP_SIZE; ++i) {
                float v = __half2float(src_row[col_start + i]);
                if (v < gmin) gmin = v;
                if (v > gmax) gmax = v;
            }
            float range = gmax - gmin;
            float scale_f = (range < 1e-8f) ? 1.0f : range / 15.0f;
            float inv_scale = 1.0f / scale_f;
            // Zero-point in INT4 space (symmetric-ish, shifted to [0,15])
            float zero_f = -gmin * inv_scale;

            h_scales[static_cast<size_t>(row) * num_groups + g] = __float2half(scale_f);
            h_zeros[static_cast<size_t>(row) * num_groups + g]  = __float2half(zero_f);

            // Pack group into INT4 bytes
            for (int i = 0; i < GROUP_SIZE; i += 2) {
                uint8_t lo = static_cast<uint8_t>(quantize_val(__half2float(src_row[col_start + i]),     gmin, inv_scale));
                uint8_t hi = static_cast<uint8_t>(quantize_val(__half2float(src_row[col_start + i + 1]), gmin, inv_scale));
                size_t byte_idx = static_cast<size_t>(row) * (d_in / 2) + (col_start + i) / 2;
                h_packed[byte_idx] = pack_int4(lo, hi);
            }
        }
    }

    QuantizedWeights q;
    q.d_out = d_out;
    q.d_in  = d_in;
    cudaMalloc(&q.packed,      packed_bytes * sizeof(uint8_t));
    cudaMalloc(&q.scales,      scale_elems  * sizeof(__half));
    cudaMalloc(&q.zeros,       scale_elems  * sizeof(__half));
    cudaMalloc(&q.fp16_weights, fp16_elems  * sizeof(__half));

    cudaMemcpy(q.packed,       h_packed,  packed_bytes * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(q.scales,       h_scales,  scale_elems  * sizeof(__half),  cudaMemcpyHostToDevice);
    cudaMemcpy(q.zeros,        h_zeros,   scale_elems  * sizeof(__half),  cudaMemcpyHostToDevice);
    cudaMemcpy(q.fp16_weights, fp16_src,  fp16_elems   * sizeof(__half),  cudaMemcpyHostToDevice);

    delete[] h_packed;
    delete[] h_scales;
    delete[] h_zeros;
    return q;
}

inline void free_quantized_weights(QuantizedWeights& q) {
    cudaFree(q.packed);
    cudaFree(q.scales);
    cudaFree(q.zeros);
    cudaFree(q.fp16_weights);
}
