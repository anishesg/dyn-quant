#pragma once

#include "quant_format.cuh"
#include <cuda_fp16.h>

// Fused split-precision GEMV: y = W_split * x
// Outlier channels use FP16 weight columns; normal channels use INT4 dequantization.
// Outlier detection runs inside the kernel via warp-cooperative Welford scan.
// No separate kernel launch, no extra global memory traffic for the split.
void launch_split_gemv(
    const QuantizedWeights& W,
    const __half* x,           // device, length d_in
    __half*       y,           // device, length d_out
    int           d_out,
    int           d_in,
    float         outlier_k,   // sigma threshold for outlier detection (default 3.0)
    cudaStream_t  stream = 0
);
