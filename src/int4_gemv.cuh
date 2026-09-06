#pragma once

#include "quant_format.cuh"
#include <cuda_fp16.h>

// Baseline dense INT4 GEMV: y = dequant(W_int4) * x
// Each thread block computes TILE_ROWS=4 output elements.
// grid: (d_out / TILE_ROWS), block: (128)
void launch_int4_gemv(
    const QuantizedWeights& W,
    const __half* x,      // device, length d_in
    __half*       y,      // device, length d_out
    int           d_out,
    int           d_in,
    cudaStream_t  stream = 0
);
