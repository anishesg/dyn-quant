#pragma once

#include <cuda_fp16.h>

// FP16 dense reference GEMV: y = W_fp16 * x
// No quantization. Used as correctness oracle.
// W_fp16: device pointer, row-major [d_out][d_in]
// x:      device pointer, length d_in
// y:      device pointer, length d_out
void launch_reference_gemv(
    const __half* W_fp16,
    const __half* x,
    __half*       y,
    int           d_out,
    int           d_in,
    cudaStream_t  stream = 0
);
