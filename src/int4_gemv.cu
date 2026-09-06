#include "int4_gemv.cuh"
#include "quant_format.cuh"
#include <cuda_fp16.h>

// TILE_K must equal GROUP_SIZE so each tile uses exactly one scale/zero pair per row.
static constexpr int TILE_K    = GROUP_SIZE;  // 128 columns per tile
static constexpr int TILE_ROWS = 4;           // output rows per block (one per warp)
static constexpr int BLOCK_DIM = 128;         // 4 warps of 32 threads

// Each warp handles one output row. Threads in a warp cover TILE_K=128 columns per iteration.
// Shared memory: activation tile s_x[TILE_K] shared across all warps.
__global__ void int4_gemv_kernel(
    const uint8_t* __restrict__ packed,
    const __half*  __restrict__ scales,
    const __half*  __restrict__ zeros,
    const __half*  __restrict__ x,
    float*         __restrict__ y,   // FP32 output for atomic-free accumulation
    int d_out,
    int d_in)
{
    const int warp_id  = threadIdx.x / 32;
    const int lane     = threadIdx.x & 31;
    const int row      = blockIdx.x * TILE_ROWS + warp_id;
    const int num_tiles = d_in / TILE_K;

    __shared__ __half s_x[TILE_K];

    float acc = 0.f;

    for (int t = 0; t < num_tiles; ++t) {
        int col_start = t * TILE_K;

        // All 128 threads cooperatively load s_x[TILE_K]
        s_x[threadIdx.x] = x[col_start + threadIdx.x];
        __syncthreads();

        if (row < d_out) {
            size_t sg_idx  = static_cast<size_t>(row) * num_tiles + t;
            float scale_f  = __half2float(scales[sg_idx]);
            float zero_f   = __half2float(zeros[sg_idx]);
            // Base byte offset for this row + tile
            size_t byte_base = static_cast<size_t>(row) * (d_in / 2) + col_start / 2;

            // Each lane covers 4 columns: lane handles cols [lane*4, lane*4+3]
            // TILE_K=128 cols, 32 lanes -> 4 cols per lane -> 2 packed bytes per lane
            int col_local = lane * 4;  // first column this lane handles in the tile
            int byte_off  = col_local / 2;  // first byte index

#pragma unroll
            for (int i = 0; i < 2; ++i) {  // 2 bytes = 4 nibbles = 4 columns
                uint8_t pb   = packed[byte_base + byte_off + i];
                int c0       = col_local + i * 2;
                int c1       = c0 + 1;
                float w0 = scale_f * (static_cast<float>(unpack_int4_lo(pb)) - zero_f);
                float w1 = scale_f * (static_cast<float>(unpack_int4_hi(pb)) - zero_f);
                acc += w0 * __half2float(s_x[c0]);
                acc += w1 * __half2float(s_x[c1]);
            }
        }
        __syncthreads();
    }

    // Warp-level reduction of acc within each warp (each warp -> one output row)
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 1);

    if (lane == 0 && row < d_out) {
        y[row] = acc;
    }
}

__global__ void fp32_to_fp16_kernel(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void launch_int4_gemv(
    const QuantizedWeights& W,
    const __half* x,
    __half*       y,
    int           d_out,
    int           d_in,
    cudaStream_t  stream)
{
    float* d_acc;
    cudaMalloc(&d_acc, d_out * sizeof(float));

    int grid  = (d_out + TILE_ROWS - 1) / TILE_ROWS;
    int block = BLOCK_DIM;
    int4_gemv_kernel<<<grid, block, 0, stream>>>(
        W.packed, W.scales, W.zeros, x, d_acc, d_out, d_in);

    int cvt_blocks = (d_out + 255) / 256;
    fp32_to_fp16_kernel<<<cvt_blocks, 256, 0, stream>>>(d_acc, y, d_out);

    cudaFree(d_acc);
}
