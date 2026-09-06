#include "split_gemv.cuh"
#include "quant_format.cuh"
#include "outlier_detect.cuh"
#include <cuda_fp16.h>

// TILE_K = GROUP_SIZE = 128: each tile covers exactly one quantization group per row.
static constexpr int SPLIT_TILE_K   = GROUP_SIZE;  // 128
static constexpr int SPLIT_TILE_ROWS = 4;           // output rows per block, one per warp
static constexpr int SPLIT_BLOCK_DIM = 128;         // 4 warps of 32 threads
// Max fraction of d_in that can be outliers before we degrade gracefully
static constexpr int MAX_OUTLIER_FRAC_INV = 4;      // capacity = d_in / 4

// Dynamic shared memory layout (split_gemv_kernel):
//   float  smem_warp_mean[num_warps]    -> 4 floats   = 16 bytes
//   float  smem_warp_M2[num_warps]      -> 4 floats   = 16 bytes
//   int    smem_warp_count[num_warps]   -> 4 ints     = 16 bytes
//   int    smem_warp_ocounts[num_warps] -> 4 ints     = 16 bytes
//   int    outlier_indices[capacity]    -> capacity ints
//   int    scratch[capacity]            -> capacity ints
//   int    outlier_count               -> 1 int
//   __half s_x[SPLIT_TILE_K]           -> 128 halfs  = 256 bytes
//   uint8  s_packed[SPLIT_TILE_ROWS][SPLIT_TILE_K/2] -> 4*64 = 256 bytes
// Total static: 16+16+16+16+2*capacity*4+4+256+256 bytes

__global__ void split_gemv_kernel(
    const uint8_t* __restrict__ packed,
    const __half*  __restrict__ scales,
    const __half*  __restrict__ zeros,
    const __half*  __restrict__ fp16_weights,
    const __half*  __restrict__ x,
    float*         __restrict__ y,
    int   d_out,
    int   d_in,
    float outlier_k,
    int   capacity)  // max outliers to track = d_in / MAX_OUTLIER_FRAC_INV
{
    extern __shared__ char smem[];

    const int warp_id   = threadIdx.x / 32;
    const int lane      = threadIdx.x & 31;
    const int num_warps = SPLIT_BLOCK_DIM / 32;  // 4
    const int row       = blockIdx.x * SPLIT_TILE_ROWS + warp_id;
    const int num_tiles = d_in / SPLIT_TILE_K;

    // Carve out shared memory regions
    float*  smem_warp_mean    = reinterpret_cast<float*>(smem);
    float*  smem_warp_M2      = smem_warp_mean + num_warps;
    int*    smem_warp_count   = reinterpret_cast<int*>(smem_warp_M2 + num_warps);
    int*    smem_warp_ocounts = smem_warp_count + num_warps;
    int*    outlier_count     = smem_warp_ocounts + num_warps;
    int*    outlier_indices   = outlier_count + 1;
    int*    scratch           = outlier_indices + capacity;
    __half* s_x               = reinterpret_cast<__half*>(scratch + capacity);
    // Align to 2 bytes (already aligned since scratch ends at int boundary)
    uint8_t* s_packed         = reinterpret_cast<uint8_t*>(s_x + SPLIT_TILE_K);

    // Phase 1: warp-cooperative Welford outlier detection over the full x vector
    if (threadIdx.x == 0) *outlier_count = 0;
    __syncthreads();

    scan_outliers_block(
        x, d_in, outlier_k,
        smem_warp_mean, smem_warp_M2, smem_warp_count, smem_warp_ocounts,
        outlier_indices, outlier_count, scratch, capacity);
    // After this call, outlier_indices[0..*outlier_count) are valid and __syncthreads() has been called.

    int n_outliers = *outlier_count;

    // Build a tile-local outlier bitmask in registers for fast per-tile lookup.
    // We'll check membership on the fly by iterating the (short) outlier_indices list.
    // For n_outliers typically < 50, a linear scan over shared memory is fast.

    float acc = 0.f;

    for (int t = 0; t < num_tiles; ++t) {
        int col_start = t * SPLIT_TILE_K;

        // Cooperatively load s_x[SPLIT_TILE_K]
        s_x[threadIdx.x] = x[col_start + threadIdx.x];

        // Cooperatively load packed weights for this tile and warp's row
        // Row for each warp: row = blockIdx.x * SPLIT_TILE_ROWS + warp_id
        // Load 64 bytes per row: first 32 threads of each warp, or reuse all 128 threads
        // Use: thread i loads byte i for the row determined by i / 64
        if (threadIdx.x < SPLIT_TILE_ROWS * (SPLIT_TILE_K / 2)) {
            int r       = threadIdx.x / (SPLIT_TILE_K / 2);
            int byte_i  = threadIdx.x % (SPLIT_TILE_K / 2);
            int global_r = blockIdx.x * SPLIT_TILE_ROWS + r;
            if (global_r < d_out) {
                s_packed[r * (SPLIT_TILE_K / 2) + byte_i] =
                    packed[static_cast<size_t>(global_r) * (d_in / 2) + col_start / 2 + byte_i];
            }
        }
        __syncthreads();

        if (row < d_out) {
            size_t sg_idx = static_cast<size_t>(row) * num_tiles + t;
            float  sf     = __half2float(scales[sg_idx]);
            float  zf     = __half2float(zeros[sg_idx]);
            uint8_t* row_packed = s_packed + warp_id * (SPLIT_TILE_K / 2);

            // Each lane handles 4 columns: lane*4 .. lane*4+3
            int col_local = lane * 4;

#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int c0 = col_local + i * 2;
                int c1 = c0 + 1;
                int global_c0 = col_start + c0;
                int global_c1 = col_start + c1;

                float x0 = __half2float(s_x[c0]);
                float x1 = __half2float(s_x[c1]);

                // Check if either column is an outlier by scanning outlier_indices
                bool out0 = false, out1 = false;
                for (int o = 0; o < n_outliers; ++o) {
                    if (outlier_indices[o] == global_c0) { out0 = true; }
                    if (outlier_indices[o] == global_c1) { out1 = true; }
                }

                float w0, w1;
                if (out0) {
                    // Use FP16 weight column at full precision
                    w0 = __half2float(fp16_weights[static_cast<size_t>(row) * d_in + global_c0]);
                } else {
                    uint8_t pb = row_packed[c0 / 2];
                    w0 = sf * (static_cast<float>(unpack_int4_lo(pb)) - zf);
                }
                if (out1) {
                    w1 = __half2float(fp16_weights[static_cast<size_t>(row) * d_in + global_c1]);
                } else {
                    uint8_t pb = row_packed[c1 / 2];
                    w1 = sf * (static_cast<float>(unpack_int4_hi(pb)) - zf);
                }

                acc += w0 * x0 + w1 * x1;
            }
        }
        __syncthreads();
    }

    // Warp-level reduction
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 1);

    if (lane == 0 && row < d_out) {
        y[row] = acc;
    }
}

__global__ void fp32_to_fp16_split(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void launch_split_gemv(
    const QuantizedWeights& W,
    const __half* x,
    __half*       y,
    int           d_out,
    int           d_in,
    float         outlier_k,
    cudaStream_t  stream)
{
    float* d_acc;
    cudaMalloc(&d_acc, d_out * sizeof(float));

    int capacity = d_in / MAX_OUTLIER_FRAC_INV;
    const int num_warps = SPLIT_BLOCK_DIM / 32;  // 4

    // Shared memory: see layout comment above
    size_t smem_bytes =
        num_warps * sizeof(float) +   // smem_warp_mean
        num_warps * sizeof(float) +   // smem_warp_M2
        num_warps * sizeof(int)   +   // smem_warp_count
        num_warps * sizeof(int)   +   // smem_warp_ocounts
        sizeof(int)               +   // outlier_count
        capacity  * sizeof(int)   +   // outlier_indices
        capacity  * sizeof(int)   +   // scratch
        SPLIT_TILE_K * sizeof(__half) +  // s_x
        SPLIT_TILE_ROWS * (SPLIT_TILE_K / 2) * sizeof(uint8_t);  // s_packed

    int grid  = (d_out + SPLIT_TILE_ROWS - 1) / SPLIT_TILE_ROWS;
    int block = SPLIT_BLOCK_DIM;

    split_gemv_kernel<<<grid, block, smem_bytes, stream>>>(
        W.packed, W.scales, W.zeros, W.fp16_weights,
        x, d_acc, d_out, d_in, outlier_k, capacity);

    int cvt_blocks = (d_out + 255) / 256;
    fp32_to_fp16_split<<<cvt_blocks, 256, 0, stream>>>(d_acc, y, d_out);

    cudaFree(d_acc);
}
