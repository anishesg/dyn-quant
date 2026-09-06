#include "reference_gemv.cuh"
#include <cuda_fp16.h>

static constexpr int REF_TILE_K    = 128;
static constexpr int REF_TILE_ROWS = 4;   // one per warp
static constexpr int REF_BLOCK_DIM = 128; // 4 warps

// Tiled FP16 GEMV. Each warp computes one output row via warp-level reduction.
__global__ void reference_gemv_kernel(
    const __half* __restrict__ W,
    const __half* __restrict__ x,
    float*        __restrict__ y,
    int d_out,
    int d_in)
{
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int row     = blockIdx.x * REF_TILE_ROWS + warp_id;
    const int num_tiles = d_in / REF_TILE_K;

    __shared__ __half s_x[REF_TILE_K];

    float acc = 0.f;

    for (int t = 0; t < num_tiles; ++t) {
        int col_start = t * REF_TILE_K;
        s_x[threadIdx.x] = x[col_start + threadIdx.x];
        __syncthreads();

        if (row < d_out) {
            // Each lane handles 4 columns
            int col_local = lane * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int c = col_local + i;
                acc += __half2float(W[static_cast<size_t>(row) * d_in + col_start + c])
                     * __half2float(s_x[c]);
            }
        }
        __syncthreads();
    }

    acc += __shfl_down_sync(0xFFFFFFFF, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 1);

    if (lane == 0 && row < d_out) {
        y[row] = acc;
    }
}

__global__ void ref_fp32_to_fp16(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void launch_reference_gemv(
    const __half* W_fp16,
    const __half* x,
    __half*       y,
    int           d_out,
    int           d_in,
    cudaStream_t  stream)
{
    float* d_acc;
    cudaMalloc(&d_acc, d_out * sizeof(float));

    int grid  = (d_out + REF_TILE_ROWS - 1) / REF_TILE_ROWS;
    int block = REF_BLOCK_DIM;
    reference_gemv_kernel<<<grid, block, 0, stream>>>(W_fp16, x, d_acc, d_out, d_in);

    int cvt_blocks = (d_out + 255) / 256;
    ref_fp32_to_fp16<<<cvt_blocks, 256, 0, stream>>>(d_acc, y, d_out);

    cudaFree(d_acc);
}
