#pragma once

#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

// Welford online statistics accumulator.
struct WelfordState {
    float mean;
    float M2;     // sum of squared deviations from running mean
    int   count;
};

__device__ __forceinline__ WelfordState welford_init() {
    return {0.f, 0.f, 0};
}

__device__ __forceinline__ void welford_update(WelfordState& s, float x) {
    s.count += 1;
    float delta  = x - s.mean;
    s.mean      += delta / static_cast<float>(s.count);
    s.M2        += delta * (x - s.mean);
}

// Chan et al. parallel merge of two partial Welford states.
__device__ __forceinline__ WelfordState welford_merge(WelfordState a, WelfordState b) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;
    WelfordState out;
    out.count = a.count + b.count;
    float delta = b.mean - a.mean;
    out.mean = a.mean + delta * static_cast<float>(b.count) / static_cast<float>(out.count);
    out.M2   = a.M2 + b.M2 + delta * delta *
               (static_cast<float>(a.count) * static_cast<float>(b.count)) /
               static_cast<float>(out.count);
    return out;
}

// Butterfly warp reduction; every lane ends with the merged global WelfordState.
__device__ __forceinline__ WelfordState warp_reduce_welford(WelfordState s) {
    for (int offset = 16; offset >= 1; offset >>= 1) {
        WelfordState other;
        other.mean  = __shfl_xor_sync(0xFFFFFFFF, s.mean,  offset);
        other.M2    = __shfl_xor_sync(0xFFFFFFFF, s.M2,    offset);
        other.count = __shfl_xor_sync(0xFFFFFFFF, s.count, offset);
        s = welford_merge(s, other);
    }
    return s;
}

// Scans activation vector x[0..n) for outliers using a single warp (32 threads).
// Accumulates per-lane Welford stats, butterfly-reduces to global mean/variance,
// then ballot-compacts outlier indices into outlier_indices[0..capacity).
// Returns the total outlier count (may exceed capacity; only capacity entries are stored).
// n should be a multiple of 32; partial trailing elements are handled safely.
__device__ __forceinline__ int scan_outliers_warp(
    const __half* __restrict__ x,
    int    n,
    float  k,
    int*   outlier_indices,  // shared or device buffer
    int    capacity)
{
    const int lane = threadIdx.x & 31;

    WelfordState local = welford_init();
    for (int i = lane; i < n; i += 32)
        welford_update(local, __half2float(x[i]));

    WelfordState gs = warp_reduce_welford(local);
    float mean   = gs.mean;
    float var    = (gs.count > 1) ? gs.M2 / static_cast<float>(gs.count - 1) : 0.f;
    float thresh = k * sqrtf(fmaxf(var, 1e-8f));

    int total_written = 0;
    for (int base = 0; base < n; base += 32) {
        int   idx        = base + lane;
        bool  active     = idx < n;
        float val        = active ? __half2float(x[idx]) : mean;
        bool  is_outlier = active && (fabsf(val - mean) > thresh);

        unsigned ballot  = __ballot_sync(0xFFFFFFFF, is_outlier);
        int  write_off   = total_written + __popc(ballot & ((1u << lane) - 1u));

        if (is_outlier && write_off < capacity)
            outlier_indices[write_off] = idx;

        total_written += __popc(ballot);
    }
    return total_written;
}

// Block-cooperative outlier scanner: all warps scan x[0..n) together.
// Phase 1: per-thread Welford, warp reduce, shared-memory merge to global stats.
// Phase 2: each warp scans its stride-interleaved elements for outliers.
// Phase 3: warp-level prefix sum merges per-warp compacted indices into output buffer.
//
// Shared memory requirements (caller must allocate):
//   smem_stats[3 * num_warps] floats: for mean, M2 (float), count (int) per warp
//   smem_warp_counts[num_warps] ints
//   smem_out_buf[capacity] ints: final output
//   smem_scratch[capacity] ints: per-warp scratch (over-allocated; each warp uses its portion)
//
// After return, outlier_indices[0..*outlier_count) are valid on all threads in the block.
__device__ __forceinline__ void scan_outliers_block(
    const __half* __restrict__ x,
    int    n,
    float  k,
    int*   smem_warp_mean,    // float*, size = num_warps
    float* smem_warp_M2,      // float*, size = num_warps
    int*   smem_warp_count,   // int*,   size = num_warps
    int*   smem_warp_ocounts, // int*,   size = num_warps (per-warp outlier counts)
    int*   outlier_indices,   // int*,   size = capacity  (output)
    int*   outlier_count,     // int*,   single element   (output count)
    int*   scratch,           // int*,   size = capacity  (scratch)
    int    capacity)
{
    const int tid       = threadIdx.x;
    const int warp_id   = tid / 32;
    const int lane      = tid & 31;
    const int num_warps = blockDim.x / 32;

    // Phase 1: compute global mean and variance
    WelfordState local = welford_init();
    for (int i = tid; i < n; i += blockDim.x)
        welford_update(local, __half2float(x[i]));

    WelfordState ws = warp_reduce_welford(local);
    if (lane == 0) {
        smem_warp_mean[warp_id]  = ws.mean;
        smem_warp_M2[warp_id]    = ws.M2;
        smem_warp_count[warp_id] = ws.count;
    }
    __syncthreads();

    if (tid == 0) {
        WelfordState g = welford_init();
        for (int w = 0; w < num_warps; ++w) {
            WelfordState tmp;
            tmp.mean  = smem_warp_mean[w];
            tmp.M2    = smem_warp_M2[w];
            tmp.count = smem_warp_count[w];
            g = welford_merge(g, tmp);
        }
        smem_warp_mean[0]  = g.mean;
        smem_warp_M2[0]    = g.M2;
        smem_warp_count[0] = g.count;
    }
    __syncthreads();

    float mean   = smem_warp_mean[0];
    float var    = (smem_warp_count[0] > 1)
                 ? smem_warp_M2[0] / static_cast<float>(smem_warp_count[0] - 1)
                 : 0.f;
    float thresh = k * sqrtf(fmaxf(var, 1e-8f));

    // Phase 2: each warp scans its stride-interleaved elements, writes to scratch
    // Each warp uses a private region of scratch: warp_id * (capacity/num_warps)
    // For simplicity, warps write to a shared scratch and use atomic offsets,
    // but to avoid atomics we use the ballot+popc pattern with a warp-local base.
    int warp_cap = capacity / num_warps;  // max per warp
    int* warp_scratch = scratch + warp_id * warp_cap;

    int warp_out = 0;
    for (int base = warp_id * 32; base < n; base += num_warps * 32) {
        int   idx        = base + lane;
        bool  active     = idx < n;
        float val        = active ? __half2float(x[idx]) : mean;
        bool  is_outlier = active && (fabsf(val - mean) > thresh);

        unsigned ballot = __ballot_sync(0xFFFFFFFF, is_outlier);
        int write_off   = warp_out + __popc(ballot & ((1u << lane) - 1u));

        if (is_outlier && write_off < warp_cap)
            warp_scratch[write_off] = idx;

        warp_out += __popc(ballot);
    }
    if (lane == 0) smem_warp_ocounts[warp_id] = min(warp_out, warp_cap);
    __syncthreads();

    // Phase 3: exclusive prefix sum over warp_ocounts, then copy to output
    if (tid == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; ++w) {
            int c = smem_warp_ocounts[w];
            smem_warp_ocounts[w] = total;
            total += c;
        }
        *outlier_count = min(total, capacity);
    }
    __syncthreads();

    int prefix = smem_warp_ocounts[warp_id];
    int my_count;
    if (warp_id + 1 < num_warps)
        my_count = smem_warp_ocounts[warp_id + 1] - prefix;
    else
        my_count = *outlier_count - prefix;

    for (int i = lane; i < my_count; i += 32) {
        if (prefix + i < capacity)
            outlier_indices[prefix + i] = warp_scratch[i];
    }
    __syncthreads();
}
