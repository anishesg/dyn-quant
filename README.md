# dyn-quant

Fused dynamic-precision GEMV: warp-cooperative outlier detection with runtime INT4/FP16 channel splitting in a single kernel launch.

## Problem: Activation Outliers in W4A16 Inference

Large language models develop persistent outlier channels during training. A small fraction of channels (typically 1-5%) carry activations with magnitudes 10-100x larger than the typical distribution. During matrix-vector products these channels contribute disproportionately to the output, yet W4A16 quantization assigns the same 16 quantization levels to both normal and outlier channels within each group.

For a group of 128 channels with one outlier at 50x the typical scale, the quantization grid is stretched to cover the outlier range, leaving only 0.02x the resolution for the 127 normal channels. The resulting reconstruction error on the normal channels dominates the output error because those channels collectively carry more weight.

## Prior Work and Limitations

**LLM.int8() (Dettmers et al., 2022)**: Detects outliers using a static per-tensor absmax threshold determined at inference time. Requires two separate matrix-multiply launches: one INT8 kernel for normal channels, one FP16 kernel for outlier columns. The separate launch overhead adds synchronization cost proportional to batch size.

**SmoothQuant (Xiao et al., 2022)**: Migrates quantization difficulty from activations to weights using per-channel scale factors calibrated offline on a representative dataset. Permanently modifies the model by fusing migration scales into weight matrices. Requires calibration before deployment and cannot adapt to distribution shift at inference time without rerunning calibration.

**Static threshold approaches**: Any approach that fixes the outlier threshold at calibration time assumes the activation distribution at inference matches the calibration set. This fails under distribution shift, prompts outside the calibration domain, or models fine-tuned after calibration.

## This Approach

**Zero calibration, zero extra global memory traffic, single kernel launch.**

A single CUDA kernel performs three fused operations:

1. **Warp-cooperative Welford scan**: All warps cooperatively scan the activation vector to compute its mean and variance using an online Welford algorithm with butterfly warp reductions. Channels where `|x_i - mean| > k * sqrt(variance)` are flagged as outliers. The detection cost is approximately 50-100 cycles amortized over the full vector scan.

2. **Per-tile outlier routing**: The kernel tiles through the input dimension. For each tile, a bitmask marks which positions are outlier channels. Normal positions dequantize INT4 weights using precomputed per-group FP16 scales and zero-points. Outlier positions gather from the original FP16 weight matrix at full precision. Both paths accumulate into the same FP32 output registers.

3. **Single FP16 write**: Output is written once. No intermediate buffers, no second kernel, no synchronization barrier between the split paths.

The result is accuracy comparable to full FP16 for the outlier channels and the memory bandwidth savings of INT4 for the 95-99% of normal channels, all with no offline calibration step.

## Design

- **Weight format**: INT4 packed two-per-byte in row-major layout, group size 128, per-group FP16 scale and zero-point. Original FP16 weights retained on device for outlier column access.
- **Outlier threshold**: Configurable sigma multiplier `k` (default 3.0). Higher `k` reduces outlier fraction at cost of accuracy on extreme channels; lower `k` catches more channels at cost of speed.
- **Tile size**: TILE_K=128 matches the quantization group size, so each tile uses exactly one scale and zero-point value per output row, eliminating group boundary conditionals.

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Requires CUDA toolkit >= 11.8 and a Turing or Ampere GPU (sm_80+).

## Running

```bash
# correctness test
./build/test_correctness

# latency benchmark (Llama-2 7B/70B, Mixtral dimensions)
./build/bench_latency

# accuracy vs speed tradeoff sweep over outlier threshold k
./build/bench_accuracy
```

## Python Extension

```bash
pip install -e .
```

```python
import torch
from dyn_quant import DynQuantLinear

linear = DynQuantLinear(d_in=4096, d_out=4096, outlier_k=3.0)
linear.load_weight(weight_fp16)  # quantizes to INT4 on construction
y = linear(x)                    # fused dynamic-precision forward pass
```
