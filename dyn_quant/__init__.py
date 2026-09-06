"""
dyn_quant: Fused dynamic-precision GEMV with warp-cooperative outlier detection.

Exports:
  DynQuantLinear  -- nn.Module that quantizes weights on construction and dispatches
                     forward() to the fused split-precision GEMV kernel.
  quantize()      -- Quantize an FP16 weight tensor to packed INT4 format.
  split_gemv()    -- Fused split-precision GEMV (functional API).
  int4_gemv()     -- Dense INT4 GEMV baseline (functional API).
  reference_gemv()-- FP16 dense reference GEMV (functional API).
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn

try:
    from dyn_quant import _C
    _EXTENSION_LOADED = True
except ImportError:
    _EXTENSION_LOADED = False


def _require_extension() -> None:
    if not _EXTENSION_LOADED:
        raise RuntimeError(
            "dyn_quant C extension not found. "
            "Build it with: pip install -e . or python setup.py build_ext --inplace"
        )


def quantize(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize an FP16 weight matrix to packed INT4 format.

    Args:
        weight: [d_out, d_in] float16 tensor on CUDA.

    Returns:
        (packed, scales, zeros, fp16_copy) where:
          packed    -- [d_out, d_in//2] uint8, two INT4 values per byte
          scales    -- [d_out, d_in//128] float16, per-group scale
          zeros     -- [d_out, d_in//128] float16, per-group zero-point
          fp16_copy -- [d_out, d_in] float16, original weights for outlier columns
    """
    _require_extension()
    if weight.dtype != torch.float16:
        weight = weight.to(torch.float16)
    if not weight.is_contiguous():
        weight = weight.contiguous()
    return _C.quantize_weights(weight)


def split_gemv(
    packed:    torch.Tensor,
    scales:    torch.Tensor,
    zeros:     torch.Tensor,
    fp16_w:    torch.Tensor,
    x:         torch.Tensor,
    outlier_k: float = 3.0,
) -> torch.Tensor:
    """Fused dynamic-precision GEMV with runtime outlier detection.

    Outlier channels (|x_i - mean| > outlier_k * std) are routed to FP16 weight
    columns; remaining channels use INT4 dequantization. Both paths execute in a
    single kernel with no extra global memory traffic for the split.

    Args:
        packed:    [d_out, d_in//2] uint8 packed INT4 weights.
        scales:    [d_out, d_in//128] float16 per-group scales.
        zeros:     [d_out, d_in//128] float16 per-group zero-points.
        fp16_w:    [d_out, d_in] float16 original weights.
        x:         [d_in] float16 activation vector.
        outlier_k: sigma threshold for outlier detection (default 3.0).

    Returns:
        y: [d_out] float16 output vector.
    """
    _require_extension()
    return _C.split_gemv(packed, scales, zeros, fp16_w, x, outlier_k)


def int4_gemv(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros:  torch.Tensor,
    fp16_w: torch.Tensor,
    x:      torch.Tensor,
) -> torch.Tensor:
    """Dense INT4 GEMV without outlier handling.

    Args:
        packed: [d_out, d_in//2] uint8.
        scales: [d_out, d_in//128] float16.
        zeros:  [d_out, d_in//128] float16.
        fp16_w: [d_out, d_in] float16 (used only for shape/device).
        x:      [d_in] float16.

    Returns:
        y: [d_out] float16.
    """
    _require_extension()
    return _C.int4_gemv(packed, scales, zeros, fp16_w, x)


def reference_gemv(fp16_w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """FP16 dense reference GEMV (correctness oracle).

    Args:
        fp16_w: [d_out, d_in] float16.
        x:      [d_in] float16.

    Returns:
        y: [d_out] float16.
    """
    _require_extension()
    return _C.reference_gemv(fp16_w, x)


class DynQuantLinear(nn.Module):
    """Linear layer using fused INT4/FP16 split-precision GEMV.

    Quantizes weights to INT4 on construction and dispatches forward() to the
    fused dynamic-precision kernel. Outlier channels are detected at runtime using
    a warp-cooperative Welford scan; no offline calibration is required.

    Args:
        d_in:       Input feature dimension.
        d_out:      Output feature dimension.
        outlier_k:  Sigma threshold for outlier detection (default 3.0).
    """

    def __init__(self, d_in: int, d_out: int, outlier_k: float = 3.0) -> None:
        super().__init__()
        self.d_in      = d_in
        self.d_out     = d_out
        self.outlier_k = outlier_k
        self._quantized = False

        # Buffers registered after quantization
        self.register_buffer("packed",  None)
        self.register_buffer("scales",  None)
        self.register_buffer("zeros",   None)
        self.register_buffer("fp16_w",  None)

    def load_weight(self, weight: torch.Tensor) -> None:
        """Quantize and store an FP16 weight matrix.

        Args:
            weight: [d_out, d_in] float16 CUDA tensor.
        """
        assert weight.shape == (self.d_out, self.d_in), (
            f"Expected weight shape ({self.d_out}, {self.d_in}), got {weight.shape}")
        packed, scales, zeros, fp16_copy = quantize(weight)
        self.packed  = packed
        self.scales  = scales
        self.zeros   = zeros
        self.fp16_w  = fp16_copy
        self._quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fused split-precision GEMV.

        Args:
            x: [d_in] float16 activation vector.

        Returns:
            y: [d_out] float16.
        """
        if not self._quantized:
            raise RuntimeError("Call load_weight() before forward()")
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        if not x.is_contiguous():
            x = x.contiguous()
        return split_gemv(self.packed, self.scales, self.zeros, self.fp16_w, x, self.outlier_k)
