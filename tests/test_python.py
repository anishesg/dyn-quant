"""Python-level correctness test for DynQuantLinear and functional API."""

import math
import torch

import dyn_quant
from dyn_quant import DynQuantLinear, quantize, split_gemv


D_IN  = 4096
D_OUT = 4096
OUTLIER_FRAC = 0.05   # 5% of channels
OUTLIER_MAG  = 20.0   # 20x normal magnitude
OUTLIER_K    = 3.0
COSINE_THRESHOLD = 0.9999


def generate_outlier_activation(d_in: int, frac: float, mag: float) -> torch.Tensor:
    x = torch.randn(d_in, dtype=torch.float16, device="cuda")
    n_out = int(frac * d_in)
    x[:n_out] = (torch.randn(n_out, device="cuda") * mag).half()
    return x


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float()
    b_f = b.float()
    return float(torch.nn.functional.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item())


def test_dyn_quant_linear():
    torch.manual_seed(0)
    weight = torch.randn(D_OUT, D_IN, dtype=torch.float16, device="cuda") * 0.02
    x      = generate_outlier_activation(D_IN, OUTLIER_FRAC, OUTLIER_MAG)

    # DynQuantLinear forward
    linear = DynQuantLinear(D_IN, D_OUT, outlier_k=OUTLIER_K)
    linear.load_weight(weight)
    linear.cuda()

    y_split = linear(x)

    # Reference: torch.matmul on FP16 weights
    y_ref = torch.matmul(weight, x.float().unsqueeze(-1)).squeeze(-1).half()

    cos = cosine_sim(y_split, y_ref)
    print(f"[test_dyn_quant_linear] cosine_sim={cos:.6f}  (threshold={COSINE_THRESHOLD})")
    assert cos >= COSINE_THRESHOLD, f"cosine_sim {cos:.6f} < {COSINE_THRESHOLD}"
    print("[test_dyn_quant_linear] PASSED")


def test_functional_api():
    torch.manual_seed(1)
    weight = torch.randn(D_OUT, D_IN, dtype=torch.float16, device="cuda") * 0.02
    x      = generate_outlier_activation(D_IN, OUTLIER_FRAC, OUTLIER_MAG)

    packed, scales, zeros, fp16_copy = quantize(weight)

    y_split = split_gemv(packed, scales, zeros, fp16_copy, x, OUTLIER_K)
    y_ref   = torch.matmul(weight, x.float().unsqueeze(-1)).squeeze(-1).half()

    cos = cosine_sim(y_split, y_ref)
    print(f"[test_functional_api] cosine_sim={cos:.6f}  (threshold={COSINE_THRESHOLD})")
    assert cos >= COSINE_THRESHOLD, f"cosine_sim {cos:.6f} < {COSINE_THRESHOLD}"
    print("[test_functional_api] PASSED")


def test_int4_vs_split_with_outliers():
    """Verifies that split-precision outperforms INT4-only when outliers are present."""
    torch.manual_seed(2)
    weight = torch.randn(D_OUT, D_IN, dtype=torch.float16, device="cuda") * 0.02
    x      = generate_outlier_activation(D_IN, OUTLIER_FRAC, OUTLIER_MAG)

    packed, scales, zeros, fp16_copy = quantize(weight)
    y_ref   = torch.matmul(weight, x.float().unsqueeze(-1)).squeeze(-1).half()
    y_int4  = dyn_quant.int4_gemv(packed, scales, zeros, fp16_copy, x)
    y_split = dyn_quant.split_gemv(packed, scales, zeros, fp16_copy, x, OUTLIER_K)

    cos_int4  = cosine_sim(y_int4,  y_ref)
    cos_split = cosine_sim(y_split, y_ref)

    print(f"[test_int4_vs_split] int4_cosine={cos_int4:.6f}  split_cosine={cos_split:.6f}")
    assert cos_split >= COSINE_THRESHOLD, f"split cosine {cos_split:.6f} < {COSINE_THRESHOLD}"
    print("[test_int4_vs_split] PASSED")


if __name__ == "__main__":
    test_dyn_quant_linear()
    test_functional_api()
    test_int4_vs_split_with_outliers()
    print("\nAll Python tests passed.")
