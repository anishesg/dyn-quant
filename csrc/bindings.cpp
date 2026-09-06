#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <tuple>

// Forward declarations of CUDA launch functions
#include "quant_format.cuh"
#include "int4_gemv.cuh"
#include "split_gemv.cuh"
#include "reference_gemv.cuh"

// Validates that a tensor is FP16, contiguous, and on CUDA.
static void check_fp16_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(),          name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(),    name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kHalf, name, " must be float16");
}

// quantize_weights(weight: [d_out, d_in] float16)
//   -> (packed: [d_out, d_in/2] uint8,
//       scales: [d_out, d_in/128] float16,
//       zeros:  [d_out, d_in/128] float16,
//       fp16_copy: [d_out, d_in] float16)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
quantize_weights_binding(const torch::Tensor& weight) {
    check_fp16_contiguous(weight, "weight");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [d_out, d_in]");

    int d_out = static_cast<int>(weight.size(0));
    int d_in  = static_cast<int>(weight.size(1));
    TORCH_CHECK(d_in % GROUP_SIZE == 0, "d_in must be divisible by GROUP_SIZE=", GROUP_SIZE);

    const at::cuda::CUDAGuard guard(weight.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Quantize on host (copies to device internally)
    // We need to copy to host first since quantize_weights_host operates on host pointer
    auto weight_cpu = weight.cpu();
    const __half* h_ptr = reinterpret_cast<const __half*>(weight_cpu.data_ptr<at::Half>());
    QuantizedWeights qW = quantize_weights_host(h_ptr, d_out, d_in);

    int num_groups = d_in / GROUP_SIZE;

    // Wrap device pointers into tensors with custom deleters via from_blob
    // We need to transfer ownership, so we copy to new torch tensors and free the qW pointers.
    auto opts_u8   = torch::TensorOptions().dtype(torch::kUInt8).device(weight.device());
    auto opts_fp16 = torch::TensorOptions().dtype(torch::kHalf).device(weight.device());

    auto t_packed = torch::empty({d_out, d_in / 2}, opts_u8);
    auto t_scales = torch::empty({d_out, num_groups}, opts_fp16);
    auto t_zeros  = torch::empty({d_out, num_groups}, opts_fp16);
    auto t_fp16   = torch::empty({d_out, d_in}, opts_fp16);

    cudaMemcpyAsync(t_packed.data_ptr(),
                    qW.packed,
                    static_cast<size_t>(d_out) * (d_in / 2),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(t_scales.data_ptr(),
                    qW.scales,
                    static_cast<size_t>(d_out) * num_groups * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(t_zeros.data_ptr(),
                    qW.zeros,
                    static_cast<size_t>(d_out) * num_groups * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(t_fp16.data_ptr(),
                    qW.fp16_weights,
                    static_cast<size_t>(d_out) * d_in * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    free_quantized_weights(qW);
    return {t_packed, t_scales, t_zeros, t_fp16};
}

// Builds a QuantizedWeights from torch tensors (does not copy; pointers must remain valid).
static QuantizedWeights tensors_to_qweights(
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    const torch::Tensor& fp16_w)
{
    QuantizedWeights q;
    q.d_out        = static_cast<int>(fp16_w.size(0));
    q.d_in         = static_cast<int>(fp16_w.size(1));
    q.packed       = reinterpret_cast<uint8_t*>(packed.data_ptr());
    q.scales       = reinterpret_cast<__half*>(scales.data_ptr<at::Half>());
    q.zeros        = reinterpret_cast<__half*>(zeros.data_ptr<at::Half>());
    q.fp16_weights = reinterpret_cast<__half*>(fp16_w.data_ptr<at::Half>());
    return q;
}

static void check_qweights_args(
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    const torch::Tensor& fp16_w,
    const torch::Tensor& x)
{
    TORCH_CHECK(fp16_w.dim() == 2, "fp16_weights must be 2D");
    check_fp16_contiguous(scales, "scales");
    check_fp16_contiguous(zeros,  "zeros");
    check_fp16_contiguous(fp16_w, "fp16_weights");
    check_fp16_contiguous(x,      "x");
    int d_out = static_cast<int>(fp16_w.size(0));
    int d_in  = static_cast<int>(fp16_w.size(1));
    TORCH_CHECK(x.numel() == d_in, "x length must equal d_in=", d_in);
    TORCH_CHECK(packed.size(1) == d_in / 2, "packed column mismatch");
    (void)d_out;
}

// split_gemv(packed, scales, zeros, fp16_weights, x, outlier_k) -> y [d_out] float16
torch::Tensor split_gemv_binding(
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    const torch::Tensor& fp16_w,
    const torch::Tensor& x,
    double               outlier_k)
{
    check_qweights_args(packed, scales, zeros, fp16_w, x);
    const at::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int d_out = static_cast<int>(fp16_w.size(0));
    int d_in  = static_cast<int>(fp16_w.size(1));

    auto y = torch::empty({d_out}, torch::TensorOptions().dtype(torch::kHalf).device(x.device()));
    QuantizedWeights qW = tensors_to_qweights(packed, scales, zeros, fp16_w);

    launch_split_gemv(qW, reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                      reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
                      d_out, d_in, static_cast<float>(outlier_k), stream);
    return y;
}

// int4_gemv(packed, scales, zeros, fp16_weights, x) -> y [d_out] float16
torch::Tensor int4_gemv_binding(
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    const torch::Tensor& fp16_w,
    const torch::Tensor& x)
{
    check_qweights_args(packed, scales, zeros, fp16_w, x);
    const at::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int d_out = static_cast<int>(fp16_w.size(0));
    int d_in  = static_cast<int>(fp16_w.size(1));

    auto y = torch::empty({d_out}, torch::TensorOptions().dtype(torch::kHalf).device(x.device()));
    QuantizedWeights qW = tensors_to_qweights(packed, scales, zeros, fp16_w);

    launch_int4_gemv(qW, reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                     reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
                     d_out, d_in, stream);
    return y;
}

// reference_gemv(fp16_weights, x) -> y [d_out] float16
torch::Tensor reference_gemv_binding(const torch::Tensor& fp16_w, const torch::Tensor& x) {
    TORCH_CHECK(fp16_w.dim() == 2, "fp16_weights must be 2D");
    check_fp16_contiguous(fp16_w, "fp16_weights");
    check_fp16_contiguous(x,      "x");

    const at::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int d_out = static_cast<int>(fp16_w.size(0));
    int d_in  = static_cast<int>(fp16_w.size(1));
    TORCH_CHECK(x.numel() == d_in, "x length must equal d_in");

    auto y = torch::empty({d_out}, torch::TensorOptions().dtype(torch::kHalf).device(x.device()));
    launch_reference_gemv(
        reinterpret_cast<const __half*>(fp16_w.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        d_out, d_in, stream);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_weights",  &quantize_weights_binding,
          "Quantize FP16 weight matrix to INT4 packed format");
    m.def("split_gemv",        &split_gemv_binding,
          "Fused dynamic-precision GEMV with runtime outlier detection");
    m.def("int4_gemv",         &int4_gemv_binding,
          "Dense INT4 GEMV baseline");
    m.def("reference_gemv",    &reference_gemv_binding,
          "FP16 dense reference GEMV");
}
