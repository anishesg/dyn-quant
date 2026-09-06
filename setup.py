from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

src_dir = os.path.join(os.path.dirname(__file__), "src")
csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-use_fast_math",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ],
}

setup(
    name="dyn_quant",
    version="0.1.0",
    description="Fused dynamic-precision GEMV with warp-cooperative outlier detection",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="dyn_quant._C",
            sources=[
                os.path.join(csrc_dir, "bindings.cpp"),
                os.path.join(src_dir,  "int4_gemv.cu"),
                os.path.join(src_dir,  "split_gemv.cu"),
                os.path.join(src_dir,  "reference_gemv.cu"),
            ],
            include_dirs=[src_dir, csrc_dir],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch"],
)
