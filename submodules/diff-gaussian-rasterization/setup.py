#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# ---------------------------------------------------------------------------
# ROCm / HIP port (mip-gaussian-l1c).
#
# How to build on ROCm:
#   1. Use a ROCm build of PyTorch (torch.version.hip is not None).
#   2. pip install .  (or: pip install <this directory>)
#
# PyTorch's cpp_extension automatically uses hipcc for .cu sources when
# PyTorch itself was built for ROCm, so CUDAExtension is still the right
# class here -- do NOT switch it to a HIP-specific extension class.
# The .cu sources are compiled by hipcc as HIP code.
#
# GPU arch: PyTorch's ROCm build selects the gfx arch from the
# TORCH_CUDA_ARCH_LIST environment variable (e.g. gfx1100 for a
# Radeon PRO W7900D, gfx942 for MI300X). If the build picks the wrong
# arch, set e.g.:
#   TORCH_CUDA_ARCH_LIST=gfx1100 pip install .
# (Arch flag handling is a TO-VERIFY item on the first ROCm build --
#  see the port report.)
#
# The only CUDA-specific build input was the glm include path, which is
# compiler-agnostic and is kept as-is. The bundled glm 0.9.9.9 already
# has a native HIP branch (GLM_COMPILER_HIP), so no glm changes needed.
# ---------------------------------------------------------------------------

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp"],
            extra_compile_args={
                # Passed to hipcc (not nvcc) when PyTorch is a ROCm build.
                # Keep these flags portable across nvcc/hipcc.
                "nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
