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
# Build with:
#   cmake -B build -DHIP_ARCH=gfx1100   # W7900D; use gfx942 for MI300X
#   cmake --build build
#
# The .cu sources are compiled as HIP. CMake's HIP language normally
# expects .hip files, so the sources are explicitly forced to LANGUAGE HIP
# below (avoids renaming files and keeps the diff vs upstream minimal).
# ---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES HIP CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 17)

# ROCm GPU arch, e.g. gfx1100 (Radeon PRO W7900D), gfx942 (MI300X).
# Override on the cmake command line: -DHIP_ARCH=gfx942
set(HIP_ARCH "gfx1100" CACHE STRING "ROCm GPU arch (offload-arch)")

add_library(HipRasterizer
	cuda_rasterizer/backward.h
	cuda_rasterizer/backward.cu
	cuda_rasterizer/forward.h
	cuda_rasterizer/forward.cu
	cuda_rasterizer/auxiliary.h
	cuda_rasterizer/rasterizer_impl.cu
	cuda_rasterizer/rasterizer_impl.h
	cuda_rasterizer/rasterizer.h
)

# Keep the .cu filenames (minimal diff vs upstream) but compile them as HIP.
set_source_files_properties(
	cuda_rasterizer/backward.cu
	cuda_rasterizer/forward.cu
	cuda_rasterizer/rasterizer_impl.cu
	PROPERTIES LANGUAGE HIP
)

set_target_properties(HipRasterizer PROPERTIES HIP_ARCHITECTURES "${HIP_ARCH}")

target_include_directories(HipRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_include_directories(HipRasterizer PRIVATE third_party/glm)
