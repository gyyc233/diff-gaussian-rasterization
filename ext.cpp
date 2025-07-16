/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

// 使用 Pybind11 定义一个 Python 扩展模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}

// 将 C++ 函数 RasterizeGaussiansCUDA 暴露为 Python 函数 rasterize_gaussians
// RasterizeGaussiansBackwardCUDA 暴露为 rasterize_gaussians_backward
// markVisible 暴露为 mark_visible
