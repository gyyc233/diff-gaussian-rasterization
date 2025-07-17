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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

// CUDA 渲染器的配置参数

#define NUM_CHANNELS 3 // Default 3, RGB

// 每个线程块中x,y维度线程数量，这里表示每个线程块包含256个线程
#define BLOCK_X 16 //  CUDA 核函数中线程块的大小
#define BLOCK_Y 16

#endif