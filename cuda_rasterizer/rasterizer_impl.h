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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	/// @brief 从内存块中获取特定类型的数据指针，并更新内存块的指针位置
	/// @tparam T 
	/// @param chunk 指向内存块的指针，通过引用传递，表示内存块的起始位置
	/// @param ptr 表示要获取的特定类型 T 的数据指针的目标地址
	/// @param count 表示要获取的数据块的数量
	/// @param alignment 表示数据的对齐要求，通常为字节对齐的值
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		// 计算偏移量 TODO:
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		// 将偏移后的指针转为T*
		ptr = reinterpret_cast<T*>(offset);
		// 更新 chunk 指针，使其指向下一个数据块的起始位置
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	// yu几何相关的状态信息
	struct GeometryState
	{
		size_t scan_size; // 扫描尺寸
		float* depths; // 指向深度值数组的指针
		char* scanning_space; // 指向扫描空间的指针，存储扫描过程的临时数据
		bool* clamped; // 指向布尔数组的指针，用于标记是否被截断
		int* internal_radii; // 存储内部半径信息
		float2* means2D; // 可能用于存储二维均值
		float* cov3D; // 存储三维协方差信息
		float4* conic_opacity; // 存储椭圆体透明度信息 float4是指连续存放的4个float32
		float* rgb;
		uint32_t* point_offsets; // 点偏移
		uint32_t* tiles_touched; // 触及的图块（瓦片）

		// 从内存块中创建一个 GeometryState 对象的静态方法
		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};