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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.

/// @brief 将每个高斯的输入球谐系数转换为RGB，该RGB还需要乘上C_0再乘255
/// @param idx 当前处理的高斯索引
/// @param deg 球谐函数最高阶数
/// @param max_coeffs 每个高斯所包含的球谐系数的最大数量
/// @param means 指向高斯中心在3维空间位置指针
/// @param campos 相机坐标
/// @param shs 指向球谐函数系数数组
/// @param clamped 用于在后向传播中跟踪rgb颜色是否为正值
/// @return rgb
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir); // 计算从相机到点的方向向量并归一化

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs; // 获取当前点的球谐系数
	glm::vec3 result = SH_C0 * sh[0]; // 计算0阶球谐函数项（基础颜色项）

	if (deg > 0)
	{
		// 球谐函数方向
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f; // 添加0.5偏移量

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation

/// @brief 前向２d协方差矩阵计算，模拟了EWA步骤
/// @param mean 高斯分布中心点
/// @param focal_x ｘ轴焦距
/// @param focal_y 
/// @param tan_fovx x 轴方向的焦距对应的水平视场角的正切值
/// @param tan_fovy 
/// @param cov3D 三维协方差矩阵，用于计算二维协方差矩阵
/// @param viewmatrix 视图矩阵，用于将世界坐标系转换为相机坐标系
/// @return 三个元素的 float3 结构，代表了计算得到的二维协方差矩阵
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	// 将3D点通过视图矩阵变换到相机空间
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// 视锥体到裁剪空间的jacobian矩阵
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	// 构建从视锥体到裁剪空间的雅可比矩阵
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	// 3d高斯协方差
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	// 2d高斯协方差矩阵计算
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.

/// @brief 通过世界坐标系下3D高斯的旋转和缩放，计算3d协方差矩阵，也确保四元数归一化
/// @param scale 缩放因子
/// @param mod 缩放修正因子
/// @param rot 旋转四元数
/// @param cov3D 输出3d协方差矩阵
/// @return 
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	// 存储对称协方差矩阵的上三角部分
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
// 对每个高斯进行光栅化前的初始步骤
// P：高斯数量。
// D：球谐函数的最高阶数。
// M：球谐函数的最大系数数量。
// orig_points：原始点的坐标数组。
// scales：每个高斯的尺度向量数组。
// scale_modifier：尺度修正因子。
// rotations：每个高斯的旋转四元数数组。
// opacities：每个高斯的不透明度数组。
// shs：每个高斯的球谐系数数组。
// clamped：用于记录RGB颜色是否被截断的布尔数组。
// cov3D_precomp：预计算的每个高斯的三维协方差矩阵数组。
// colors_precomp：预计算的每个高斯的颜色数组。
// viewmatrix：视图矩阵。
// projmatrix：投影矩阵。
// cam_pos：摄像机位置。
// W、H：输出图像的宽度和高度。
// tan_fovx、tan_fovy：水平和垂直方向上的视场角的正切值。
// focal_x、focal_y：焦距。
// radii：每个高斯的半径数组。
// points_xy_image：每个高斯在图像上的坐标数组。
// depths：每个高斯的深度数组。
// cov3Ds：每个高斯的三维协方差矩阵数组。
// rgb：RGB颜色数组。
// conic_opacity：用于光栅化的椎体和不透明度数组。
// grid：二维线程块数量。
// tiles_touched：记录每个高斯覆盖的图像块数量的数组。
// prefiltered：指示是否对输入进行了预过滤的布尔值

template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	// 获取当前线程的全局索引，如果超出点数则返回
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	// 执行视锥体剔除
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] }; // 获取原始点坐标
	float4 p_hom = transformPoint4x4(p_orig, projmatrix); // 通过投影变换到裁剪空间
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w }; // 转为归一化设备坐标

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	// 如果提供了预计算的3D协方差则使用它，否则计算3D协方差
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	// 计算2D协方差矩阵
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f; // 添加小方差以提高数值稳定性
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	// 如果启用抗锯齿，计算缩放因子
	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov; // 计算2d协方差矩阵的行列式

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	// 计算协方差矩阵的逆（圆锥参数）
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	// 通过查找2d协方差矩阵特征值，计算2d高斯半径
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2))); // 计算半径（3倍标准差）

	// 计算与该2d高斯重叠的图像边界矩形rect
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) }; // 将投影点转换为像素坐标
	uint2 rect_min, rect_max;
	// 计算包围矩形
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	// 如果矩形覆盖0则返回
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.

	// 如果已预先计算颜色，请使用它们，否则将球谐系数转换为 RGB 颜色
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx];

	// 2d协方差的逆与不透明度打包到float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };

	// 2d高斯分布对应图块大小
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// 3D-2D光栅化主流程，在每个线程块上block协同处理一个tile, 每个线程处理一个像素，在获取与光栅化数据之间交替进行

// 声明了一个名为renderCUDA的CUDA核函数，具有模板参数CHANNELS，代表输出的颜色通道数

// 使用CUDA启动限制，设置了每个线程块的最大线程数量为BLOCK_X * BLOCK_Y

// const uint2* __restrict__ ranges, // 表示每个线程块要处理的点的范围
// const uint32_t* __restrict__ point_list, // 表示点的列表
// int W, int H, // 图像的宽度和高度
// const float2* __restrict__ points_xy_image, // 表示点在图像上的位置坐标
// const float* __restrict__ features, // 表示点的颜色
// const float4* __restrict__ conic_opacity, // 表示点的2d协方差逆和不透明度
// float* __restrict__ final_T, // 每个像素最终的不透明度
// uint32_t* __restrict__ n_contrib, // 每个像素其有贡献的高斯模型的数量
// const float* __restrict__ bg_color, // 背景颜色
// float* __restrict__ out_color, // 每个像素最终输出的颜色
// const float* __restrict__ depths,
// float* __restrict__ invdepth

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
	// 这是一个CUDA核函数模板，用于执行实际的光栅化渲染
	// printf("7. renderCUDA");
	// Identify current tile and associated min/max pixel range.
	// 确定当前图块和关联的最小/最大像素范围
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	// 在w位排序列表中加载要处理的id的开始/结束范围
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data. 为批量获取的数据进行保存
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f; // 该点未被阻挡的概率
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete 迭代处理所有点批次
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch 集体获取当前批次的高斯点数据
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			// 使用圆锥矩阵重采样
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// 透过乘以高斯不透明度与相对于平均值的指数衰减来获得 alpha (高斯球本身的不透明度)
			// 对每个高斯点计算其对当前像素的贡献，使用圆锥矩阵重采样
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			// 累积颜色和逆深度贡献
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T; // 颜色 = 球谐函数计算出的颜色*高斯球本身不透明度*该点未被阻挡的概率

			// 逆深度的计算
			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			// 更新该像素下一个高斯球未被阻挡的概率
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	// 更新累积不透明度
	if (inside)
	{
		// 得到该像素最终的不透明度
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch]; // 最后使用背景色填充,输出最终颜色

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

// 执行渲染流程，执行上面的renderCUDA

// const dim3 grid, // CUDA的网格维度,每个网格包含一组线程块
// dim3 block, // CUDA线程块的维度。线程块是一组并行执行的线程集合
// const uint2* ranges, // 每个线程块需要处理的像素范围
// const uint32_t* point_list, // 每个高斯模型的索引
// int W, int H, // 
// const float2* means2D, // 每个2d高斯的中心
// const float* colors, // 每个2d高斯的颜色
// const float4* conic_opacity, // 每个高斯模型的协方差矩阵逆和不透明度
// float* final_T, // 每个像素的不透明度
// uint32_t* n_contrib, // 每个像素其有贡献的高斯模型的数量
// const float* bg_color, // 背景颜色
// float* out_color, // 最终输出颜色

void FORWARD::render(
	const dim3 grid,
	dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	// 调用 renderCUDA 核函数执行渲染
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	// 调用 preprocessCUDA 核函数执行预处理
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}
