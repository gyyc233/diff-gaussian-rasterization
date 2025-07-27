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

# 申明和定义一些gaussian_rasterization的接口类和函数
# 作为pytorch和CUDA之间的API接口作用

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    """
    将输入元组input_tuple中的 PyTorch 张量深度复制到 CPU 上
    """
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    # print("3. rasterize_gaussians _RasterizeGaussians.apply")
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    """
    自定义的autograd自动求导函数，实现高斯渲染的前向传播与后向传播
    """
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        """
        前向传播，用 _C.rasterize_gaussians
        """

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        # print("4. staticmethod forward")
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):
        """
        后向传播,用 _C.rasterize_gaussians_backward(*args) 计算相关梯度
        后向传播被封装到这个静态方法中
        """

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_depth, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        # 后向传播调用处
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)        

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    """
    存储高斯渲染器的设置参数,包括图像的尺寸、焦距、背景张量、缩放修正因子、视图矩阵、投影矩阵、球谐函数阶数、相机位置以及调试模式
    """
    image_height: int
    image_width: int 
    tanfovx : float # X轴的焦距（tan）
    tanfovy : float # Y轴的焦距（tan）
    bg : torch.Tensor # 背景张量
    scale_modifier : float # 缩放修正因子
    viewmatrix : torch.Tensor # 观测矩阵
    projmatrix : torch.Tensor # 投影矩阵
    sh_degree : int # sh 阶数
    campos : torch.Tensor # 相机位置
    prefiltered : bool #　是否进行预过滤
    debug : bool # 调试模式
    antialiasing : bool # 抗锯齿

# 在使用pytorch的时候，模型训练时，不需要使用forward，只要在实例化一个对象中传入对应的参数就可以自动调用 forward 函数
# y = model(x)是调用了对象model的__call__方法，而nn.Module把__call__方法实现为类对象的forward函数
# 执行y = model(x)时，由于GaussianRasterizer类继承了Module类，而Module这个基类中定义了__call__方法，所以会执行__call__方法，而__call__方法中调用了forward()方法
# 当执行model(x)的时候，底层自动调用forward方法计算结果
class GaussianRasterizer(nn.Module):
    """
    实现高斯渲染器相关功能，包括可见性标记与前向渲染
    通过继承自 nn.Module 类，可以利用 PyTorch 的自动求导功能进行梯度计算和优化
    """
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings
        # print("1. GaussianRasterizer(nn.Module) ")

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        # 基于视锥体剔除原理，标记可见点
        with torch.no_grad():
            # 在这里禁用了自动梯度求导
            print('torch.no_grad')
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        # 实现了模型的前向传播逻辑
        # 检查输入参数的合法性，并根据需要填充缺失的参数
        # 调用 C++/CUDA 的渲染器函数 rasterize_gaussians 进行高斯渲染
        
        raster_settings = self.raster_settings
        # print("2. GaussianRasterizer(nn.Module) forward")

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

