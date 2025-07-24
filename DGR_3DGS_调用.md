- [DGR 3dgs 调用关系](#dgr-3dgs-调用关系)
  - [PYBIND11 绑定cuda函数与python对象](#pybind11-绑定cuda函数与python对象)
  - [3dgs前向传播](#3dgs前向传播)
  - [继承`torch.nn.Module`](#继承torchnnmodule)
  - [继承`torch.autograd.Function`](#继承torchautogradfunction)
  - [3dgs 反向传播](#3dgs-反向传播)

# DGR 3dgs 调用关系

## PYBIND11 绑定cuda函数与python对象

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}
```
## 3dgs前向传播

```python

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    #......
    # 创建高斯光栅化实例
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 在这里自动进行前向传播
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    #......

```

## 继承`torch.nn.Module`

1. 在`__init__.py`继承`nn.Module`实现了`GaussianRasterizer`,并重新实现构造函数`__init__`和`forward`这两个方法,当调用`GaussianRasterizer(x1,x2,x3...)`时,内部会调用`__call__`方法，而`__call__`方法内部会调用子类的`forward`方法进行前向传播
2. 当3dgs调用`GaussianRasterizer(x1,x2,x3...)`时就会进行前向传播，调用`def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None)`
3. 最后调用`def rasterize_gaussians(means3D,means2D,sh,colors_precomp,opacities,scales,rotations,cov3Ds_precomp,raster_settings,)`
4. 执行`_RasterizeGaussians.apply()`

## 继承`torch.autograd.Function`

`class _RasterizeGaussians(torch.autograd.Function)` 继承了`torch.autograd.Function`,重写了`forward` `backward` 两个静态方法

- 因为我们自定义了前向传播`forward`和反向传播`backward`函数，在`apply`后，自动求导就不起作用
- 调用了`_RasterizeGaussians.apply()`,会调用这里的`forward`函数-->`_C.rasterize_gaussians(*args)`这时候，与`rasterize_gaussians`绑定的`RasterizeGaussiansCUDA`函数就起作用-->最后到`CudaRasterizer::Rasterizer::forward()`

## 3dgs 反向传播

3dgs中反向传播在`loss.backward()`体现，此时会`class _RasterizeGaussians(torch.autograd.Function)`里面的`backward`静态函数-->`_C.rasterize_gaussians_backward`-->关联`PYBIND11_MODULE`中的`RasterizeGaussiansBackwardCUDA`-->`CudaRasterizer::Rasterizer::backward`
