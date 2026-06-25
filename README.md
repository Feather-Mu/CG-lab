## CG-lab 计算机图形学实验

## work0 环境配置与基础测试

1. 实验内容：配置开发环境（IDE、Trae、uv等），配置git，测试GPU环境，运行基础物理模拟。
2. 架构说明：config.py定义参数，physics.py实现物理计算，main.py负责渲染和交互。
3. 结果展示：**基础物理模拟** <br>![基础物理模拟](https://gitee.com/feather-mu/images/raw/master/work0.gif) <br> **GPU加速测试** <br>![GPU加速测试](https://gitee.com/feather-mu/images/raw/master/work0gpu.png)

## work1 三维变换与动画

1. 实验内容：理解Model、View、Projection变换，实现立方体旋转、摄像机轨道控制、球面线性插值（slerp）动画。
2. 架构说明：main.py实现MVP变换和摄像机控制，cube.py绘制立方体，spin.py实现slerp动画。
3. 结果展示：**main.py** <br>![](https://gitee.com/feather-mu/images/raw/master/wok1main.gif) <br> **cube.py** <br>![](https://gitee.com/feather-mu/images/raw/master/work1cube.gif) <br> **spin.py** <br>![](https://gitee.com/feather-mu/images/raw/master/work1spin.gif)

## work2 Bezier曲线绘制

1. 实验内容：使用De Casteljau算法绘制Bezier曲线，实现控制点交互。
2. 架构说明：main.py实现交互界面，expand.py实现De Casteljau算法，使用GPU加速计算。
3. 结果展示：**Bezier曲线绘制** <br>![](https://gitee.com/feather-mu/images/raw/master/work2bezier-curve.gif) <br> **3次Bezier曲线** 
   
   ![](https://gitee.com/feather-mu/images/raw/master/B样条.gif)

## work3 光线投射与着色

1. 实验内容：使用Ray Casting实现球体和圆锥的渲染，实现Blinn-Phong着色模型和硬阴影。
2. 架构说明：
   - **几何体定义**：在Taichi Kernel中隐式定义球体和圆锥
   - **光线求交**：计算射线与几何体的交点距离
   - **深度测试**：使用Z-buffer确保正确的遮挡关系
   - **Blinn-Phong着色**：使用半程向量H替代反射向量R
   - **硬阴影**：发射Shadow Ray检测阴影
   - **UI交互**：4个滑动条控制Ka、Kd、Ks、Shininess参数
3. 结果展示：**渲染结果** <br>![](https://gitee.com/feather-mu/images/raw/master/Phong.gif)

## work4 光线追踪与反射

1. 实验内容：实现基于迭代的光线追踪系统，支持镜面反射和硬阴影。
2. 架构说明：
   - **场景构建**：地面平面（棋盘格纹理）、红色漫反射球、银色镜面球
   - **材质ID系统**：MAT_DIFFUSE和MAT_MIRROR区分材质
   - **光线弹射循环**：使用for循环实现最多3次弹射，throughput记录衰减
   - **硬阴影**：发射Shadow Ray检测遮挡
   - **精度处理**：反射射线和暗影射线起点偏移1e-4，避免Shadow Acne
   - **UI交互**：Light X/Y/Z控制光源位置，Max Bounces控制弹射次数
3. 结果展示：**渲染结果** <br>![](https://gitee.com/feather-mu/images/raw/master/ray-tracing.gif)

## work5 可微光栅化与网格优化

1. 实验目标：理解可微光栅化原理，掌握通过多视角剪影图像反推三维网格顶点坐标的方法，理解正则化在防止拓扑崩坏中的作用。
2. 架构说明：
   - **软光栅化**：使用PyTorch3D的SoftSilhouetteShader，通过Sigmoid函数实现边界平滑过渡，解决梯度消失问题
   - **多视角渲染**：设置20个均匀分布的摄像机视角，渲染目标奶牛的剪影图
   - **可微优化**：从细分球体开始，使用SGD优化器更新顶点偏移量
   - **三种正则化损失**：
     - 拉普拉斯平滑 (Laplacian Smoothing)：防止表面尖锐突起
     - 边长一致性 (Edge Length Penalty)：防止三角形严重拉伸
     - 法线一致性 (Normal Consistency)：保持表面平滑
3. 结果展示：**优化过程** 
   
   ![](https://gitee.com/feather-mu/images/raw/master/mesh.png)

## work6 质点-弹簧布料模拟

1. 实验目标：掌握动态场景渲染、质点-弹簧模型、三种数值积分方法的实现与对比，理解GPU并行计算基础。
2. 架构说明：
   - **质点-弹簧系统**：20x20网格布料，包含结构弹簧、剪切弹簧、弯曲弹簧
   - **力学计算**：重力、阻尼力、弹簧力（胡克定律），使用ti.atomic_add避免多线程冲突
   - **三种积分求解器**：
     - 显式欧拉 (Explicit Euler)：先更新位置，再更新速度（不稳定，易爆炸）
     - 半隐式欧拉 (Semi-Implicit Euler)：先更新速度，再更新位置（较稳定）
     - 隐式欧拉 (Implicit Euler)：使用定点迭代法近似求解（最稳定）
   - **交互功能**：鼠标拖拽布料、摄像机轨道控制、GGUI控制面板
   - **GPU加速**：使用Taichi的@ti.kernel和@ti.func实现并行计算
3. 结果展示：**三种积分方法对比** 
   
   ![](https://gitee.com/feather-mu/images/raw/master/3mode.gif)
   
     **交互式布料模拟** 
   
   ![](https://gitee.com/feather-mu/images/raw/master/drag.gif)