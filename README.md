## CG-lab 图形学实验

## work0 环境搭建

1. 目标：学习如何下载IDEtrae，利用uv管理项目，git仓库的使用。最后构建出粒子物理模拟系统。
2. 架构：config.py数据集中定义，physics.py物理计算，main.py主程序（粒子受鼠标吸引）
3. 结果：**仿真结果**  <br>![仿真结果](https://gitee.com/feather-mu/images/raw/master/work0.gif)  <br>  **GPU使用情况**  <br>![GPU使用](https://gitee.com/feather-mu/images/raw/master/work0gpu.png)

## work1 旋转与变换

1. 目标：独立推导并用代码实现模型变换（Model）、视图变换（View）和投影变换（Projection）矩阵。掌握面向数据编程框架 Taichi 的基本语法与矩阵操作。
   
   掌握旋转插值操作。

2. 架构：main.py中先接受一个旋转角，然后返回其绕z轴旋转的变换矩阵，然后对设置相机位置为原点，面向-z方向，最后先将透视平截头体挤压为正交立方体，然后对其进行正交投影至屏幕。
   
   cube.py的大致流程一样，注意由于边长变大，相机现在平移至(0,0,5)的位置。
   
   spin.py先将欧拉角转化为四元数，然后应用slerp，计算旋转插值，最后计算出旋转矩阵，用MVP计算出屏幕坐标。

3. 结果
   
   **main.py**
   
   ![](https://gitee.com/feather-mu/images/raw/master/wok1main.gif)
   
   **cube.py**
   
   ![](https://gitee.com/feather-mu/images/raw/master/work1cube.gif)**spin.py**
   
   ![](https://gitee.com/feather-mu/images/raw/master/work1spin.gif)
