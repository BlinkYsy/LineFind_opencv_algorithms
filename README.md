# LineFind OpenCV Algorithms

基于 **OpenCV 4.4.0** 与 C++ 的交互式直线检测演示项目。程序通过卡尺（Caliper）在图像 ROI 内采样边缘点，再拟合出目标直线；界面支持创建和调整 ROI、调节检测参数、查看检测点与结果线。

## 功能

- 支持打开本地图片及切换示例图片。
- 支持两种检测区域：箭头卡尺组和旋转矩形 ROI。
- 支持拖拽、缩放检测区域，以及图像缩放与平移。
- 支持调节卡尺数量、采样步长、滤波尺寸、边缘阈值、边缘极性、候选点选择方式、扫描方向、拟合模式和外点剔除数量。
- 在画布中显示卡尺预览、有效/剔除边缘点和拟合直线结果。
- 支持最小二乘、Huber、Welsch 与 Tukey 加权四种直线拟合方式。

## 算法流程

```text
输入图像
  → 灰度化
  → 根据 ROI 或箭头卡尺生成多个卡尺
  → 沿扫描方向采样，并在卡尺宽度方向做灰度平均
  → 从一维灰度剖面提取满足阈值与极性的边缘候选点
  → 按“最强 / 第一个 / 最后一个”选择每个卡尺的边缘点
  → 可选外点剔除
  → 直线拟合
  → 显示边缘点和结果线段
```

核心检测入口为 `LineDetectionOperator`，其输入可以是旋转矩形 `LineDetectionFrame`，也可以是一组 `LineCaliper`。检测结果包含拟合线段、角度、误差、边缘强度、耗时，以及保留和剔除的边缘点。

## 目录结构

```text
.
├─ Linefind_demo/                 # 直线检测 Demo 源码与 Visual Studio 项目
│  ├─ demo_canvas.*               # OpenCV 交互式画布与 ROI 操作
│  ├─ line_detection_operator.*   # 卡尺采样、边缘提取与直线拟合
│  ├─ line_detection_types.h      # 参数、ROI、卡尺和结果数据结构
│  └─ main.cpp                    # Demo 入口
├─ testImage/                     # 示例图片
├─ dist/Linefind_demo-x64/        # 已编译的 x64 发布版
│  ├─ Linefind_demo.exe
│  └─ opencv_world440.dll
└─ opencv_Linefind.sln            # Visual Studio 解决方案
```

## 编译环境

- Windows 10/11 x64
- Visual Studio 2022（C++ 桌面开发工作负载，v143 工具集）
- C++17 或更高版本编译器
- OpenCV 4.4.0 x64

`Linefind_demo/Linefind_demo.vcxproj` 当前配置使用 OpenCV 4.4.0 的头文件、库目录和 `opencv_world440.lib`。在其他电脑上编译时，请将项目属性中的 OpenCV 包含目录、库目录和 DLL 复制规则改为本机实际安装位置。

## 构建与运行

1. 使用 Visual Studio 2022 打开 `Linefind_demo/Linefind_demo.vcxproj`。
2. 选择 `Release | x64`。
3. 配置本机 OpenCV 4.4.0 x64 的 include、lib 和 bin 路径。
4. 生成并运行 `Linefind_demo`。

程序启动时会读取 `Linefind_demo/main.cpp` 中定义的示例图列表。该列表目前使用开发机的绝对路径：

```cpp
D:\Visual Studio 2022\LineFind_algorithms\testImage\...
```

因此，克隆到其他目录或其他电脑后，请先把这些路径改为本地 `testImage/` 目录的实际路径，再重新编译运行。

## 已编译发布版

x64 发布版位于 [`dist/Linefind_demo-x64/`](dist/Linefind_demo-x64/)：

- `Linefind_demo.exe`
- `opencv_world440.dll`

运行该发布版需要 Windows x64，并安装 [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)。发布目录未包含 VC++ 运行库 DLL。

## 已知限制

- `opencv_Linefind.sln` 同时引用 `linedectect_master/opencv_linedectect.vcxproj`，但该项目目录未纳入本仓库；若只使用 Demo，请直接打开 `Linefind_demo.vcxproj`。
- 示例图片路径为绝对路径，发布版不能在未修改路径的其他电脑上直接加载这些内置示例图。
- 当前仓库未提供自动化测试项目。

## 后续优化方向

- 将示例图路径改为相对路径或命令行参数，便于分发和运行。
- 补充检测质量指标，例如残差、内点比例和卡尺命中率。
- 增加对弱边缘、噪声与离群点场景的预处理和鲁棒性配置。
- 为卡尺采样、候选点选择和直线拟合补充自动化测试。
