#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

/// <summary>
/// 表示直线检测时允许通过的边缘极性。
/// </summary>
enum class LineEdgePolarity {
    Positive = 0,
    Negative = 1,
    Any = 2,
};

/// <summary>
/// 表示每条卡尺线上候选边缘类型的选择方式。
/// </summary>
enum class LineSelectionMode {
    Strongest = 0,
    First = 1,
    Last = 2,
};

/// <summary>
/// 表示直线检测在ROI内部的扫描方向。
/// </summary>
enum class LineScanDirection {
    LeftToRight = 0,
    TopToBottom = 1,
    BottomToTop = 2,
    RightToLeft = 3,
};

/// <summary>
/// 表示检测点拟合为直线时使用的方式。
/// </summary>
enum class LineFitMode {
    LeastSquares = 0,
    RobustHuber = 1,
    RobustWelsch = 2,
    WeightTukey = 3,
};

/// <summary>
/// 初始化直线检测参数默认值。
/// </summary>
/// <param name="edgeThreshold">边缘阈值。</param>
/// <param name="caliperCount">卡尺数量。</param>
/// <param name="sampleStep">边缘扫描方向的采样步长。（步长单位像素）</param>
/// <param name="filterSize">滤波平滑尺寸。</param>
/// <param name="excludPts">剔除点数。</param>
/// <param name="edgePolarity">边缘极性。</param>
/// <param name="selectionMode">边缘类型选择。</param>
/// <param name="scanDirection">卡尺内边缘扫描方向。</param>
/// <param name="fitMode">拟合模式。</param>
struct LineDetectionParams {
    int edgeThreshold = 8;
    int caliperCount = 10;
    int sampleStep = 2;
    int filterSize = 3;
    int excludPts = 0;
    LineEdgePolarity edgePolarity = LineEdgePolarity::Any;
    LineSelectionMode selectionMode = LineSelectionMode::Strongest;
    LineScanDirection scanDirection = LineScanDirection::TopToBottom;
    LineFitMode fitMode = LineFitMode::LeastSquares;
};

/// <summary>
/// 初始化直线检测测量框。
/// </summary>
/// <param name="center">测量框中心点。</param>
/// <param name="width">测量框宽度。</param>
/// <param name="height">测量框高度。</param>
/// <param name="angleDegrees">测量框旋转角度。</param>
struct LineDetectionFrame {
    cv::Point2f center = cv::Point2f(0.0f, 0.0f);
    float width = 1.0f;
    float height = 1.0f;
    float angleDegrees = 0.0f;

    bool IsValid() const {
        return width > 0.0f && height > 0.0f;
    }

    cv::Point2f XDirection() const;
    cv::Point2f YDirection() const;
    cv::Point2f GetScanDirection(LineScanDirection direction) const;
    cv::Point2f GetArrangeDirection(LineScanDirection direction) const;
    float GetScanLength(LineScanDirection direction) const;
    float GetArrangeLength(LineScanDirection direction) const;
    std::vector<cv::Point2f> GetCorners() const;
};

/// <summary>
/// 初始化直线检测点。
/// </summary>
/// <param name="point">检测点图像坐标。</param>
/// <param name="strength">检测点边缘强度。</param>
struct LineEdgePoint {
    cv::Point2f point = cv::Point2f(0.0f, 0.0f);
    float strength = 0.0f;
};

struct LineCaliper {
    cv::Point2f center = cv::Point2f(0.0f, 0.0f);
    cv::Point2f scanDirection = cv::Point2f(1.0f, 0.0f);
    cv::Point2f arrangeDirection = cv::Point2f(0.0f, 1.0f);
    float scanLength = 1.0f;
    float width = 1.0f;
};

/// <summary>
/// 初始化直线检测结果。
/// </summary>
/// <param name="success">是否检测成功。</param>
/// <param name="message">结果消息。</param>
/// <param name="frame">检测测量框。</param>
/// <param name="scanDirection">扫描方向。</param>
/// <param name="lineStart">结果线段起点。</param>
/// <param name="lineEnd">结果线段终点。</param>
/// <param name="angleDegrees">检测直线角度，单位为度。</param>
/// <param name="fitError">参与拟合点到结果直线的平均绝对距离误差。</param>
/// <param name="fitErrorRmse">参与拟合点到结果直线的均方根距离误差。</param>
/// <param name="averageStrength">参与拟合点的平均边缘强度。</param>
/// <param name="maxStrength">参与拟合点中的最大边缘强度。</param>
/// <param name="elapsedMilliseconds">检测耗时，单位为毫秒。</param>
/// <param name="rawPointCount">离群点剔除前的原始检测点数量。</param>
/// <param name="fitPointCount">实际参与最终直线拟合的点数量。</param>
/// <param name="excludedPointCount">被判定为离群点并剔除的点数量。</param>
/// <param name="edgePoints">参与最终拟合的检测点集合。</param>
/// <param name="excludedEdgePoints">被剔除的离群检测点集合。</param>
struct LineDetectionResult {
    bool success = false;
    std::string message;
    LineDetectionFrame frame;
    LineScanDirection scanDirection = LineScanDirection::LeftToRight;
    cv::Point2f lineStart = cv::Point2f(0.0f, 0.0f);
    cv::Point2f lineEnd = cv::Point2f(0.0f, 0.0f);
    float angleDegrees = 0.0f;
    float fitError = 0.0f;
    float fitErrorRmse = 0.0f;
    float averageStrength = 0.0f;
    float maxStrength = 0.0f;
    double elapsedMilliseconds = 0.0;
    int rawPointCount = 0;
    int fitPointCount = 0;
    int excludedPointCount = 0;
    std::vector<LineEdgePoint> edgePoints;
    std::vector<LineEdgePoint> excludedEdgePoints;
};

std::string ToDisplayString(LineEdgePolarity polarity);
std::string ToDisplayString(LineSelectionMode selectionMode);
std::string ToDisplayString(LineScanDirection scanDirection);
std::string ToDisplayString(LineFitMode fitMode);
