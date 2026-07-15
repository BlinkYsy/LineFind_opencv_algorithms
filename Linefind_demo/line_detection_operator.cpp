#include "line_detection_operator.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

#include <opencv2/imgproc.hpp>

namespace {

constexpr float kPi = 3.14159265358979323846f;

cv::Point2f Normalize(const cv::Point2f& value) {
    const float norm = std::sqrt(value.dot(value));
    if (norm <= 1e-6f) {
        return cv::Point2f(1.0f, 0.0f);
    }

    return value * (1.0f / norm);
}

cv::Point2f Rotate90(const cv::Point2f& value) {
    return cv::Point2f(-value.y, value.x);
}

cv::Mat ToGray(const cv::Mat& image) {
    if (image.empty()) {
        return cv::Mat();
    }

    cv::Mat gray;
    if (image.channels() == 1) {
        image.convertTo(gray, CV_8U);
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        image.convertTo(gray, CV_8U);
    }

    return gray;
}

LineDetectionParams NormalizeParams(LineDetectionParams params) {
    params.edgeThreshold = std::max(0, params.edgeThreshold);
    params.caliperCount = std::max(2, params.caliperCount);
    params.sampleStep = std::max(1, params.sampleStep);
    params.filterSize = std::max(1, params.filterSize);
    params.excludPts = std::max(0, params.excludPts);
    if (params.filterSize % 2 == 0) {
        ++params.filterSize;
    }
    return params;
}

float CalculateAngleDegrees(const cv::Point2f& start, const cv::Point2f& end) {
    float angle = std::atan2(end.y - start.y, end.x - start.x) * 180.0f / kPi;
    if (angle < 0.0f) {
        angle += 180.0f;
    }
    if (angle >= 180.0f) {
        angle -= 180.0f;
    }
    return angle;
}

//拟合误差的计算公式主要包括均方误差（MSE）、均方根误差（RMSE）、平均绝对误差（MAE）和R平方（R²）
//计算一组边缘点 edgePoints 到线段 start -> end 的平均绝对误差（MAE）作为 拟合误差
float CalculateFitError(
    const std::vector<LineEdgePoint>& edgePoints,
    const cv::Point2f& start,
    const cv::Point2f& end) {
    const cv::Point2f direction = end - start;
    const float length = std::sqrt(direction.dot(direction));
    if (edgePoints.empty() || length <= 1e-6f) {
        return 0.0f;
    }

    float sumDistance = 0.0f;
    //当前点到由 start -> end 确定的直线的垂直距离
    for (const LineEdgePoint& edgePoint : edgePoints) {
        sumDistance += static_cast<float>(std::fabs((edgePoint.point - start).cross(direction)) / length);
    }

    return sumDistance / static_cast<float>(edgePoints.size());
}

//计算一组边缘点 edgePoints 到线段 start -> end 的均方根误差（RMSE）作为 拟合误差
float CalculateFitErrorRmse(
    const std::vector<LineEdgePoint>& edgePoints,
    const cv::Point2f& start,
    const cv::Point2f& end) {
    const cv::Point2f direction = end - start;
    const float length = std::sqrt(direction.dot(direction));
    if (edgePoints.empty() || length <= 1e-6f) {
        return 0.0f;
    }

    float sumSquaredDistance = 0.0f;
    for (const LineEdgePoint& edgePoint : edgePoints) {
        const float distance = static_cast<float>(std::fabs((edgePoint.point - start).cross(direction)) / length);
        sumSquaredDistance += distance * distance;
    }

    return std::sqrt(sumSquaredDistance / static_cast<float>(edgePoints.size()));
}

float CalculatePointDistanceToLine(
    const LineEdgePoint& edgePoint,
    const cv::Point2f& start,
    const cv::Point2f& end) {
    const cv::Point2f direction = end - start;
    const float length = std::sqrt(direction.dot(direction));
    if (length <= 1e-6f) {
        return 0.0f;
    }

    return static_cast<float>(std::fabs((edgePoint.point - start).cross(direction)) / length);
}

//根据梯度正负号来筛边缘极性。
bool MatchPolarity(double gradient, LineEdgePolarity polarity) {
    switch (polarity) {
    case LineEdgePolarity::Positive:
        return gradient > 0.0;
    case LineEdgePolarity::Negative:
        return gradient < 0.0;
    case LineEdgePolarity::Any:
    default:
        return true;
    }
}

/// <summary>
/// 卡尺检测区域的几何信息。
/// </summary>
/// <param name="center">卡尺区域中心。</param>
/// <param name="direction">卡尺长度轴方向的单位向量。(与scanDirection方向一致)</param>
/// <param name="perpendicular">卡尺宽度轴方向的单位向量。(与arrangeDirection方向一致)</param>
/// <param name="length">卡尺长度。</param>
/// <param name="width">卡尺宽度。</param>
struct CaliperInfo {
    cv::Point2f center;
    cv::Point2f direction;
    cv::Point2f perpendicular;
    float length = 0.0f;
    float width = 0.0f;
};

struct CandidateEdgePoint {
    cv::Point2f position;
    float strength = 0.0f;
    bool isValid = false;
};

struct CaliperProfile {
    std::vector<double> values;
    std::vector<float> sampleOffsets;
};

/// <summary>
/// 根据旋转ROI和检测参数，沿卡尺排列方向生成等间距卡尺。
/// </summary>
/// <param name="frame">旋转ROI测量框。</param>
/// <param name="params">直线检测参数。</param>
/// <returns>用于边缘扫描的卡尺几何信息集合。</returns>
std::vector<CaliperInfo> GenerateCalipers(
    const LineDetectionFrame& frame,
    const LineDetectionParams& params) 
{
    const cv::Point2f arrangeDir = Normalize(frame.GetArrangeDirection(params.scanDirection));
    const cv::Point2f scanDir = Normalize(frame.GetScanDirection(params.scanDirection));
    const float arrangeLength = frame.GetArrangeLength(params.scanDirection);
    const float scanLength = frame.GetScanLength(params.scanDirection);
    const float halfArrange = arrangeLength * 0.5f;
    const cv::Point2f segmentStart = frame.center - arrangeDir * halfArrange;
    const float spacing = arrangeLength / static_cast<float>(params.caliperCount);
    const float caliperWidth = std::max(1.0f, arrangeLength / static_cast<float>(params.caliperCount));

    std::vector<CaliperInfo> calipers;
    calipers.reserve(params.caliperCount);
    for (int index = 0; index < params.caliperCount; ++index) {
        const float offset = (static_cast<float>(index) + 0.5f) * spacing;
        CaliperInfo caliper;
        caliper.center = segmentStart + arrangeDir * offset;
        caliper.direction = scanDir;
        caliper.perpendicular = arrangeDir;
        caliper.length = scanLength;
        caliper.width = caliperWidth;
        calipers.push_back(caliper);
    }

    return calipers;
}

/// <summary>
/// 将外部定义的箭头卡尺转换为内部统一的卡尺几何信息。
/// </summary>
/// <param name="calipers">箭头卡尺集合。</param>
/// <returns>内部使用的卡尺几何信息集合。</returns>
std::vector<CaliperInfo> ConvertCalipers(const std::vector<LineCaliper>& calipers) {
    std::vector<CaliperInfo> converted;
    converted.reserve(calipers.size());
    for (const LineCaliper& caliper : calipers) {
        CaliperInfo info;
        info.center = caliper.center;
        info.direction = Normalize(caliper.scanDirection);
        info.perpendicular = Normalize(caliper.arrangeDirection);
        info.length = std::max(1.0f, caliper.scanLength);
        info.width = std::max(1.0f, caliper.width);
        converted.push_back(info);
    }
    return converted;
}

/// <summary>
/// 构建将图像中的旋转卡尺区域映射到局部矩形坐标系的仿射变换矩阵。
/// </summary>
/// <param name="caliper">卡尺几何信息。</param>
/// <param name="length">卡尺扫描方向的采样长度。</param>
/// <param name="width">卡尺排列方向的投影宽度。</param>
/// <returns>从图像坐标映射到卡尺局部坐标的仿射矩阵。</returns>
cv::Mat BuildTransformMatrix(const CaliperInfo& caliper, int length, int width) {
    const float halfLength = caliper.length * 0.5f;
    const float halfWidth = caliper.width * 0.5f;
    std::vector<cv::Point2f> srcPoints(3);
    srcPoints[0] = cv::Point2f(0.0f, 0.0f);
    srcPoints[1] = cv::Point2f(static_cast<float>(length - 1), 0.0f);
    srcPoints[2] = cv::Point2f(0.0f, static_cast<float>(width - 1));

    std::vector<cv::Point2f> dstPoints(3);
    dstPoints[0] = caliper.center - caliper.direction * halfLength - caliper.perpendicular * halfWidth;
    dstPoints[1] = caliper.center + caliper.direction * halfLength - caliper.perpendicular * halfWidth;
    dstPoints[2] = caliper.center - caliper.direction * halfLength + caliper.perpendicular * halfWidth;

    return cv::getAffineTransform(dstPoints, srcPoints);
}

//从灰度图里，按照一个卡尺区域 caliper 提取出一条一维灰度曲线（profile）
CaliperProfile ExtractCaliperProfile(
    const cv::Mat& grayImage,
    const CaliperInfo& caliper,
    float sampleStep) 
{
    const int length = std::max(1, cvRound(caliper.length));
    const int width = std::max(1, cvRound(caliper.width));
    //生成一个仿射变换矩阵，根据原图中“沿 caliper 方向摆放的旋转矩形区域”生成一个坐标变化规则 
    const cv::Mat transform = BuildTransformMatrix(caliper, length, width);
    cv::Mat croppedRegion;
    //把卡尺区域裁出来并拉正，按对应规则真正生成裁正后的卡尺小图croppedRegion
    cv::warpAffine(
        grayImage,
        croppedRegion,
        transform,
        cv::Size(length, width),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar::all(0));

    // ===== DEBUG: caliper extraction =====
    //std::cout << "[DEBUG] transform size = "
    //    << transform.rows << " x " << transform.cols
    //    << ", type = " << transform.type() << std::endl;
    //std::cout << "[DEBUG] transform =\n" << transform << std::endl;
    //std::cout << "[DEBUG] croppedRegion size = "
    //    << croppedRegion.rows << " x " << croppedRegion.cols
    //    << ", type = " << croppedRegion.type() << std::endl;
    //std::cout << "[DEBUG] croppedRegion =\n" << croppedRegion << std::endl;
    // ===== DEBUG END =====

    //profile 是沿 卡尺内的边缘扫描 方向展开
    CaliperProfile profile;
    const int step = std::max(1, cvRound(sampleStep));
    //profile 的采样点数
    const int sampleCount = std::max(1, (length - 1) / step + 1);
    //每个采样位置对应的平均灰度值
    profile.values.reserve(sampleCount);
    //每个采样点相对卡尺中心的偏移
    profile.sampleOffsets.reserve(sampleCount);

    for (int x = 0; x < length; x += step) {
        double sum = 0.0;
        for (int y = 0; y < width; ++y) {
            sum += static_cast<double>(croppedRegion.at<uchar>(y, x));
        }
        profile.values.push_back(sum / static_cast<double>(width));
        profile.sampleOffsets.push_back(static_cast<float>(x) - static_cast<float>(length - 1) * 0.5f);
    }
    //单个卡尺内部的平均灰度采样序列
    return profile;
}

//双线性插值采样
double BilinearSample(const cv::Mat& grayImage, float x, float y) {
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);

    const auto readPixel = [&](int pixelX, int pixelY) -> float {
        if (pixelX < 0 || pixelX >= grayImage.cols || pixelY < 0 || pixelY >= grayImage.rows) {
            return 0.0f;
        }
        return static_cast<float>(grayImage.at<uchar>(pixelY, pixelX));
    };
    //上边一行的两个像素之间横向插值
    const float top = readPixel(x0, y0) * (1.0f - fx) + readPixel(x1, y0) * fx;
    //下边一行的两个像素之间横向插值
    const float bottom = readPixel(x0, y1) * (1.0f - fx) + readPixel(x1, y1) * fx;
    //两个水平结果再进行纵向插值 -> 最终灰度值
    return static_cast<double>(top * (1.0f - fy) + bottom * fy);
}

//warpAffine变换 → 生成矩形卡尺图croppedRegion → 读取采样并平均  --->>>  根据卡尺坐标直接双线性采样 → 平均
//改进点/优势：减少临时内存分配；避免对整个卡尺区域进行仿射变换复杂处理；可直接在原图坐标系中进行亚像素采样
CaliperProfile ExtractCaliperProfile_new(
    const cv::Mat& grayImage,
    const CaliperInfo& caliper,
    float sampleStep) 
{
    const int length = std::max(1, cvRound(caliper.length));
    const int width = std::max(1, cvRound(caliper.width));
    const int step = std::max(1, cvRound(sampleStep));
    const int sampleCount = std::max(1, (length - 1) / step + 1);

    CaliperProfile profile;
    profile.values.reserve(sampleCount);
    profile.sampleOffsets.reserve(sampleCount);
    //卡尺矩形的左上角起始点
    const cv::Point2f start =caliper.center - caliper.direction * (caliper.length * 0.5f) - caliper.perpendicular * (caliper.width * 0.5f);
    //长度方向相邻两个采样点之间的实际坐标增量
    const cv::Point2f scanIncrement = length > 1
        ? caliper.direction * (caliper.length / static_cast<float>(length - 1)): cv::Point2f(0.0f, 0.0f);
    //宽度方向相邻两个采样点之间的实际坐标增量
    const cv::Point2f widthIncrement = width > 1
        ? caliper.perpendicular * (caliper.width / static_cast<float>(width - 1)): cv::Point2f(0.0f, 0.0f);

    for (int x = 0; x < length; x += step) {
        double sum = 0.0;
        const cv::Point2f scanStart = start + scanIncrement * static_cast<float>(x);
        for (int y = 0; y < width; ++y) {
            sum += BilinearSample(
                grayImage,
                scanStart.x + widthIncrement.x * static_cast<float>(y),
                scanStart.y + widthIncrement.y * static_cast<float>(y));
        }
        profile.values.push_back(sum / static_cast<double>(width));
        profile.sampleOffsets.push_back(static_cast<float>(x) - static_cast<float>(length - 1) * 0.5f);
    }

    return profile;
}

/// <summary>
/// 对一维灰度剖面 profile 执行均值滤波平滑。
/// </summary>
/// <param name="profile">原始灰度剖面 profile。</param>
/// <param name="filterSize">滤波窗口大小。</param>
/// <returns>平滑后的灰度剖面 profile。</returns>
std::vector<double> SmoothProfile(const std::vector<double>& profile, int filterSize) {
    if (profile.empty() || filterSize <= 1) {
        return profile;
    }

    cv::Mat source(static_cast<int>(profile.size()), 1, CV_64F);
    for (int index = 0; index < static_cast<int>(profile.size()); ++index) {
        source.at<double>(index, 0) = profile[index];
    }

    cv::Mat blurred;
    cv::blur(source, blurred, cv::Size(1, filterSize));

    std::vector<double> smoothed(profile.size(), 0.0);
    for (int index = 0; index < static_cast<int>(profile.size()); ++index) {
        smoothed[index] = blurred.at<double>(index, 0);
    }

    return smoothed;
}

/// <summary>
/// 在单个卡尺内提取灰度剖面 profile、计算梯度，并筛选满足阈值与极性的候选边缘点。
/// </summary>
/// <param name="grayImage">8位单通道灰度图像。</param>
/// <param name="caliper">待检测的卡尺。</param>
/// <param name="params">边缘阈值、极性、滤波和采样参数。</param>
/// <returns>当前卡尺内全部有效的候选边缘点。</returns>
std::vector<CandidateEdgePoint> FindEdgesInCaliper(
    const cv::Mat& grayImage,
    const CaliperInfo& caliper,
    const LineDetectionParams& params) {
    std::vector<CandidateEdgePoint> edges;
    //const CaliperProfile rawProfile = ExtractCaliperProfile(grayImage, caliper, params.sampleStep);
    const CaliperProfile rawProfile = ExtractCaliperProfile_new(grayImage, caliper, params.sampleStep);
    if (rawProfile.values.size() < 3) {
        return edges;
    }

    //对单个卡尺提取出来的 一维灰度曲线profile 做平滑滤波
    const std::vector<double> filterProfile = SmoothProfile(rawProfile.values, params.filterSize);

    // ===== DEBUG: profile and filterProfile =====
    //std::cout << "[DEBUG] profile.values = ";
    //for (double value : rawProfile.values) {
    //    std::cout << value << " ";
    //}
    //std::cout << std::endl;
    //std::cout << "[DEBUG] filtered profile = ";
    //for (double value : filterProfile) {
    //    std::cout << value << " ";
    //}
    //std::cout << std::endl;
    // ===== DEBUG END =====

    //对滤波后的 profile 进行梯度计算
    std::vector<double> gradient(filterProfile.size(), 0.0);
    for (int index = 1; index + 1 < static_cast<int>(filterProfile.size()); ++index) {
        //后向差分梯度 -> edgeThreshold 效范围变为 0-255
        gradient[index] = filterProfile[index] - filterProfile[index - 1];
        //中心差分梯度 -> edgeThreshold 效范围变为 0-127
        //gradient[index] = (profile[index + 1] - profile[index - 1]) / 2.0;
    }

    for (int index = 1; index + 1 < static_cast<int>(gradient.size()); ++index) {
        const double gradientStrength = std::fabs(gradient[index]);
        if (gradientStrength < params.edgeThreshold) {
            continue;
        }
        if (!MatchPolarity(gradient[index], params.edgePolarity)) {
            continue;
        }
        //是否满足局部极大值
        //if (std::fabs(gradient[index]) <= std::fabs(gradient[index - 1]) ||
        //    std::fabs(gradient[index]) <= std::fabs(gradient[index + 1])) {
        //    continue;
        //}

        //将一维位置映射回图像坐标（中心差分）
        //const float offset = rawProfile.sampleOffsets[index];
        //将一维位置映射回图像坐标（后向差分）
        const float offset =(rawProfile.sampleOffsets[index - 1] + rawProfile.sampleOffsets[index]) * 0.5f;
        CandidateEdgePoint point;
        point.position = caliper.center + caliper.direction * offset;
        point.strength = static_cast<float>(gradientStrength);
        point.isValid = true;
        edges.push_back(point);
    }

    return edges;
}

/// <summary>
/// 根据指定的选择模式，从单个卡尺的候选边缘点中选出一个检测点。
/// </summary>
/// <param name="edges">当前卡尺中的候选边缘点。</param>
/// <param name="selectionMode">首个、末个或最强边缘的选择方式。</param>
/// <returns>选中的候选点；无候选点时返回无效点。</returns>
CandidateEdgePoint SelectPoint(
    const std::vector<CandidateEdgePoint>& edges,
    LineSelectionMode selectionMode) 
{
    if (edges.empty()) {
        return CandidateEdgePoint();
    }

    switch (selectionMode) {
    case LineSelectionMode::First:
        return edges.front();
    case LineSelectionMode::Last:
        return edges.back();
    case LineSelectionMode::Strongest:
    default: {
        CandidateEdgePoint strongest = edges.front();
        for (const CandidateEdgePoint& edge : edges) {
            if (edge.strength > strongest.strength) {
                strongest = edge;
            }
        }
        return strongest;
    }
    }
}

/// <summary>
/// 使用最小二乘法拟合边缘点的直线 -> 尽量让所有点到直线的 总体误差平方和 最小。
/// </summary>
/// <param name="edgePoints">参与拟合的边缘点集合。</param>
/// <param name="start">输出拟合线段起点。</param>
/// <param name="end">输出拟合线段终点。</param>
/// <returns>至少存在两个有效点且拟合成功时返回true。</returns>
bool FitLeastSquares(
    const std::vector<LineEdgePoint>& edgePoints,
    cv::Point2f& start,
    cv::Point2f& end) {
    if (edgePoints.size() < 2) {
        return false;
    }

    const int count = static_cast<int>(edgePoints.size());
    double sumX = 0.0;
    double sumY = 0.0;
    double sumXY = 0.0;
    double sumX2 = 0.0;
    double sumY2 = 0.0;
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();

    //确定线段端点范围边界
    for (const LineEdgePoint& edgePoint : edgePoints) {
        const double x = edgePoint.point.x;
        const double y = edgePoint.point.y;
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumX2 += x * x;
        sumY2 += y * y;
        minX = std::min(minX, x);
        maxX = std::max(maxX, x);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
    }
    //denominator 衡量所有点在 x 方向上有没有足够变化（能不能正常按 y(x) 的形式拟合）
    const double denominator = count * sumX2 - sumX * sumX;
    if (std::fabs(denominator) < 1e-4) {
        //verticalDenominator 衡量所有点在 y 方向上有没有足够变化（能不能正常按 x(y) 的形式拟合）
        const double verticalDenominator = count * sumY2 - sumY * sumY;
        if (std::fabs(verticalDenominator) < 1e-4) {
            start = edgePoints.front().point;
            end = start;
            return true;
        }

        const double slope = (count * sumXY - sumX * sumY) / verticalDenominator;
        const double intercept = (sumX - slope * sumY) / count;
        start = cv::Point2f(static_cast<float>(slope * minY + intercept), static_cast<float>(minY));
        end = cv::Point2f(static_cast<float>(slope * maxY + intercept), static_cast<float>(maxY));
        return true;
    }
    //直线拟合的斜率和截距
    const double slope = (count * sumXY - sumX * sumY) / denominator;
    const double intercept = (sumY - slope * sumX) / count;
    //生成线段的两个端点
    start = cv::Point2f(static_cast<float>(minX), static_cast<float>(slope * minX + intercept));
    end = cv::Point2f(static_cast<float>(maxX), static_cast<float>(slope * maxX + intercept));
    return true;
}

/// <summary>
/// 使用OpenCV鲁棒距离函数拟合边缘点直线。(鲁棒性强度：DIST_L2 < DIST_HUBER < DIST_WELSCH)
/// </summary>
/// <param name="edgePoints">参与拟合的边缘点集合。</param>
/// <param name="start">输出拟合线段起点。</param>
/// <param name="end">输出拟合线段终点。</param>
/// <param name="fitMode">鲁棒拟合模式。</param>
/// <returns>拟合成功时返回true。</returns>
bool FitRobust(
    const std::vector<LineEdgePoint>& edgePoints,
    cv::Point2f& start,
    cv::Point2f& end,
    LineFitMode lineFitMode)
{
    if (edgePoints.size() < 2) {
        return false;
    }

    std::vector<cv::Point2f> points;
    points.reserve(edgePoints.size());
    float minProjection = std::numeric_limits<float>::max();
    float maxProjection = std::numeric_limits<float>::lowest();
    for (const LineEdgePoint& edgePoint : edgePoints) {
        points.push_back(edgePoint.point);
    }

    cv::Vec4f line;
    int distType = cv::DIST_HUBER;

    switch (lineFitMode)
    {
    case LineFitMode::RobustHuber:
        distType = cv::DIST_HUBER;//ρ(r) = r² / 2;ρ(r) = C * (r - C/2)
        break;

    case LineFitMode::RobustWelsch://DIST_WELSCH -> 对离群点进行降权惩罚 （将异常点影响被压小） 
        distType = cv::DIST_WELSCH;//ρ(r) = C²/2 * (1 - exp(-(r/C)²))
        break;

    default:
        return false;
    }
    cv::fitLine(points, line, distType, 0, 0.01, 0.01);
    const cv::Point2f direction = Normalize(cv::Point2f(line[0], line[1]));
    const cv::Point2f anchor(line[2], line[3]);

    for (const cv::Point2f& point : points) {
        const float projection = (point - anchor).dot(direction);
        minProjection = std::min(minProjection, projection);
        maxProjection = std::max(maxProjection, projection);
    }

    start = anchor + direction * minProjection;
    end = anchor + direction * maxProjection;
    return true;
}

/// <summary>
/// 先用普通最小二乘拟合出一条初始直线，再用 Tukey 权重做迭代加权拟合，让离群点影响越来越小，最后得到一条更稳的直线
/// </summary>
/// <param name="edgePoints">参与拟合的边缘点集合。</param>
/// <param name="start">输出拟合线段起点。</param>
/// <param name="end">输出拟合线段终点。</param>
/// <returns>拟合成功时返回true。</returns>
bool FitWeightTukey(
    const std::vector<LineEdgePoint>& edgePoints,
    cv::Point2f& start,
    cv::Point2f& end) {
    if (edgePoints.size() < 2) {
        return false;
    }

    cv::Point2f currentStart;
    cv::Point2f currentEnd;
    if (!FitLeastSquares(edgePoints, currentStart, currentEnd)) {
        return false;
    }

    constexpr float kTukeyC = 4.685f;
    constexpr int kMaxIterations = 20;
    constexpr float kMinScale = 1e-3f;
    constexpr float kConvergence = 1e-4f;

    std::vector<cv::Point2f> points;
    points.reserve(edgePoints.size());
    for (const LineEdgePoint& edgePoint : edgePoints) {
        points.push_back(edgePoint.point);
    }

    for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
        const cv::Point2f lineDirection = currentEnd - currentStart;
        const float lineLength = std::sqrt(lineDirection.dot(lineDirection));
        if (lineLength <= 1e-6f) {
            break;
        }

        std::vector<float> distances;
        distances.reserve(points.size());
        for (const cv::Point2f& point : points) {
            distances.push_back(static_cast<float>(std::fabs((point - currentStart).cross(lineDirection)) / lineLength));
        }

        std::vector<float> sortedDistances = distances;
        std::sort(sortedDistances.begin(), sortedDistances.end());
        const float medianDistance = sortedDistances[sortedDistances.size() / 2];
        const float scale = std::max(kMinScale, medianDistance * 1.4826f);

        double sumWeights = 0.0;
        cv::Point2d centroid(0.0, 0.0);
        std::vector<double> weights(points.size(), 0.0);
        for (size_t index = 0; index < points.size(); ++index) {
            const float normalizedResidual = distances[index] / (kTukeyC * scale);
            double weight = 0.0;
            if (normalizedResidual < 1.0f) {
                const double term = 1.0 - static_cast<double>(normalizedResidual * normalizedResidual);
                weight = term * term;
            }
            weights[index] = weight;
            sumWeights += weight;
            centroid.x += static_cast<double>(points[index].x) * weight;
            centroid.y += static_cast<double>(points[index].y) * weight;
        }

        if (sumWeights <= 1e-6) {
            break;
        }

        centroid.x /= sumWeights;
        centroid.y /= sumWeights;

        double sxx = 0.0;
        double sxy = 0.0;
        double syy = 0.0;
        for (size_t index = 0; index < points.size(); ++index) {
            const double dx = static_cast<double>(points[index].x) - centroid.x;
            const double dy = static_cast<double>(points[index].y) - centroid.y;
            sxx += weights[index] * dx * dx;
            sxy += weights[index] * dx * dy;
            syy += weights[index] * dy * dy;
        }

        const double theta = 0.5 * std::atan2(2.0 * sxy, sxx - syy);
        const cv::Point2f direction(static_cast<float>(std::cos(theta)), static_cast<float>(std::sin(theta)));
        const cv::Point2f anchor(static_cast<float>(centroid.x), static_cast<float>(centroid.y));

        float minProjection = std::numeric_limits<float>::max();
        float maxProjection = std::numeric_limits<float>::lowest();
        for (const cv::Point2f& point : points) {
            const float projection = (point - anchor).dot(direction);
            minProjection = std::min(minProjection, projection);
            maxProjection = std::max(maxProjection, projection);
        }

        const cv::Point2f nextStart = anchor + direction * minProjection;
        const cv::Point2f nextEnd = anchor + direction * maxProjection;
        if (cv::norm(nextStart - currentStart) + cv::norm(nextEnd - currentEnd) <= kConvergence) {
            currentStart = nextStart;
            currentEnd = nextEnd;
            break;
        }

        currentStart = nextStart;
        currentEnd = nextEnd;
    }

    start = currentStart;
    end = currentEnd;
    return true;
}

LineDetectionResult BuildFailure(
    const std::string& message,
    const LineDetectionFrame& frame,
    LineScanDirection scanDirection,
    std::vector<LineEdgePoint> edgePoints,
    double elapsedMilliseconds) {
    LineDetectionResult result;
    result.success = false;
    result.message = message;
    result.frame = frame;
    result.scanDirection = scanDirection;
    result.rawPointCount = static_cast<int>(edgePoints.size());
    result.fitPointCount = static_cast<int>(edgePoints.size());
    result.edgePoints = std::move(edgePoints);
    result.elapsedMilliseconds = elapsedMilliseconds;
    return result;
}

LineDetectionResult BuildSuccess(
    const LineDetectionFrame& frame,
    LineScanDirection scanDirection,
    const cv::Point2f& start,
    const cv::Point2f& end,
    int rawPointCount,
    std::vector<LineEdgePoint> edgePoints,
    std::vector<LineEdgePoint> excludedEdgePoints,
    double elapsedMilliseconds) {
    LineDetectionResult result;
    result.success = true;
    result.message = "Line Detection succeeded";
    result.frame = frame;
    result.scanDirection = scanDirection;
    result.lineStart = start;
    result.lineEnd = end;
    result.angleDegrees = CalculateAngleDegrees(start, end);
    result.fitError = CalculateFitError(edgePoints, start, end);
    result.fitErrorRmse = CalculateFitErrorRmse(edgePoints, start, end);
    result.elapsedMilliseconds = elapsedMilliseconds;
    result.rawPointCount = rawPointCount;
    result.fitPointCount = static_cast<int>(edgePoints.size());
    result.excludedPointCount = std::max(0, rawPointCount - result.fitPointCount);
    result.edgePoints = std::move(edgePoints);
    result.excludedEdgePoints = std::move(excludedEdgePoints);

    if (!result.edgePoints.empty()) {
        float sumStrength = 0.0f;
        float maxStrength = 0.0f;
        for (const LineEdgePoint& point : result.edgePoints) {
            sumStrength += point.strength;
            maxStrength = std::max(maxStrength, point.strength);
        }
        result.averageStrength = sumStrength / static_cast<float>(result.edgePoints.size());
        result.maxStrength = maxStrength;
    }

    return result;
}

/// <summary>
/// 根据初始最小二乘拟合结果，剔除距离拟合直线最远的指定数量边缘点。
/// </summary>
/// <param name="edgePoints">原始边缘点集合。</param>
/// <param name="excludeCount">需要剔除的点数。</param>
/// <param name="excludedPoints">用于接收被剔除点的输出集合，可为空。</param>
/// <returns>剔除离群点后用于最终拟合的边缘点集合。</returns>
std::vector<LineEdgePoint> ExcludeOutlierPoints(
    const std::vector<LineEdgePoint>& edgePoints,
    int excludeCount,
    std::vector<LineEdgePoint>* excludedPoints) {
    if (edgePoints.size() < 3 || excludeCount <= 0) {
        return edgePoints;
    }

    const int maxExcludeCount = std::max(0, static_cast<int>(edgePoints.size()) - 2);
    excludeCount = std::min(excludeCount, maxExcludeCount);
    if (excludeCount <= 0) {
        return edgePoints;
    }

    cv::Point2f start;
    cv::Point2f end;
    if (!FitLeastSquares(edgePoints, start, end)) {
        return edgePoints;
    }

    struct PointDistance {
        size_t index = 0;
        float distance = 0.0f;
    };

    std::vector<PointDistance> rankedPoints;
    rankedPoints.reserve(edgePoints.size());
    for (size_t index = 0; index < edgePoints.size(); ++index) {
        rankedPoints.push_back({index, CalculatePointDistanceToLine(edgePoints[index], start, end)});
    }

    std::sort(rankedPoints.begin(), rankedPoints.end(), [](const PointDistance& left, const PointDistance& right) {
        if (left.distance != right.distance) {
            return left.distance > right.distance;
        }
        return left.index < right.index;
    });

    std::vector<bool> removed(edgePoints.size(), false);
    for (int index = 0; index < excludeCount; ++index) {
        removed[rankedPoints[index].index] = true;
    }

    std::vector<LineEdgePoint> filteredPoints;
    filteredPoints.reserve(edgePoints.size() - excludeCount);
    if (excludedPoints != nullptr) {
        excludedPoints->clear();
        excludedPoints->reserve(excludeCount);
    }
    for (size_t index = 0; index < edgePoints.size(); ++index) {
        if (removed[index]) {
            if (excludedPoints != nullptr) {
                excludedPoints->push_back(edgePoints[index]);
            }
        } else {
            filteredPoints.push_back(edgePoints[index]);
        }
    }

    return filteredPoints;
}

/// <summary>
/// 执行卡尺边缘检测、候选点选择、离群点剔除和直线拟合的完整流程。
/// </summary>
/// <param name="grayImage">8位单通道灰度图像。</param>
/// <param name="calipers">待执行边缘扫描的卡尺集合。</param>
/// <param name="frame">检测ROI；箭头卡尺模式下可为空框。</param>
/// <param name="params">完整直线检测参数。</param>
/// <returns>包含检测点、剔除点、拟合线和统计信息的检测结果。</returns>
LineDetectionResult DetectFromCalipers(
    const cv::Mat& grayImage,
    const std::vector<CaliperInfo>& calipers,
    const LineDetectionFrame& frame,
    const LineDetectionParams& params) 
{
    const auto startTime = std::chrono::steady_clock::now();
    const auto elapsedMs = [&]() -> double {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - startTime).count();
    };
    const LineDetectionParams normalizedParams = NormalizeParams(params);

    if (grayImage.empty()) {
        return BuildFailure("No image loaded", frame, normalizedParams.scanDirection, {}, 0.0);
    }
    if (calipers.empty()) {
        return BuildFailure("No calipers available", frame, normalizedParams.scanDirection, {}, 0.0);
    }

    std::vector<LineEdgePoint> collectedPoints;
    collectedPoints.reserve(calipers.size());
    for (const CaliperInfo& caliper : calipers) {
        const std::vector<CandidateEdgePoint> edges = FindEdgesInCaliper(grayImage, caliper, normalizedParams);

        // ===== DEBUG: edgePoints=====
        //std::cout << "[DEBUG] edges count = " << edges.size() << '\n';
        //for (size_t i = 0; i < edges.size(); ++i) {
        //    const auto& edge = edges[i];
        //    std::cout << "  edges[" << i << "]"
        //        << ": position=(" << edge.position.x << ", " << edge.position.y << ")"
        //        << ", strength=" << edge.strength
        //        << ", isValid=" << edge.isValid
        //        << '\n';
        //}
        // ===== DEBUG END =====

        const CandidateEdgePoint selected = SelectPoint(edges, normalizedParams.selectionMode);
        if (selected.isValid) {
            collectedPoints.push_back(LineEdgePoint{selected.position, selected.strength});
        }
    }

    if (collectedPoints.size() < 2) {
        return BuildFailure(
            "Not enough edge points to fit a line",
            frame,
            normalizedParams.scanDirection,
            std::move(collectedPoints),
            elapsedMs());
    }

    const int rawPointCount = static_cast<int>(collectedPoints.size());
    std::vector<LineEdgePoint> excludedPoints;
    std::vector<LineEdgePoint> fitPoints = ExcludeOutlierPoints(
        collectedPoints,
        normalizedParams.excludPts,
        &excludedPoints);
    if (fitPoints.size() < 2) {
        return BuildFailure(
            "Not enough points left after excluding outliers",
            frame,
            normalizedParams.scanDirection,
            std::move(collectedPoints),
            elapsedMs());
    }

    cv::Point2f fitStart;
    cv::Point2f fitEnd;

    bool fitOk = false;
    switch (normalizedParams.fitMode)
    {
    case LineFitMode::LeastSquares:
        fitOk = FitLeastSquares(fitPoints, fitStart, fitEnd);
        break;
    case LineFitMode::RobustHuber:
        fitOk = FitRobust(fitPoints, fitStart, fitEnd, normalizedParams.fitMode);
        break;
    case LineFitMode::RobustWelsch:
        fitOk = FitRobust(fitPoints, fitStart, fitEnd, normalizedParams.fitMode);
        break;
    case LineFitMode::WeightTukey:
        fitOk = FitWeightTukey(fitPoints, fitStart, fitEnd);
        break;
    }

    if (!fitOk) {
        return BuildFailure(
            "Line fitting failed",
            frame,
            normalizedParams.scanDirection,
            std::move(collectedPoints),
            elapsedMs());
    }

    return BuildSuccess(
        frame,
        normalizedParams.scanDirection,
        fitStart,
        fitEnd,
        rawPointCount,
        std::move(fitPoints),
        std::move(excludedPoints),
        elapsedMs());
}

}  // namespace

cv::Point2f LineDetectionFrame::XDirection() const {
    const float radians = angleDegrees * kPi / 180.0f;
    return cv::Point2f(std::cos(radians), std::sin(radians));
}

cv::Point2f LineDetectionFrame::YDirection() const {
    const cv::Point2f xDirection = XDirection();
    return cv::Point2f(-xDirection.y, xDirection.x);
}

/// <summary>
/// 根据扫描方向枚举值，获取旋转ROI坐标系中的单位扫描向量。
/// </summary>
/// <param name="direction">扫描方向。</param>
/// <returns>图像坐标系中的单位扫描方向向量。</returns>
cv::Point2f LineDetectionFrame::GetScanDirection(LineScanDirection direction) const {
    const cv::Point2f xDirection = XDirection();
    const cv::Point2f yDirection = YDirection();
    switch (direction) {
    case LineScanDirection::RightToLeft:
        return -xDirection;
    case LineScanDirection::TopToBottom:
        return yDirection;
    case LineScanDirection::BottomToTop:
        return -yDirection;
    case LineScanDirection::LeftToRight:
    default:
        return xDirection;
    }
}

/// <summary>
/// 获取垂直于扫描方向的卡尺排列方向。
/// </summary>
/// <param name="direction">扫描方向。</param>
/// <returns>图像坐标系中的单位卡尺排列方向向量。</returns>
cv::Point2f LineDetectionFrame::GetArrangeDirection(LineScanDirection direction) const {
    return Rotate90(GetScanDirection(direction));
}

/// <summary>
/// 获取ROI在指定扫描方向上的长度。
/// </summary>
/// <param name="direction">扫描方向。</param>
/// <returns>卡尺扫描长度。</returns>
float LineDetectionFrame::GetScanLength(LineScanDirection direction) const {
    return (direction == LineScanDirection::LeftToRight || direction == LineScanDirection::RightToLeft) ? width : height;
}

/// <summary>
/// 获取ROI在垂直于指定扫描方向上的排列长度。
/// </summary>
/// <param name="direction">扫描方向。</param>
/// <returns>卡尺排列长度。</returns>
float LineDetectionFrame::GetArrangeLength(LineScanDirection direction) const {
    return (direction == LineScanDirection::LeftToRight || direction == LineScanDirection::RightToLeft) ? height : width;
}

std::vector<cv::Point2f> LineDetectionFrame::GetCorners() const {
    const cv::Point2f xDirection = XDirection();
    const cv::Point2f yDirection = YDirection();
    const float halfWidth = width * 0.5f;
    const float halfHeight = height * 0.5f;
    return {
        center - xDirection * halfWidth - yDirection * halfHeight,
        center + xDirection * halfWidth - yDirection * halfHeight,
        center + xDirection * halfWidth + yDirection * halfHeight,
        center - xDirection * halfWidth + yDirection * halfHeight,
    };
}

std::string ToDisplayString(LineEdgePolarity polarity) {
    switch (polarity) {
    case LineEdgePolarity::Positive:
        return "Positive";
    case LineEdgePolarity::Negative:
        return "Negative";
    case LineEdgePolarity::Any:
    default:
        return "Any";
    }
}

std::string ToDisplayString(LineSelectionMode selectionMode) {
    switch (selectionMode) {
    case LineSelectionMode::First:
        return "First";
    case LineSelectionMode::Last:
        return "Last";
    case LineSelectionMode::Strongest:
    default:
        return "Strongest";
    }
}

std::string ToDisplayString(LineScanDirection scanDirection) {
    switch (scanDirection) {
    case LineScanDirection::TopToBottom:
        return "TopToBottom";
    case LineScanDirection::BottomToTop:
        return "BottomToTop";
    case LineScanDirection::RightToLeft:
        return "RightToLeft";
    case LineScanDirection::LeftToRight:
    default:
        return "LeftToRight";
    }
}

std::string ToDisplayString(LineFitMode fitMode) {
    switch (fitMode) {
    case LineFitMode::LeastSquares:
        return "LeastSquares";
    case LineFitMode::RobustHuber:
        return "Huber";
    case LineFitMode::RobustWelsch :
        return "Welsch";
    case LineFitMode::WeightTukey:
        return "WeightTukey";
    default:
        return "Unknown";
    }
}

/// <summary>
/// 在输入图像的旋转ROI内生成卡尺并执行直线检测。
/// </summary>
/// <param name="image">输入图像。</param>
/// <param name="frame">旋转ROI测量框。</param>
/// <param name="params">直线检测参数。</param>
/// <returns>直线检测结果。</returns>
LineDetectionResult LineDetectionOperator::Detect(
    const cv::Mat& image,
    const LineDetectionFrame& frame,
    const LineDetectionParams& params) const {
    if (!frame.IsValid()) {
        const LineDetectionParams normalizedParams = NormalizeParams(params);
        return BuildFailure("ROI is invalid", frame, normalizedParams.scanDirection, {}, 0.0);
    }

    const cv::Mat grayImage = ToGray(image);
    return DetectFromCalipers(grayImage, GenerateCalipers(frame, params), frame, params);
}

LineDetectionResult LineDetectionOperator::DetectGray(
    const cv::Mat& grayImage,
    const LineDetectionFrame& frame,
    const LineDetectionParams& params) const {
    if (grayImage.channels() != 1 || grayImage.depth() != CV_8U) {
        return Detect(grayImage, frame, params);
    }
    if (!frame.IsValid()) {
        const LineDetectionParams normalizedParams = NormalizeParams(params);
        return BuildFailure("ROI is invalid", frame, normalizedParams.scanDirection, {}, 0.0);
    }

    return DetectFromCalipers(grayImage, GenerateCalipers(frame, params), frame, params);
}

/// <summary>
/// 使用箭头卡尺集合执行直线检测。
/// </summary>
/// <param name="image">输入图像。</param>
/// <param name="calipers">用户创建的箭头卡尺集合。</param>
/// <param name="params">直线检测参数。</param>
/// <returns>直线检测结果。</returns>
LineDetectionResult LineDetectionOperator::Detect(
    const cv::Mat& image,
    const std::vector<LineCaliper>& calipers,
    const LineDetectionParams& params) const {
    const cv::Mat grayImage = ToGray(image);
    return DetectFromCalipers(grayImage, ConvertCalipers(calipers), LineDetectionFrame(), params);
}

LineDetectionResult LineDetectionOperator::DetectGray(
    const cv::Mat& grayImage,
    const std::vector<LineCaliper>& calipers,
    const LineDetectionParams& params) const {
    if (grayImage.channels() != 1 || grayImage.depth() != CV_8U) {
        return Detect(grayImage, calipers, params);
    }

    return DetectFromCalipers(grayImage, ConvertCalipers(calipers), LineDetectionFrame(), params);
}
