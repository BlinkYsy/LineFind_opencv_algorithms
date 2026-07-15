#include "demo_canvas.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>

namespace {

constexpr int kSidebarWidth = 320;
constexpr int kStatusHeight = 136;
constexpr int kMargin = 16;
constexpr int kButtonHeight = 34;
constexpr int kParamHeight = 30;
constexpr float kMinFrameSize = 10.0f;
constexpr float kMinAxisLength = 20.0f;
constexpr float kMinCaliperLength = 20.0f;
constexpr float kMinCaliperWidth = 6.0f;
constexpr float kResizeHandleRadius = 10.0f;
constexpr float kRotationStep = 2.0f;
constexpr float kArrowHeadLength = 18.0f;
constexpr float kArrowAxisHitPadding = 10.0f;
constexpr float kArrowHandleOuterRadius = 9.0f;
constexpr float kArrowHandleInnerRadius = 4.0f;
constexpr float kArrowLengthHandleRadius = 7.0f;
constexpr double kStatusFontScale = 0.48;
constexpr float kZoomStep = 1.12f;
constexpr float kMinZoomFactor = 0.2f;
constexpr float kMaxZoomFactor = 16.0f;
constexpr float kPixelGridMinScale = 2.0f;

std::string WideToUtf8(const std::wstring& value) {
    if (value.empty()) {
        return std::string();
    }

    const int required = WideCharToMultiByte(CP_UTF8, 0, value.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string result(required > 0 ? required : 0, '\0');
    if (required > 1) {
        WideCharToMultiByte(CP_UTF8, 0, value.c_str(), -1, &result[0], required, nullptr, nullptr);
        if (!result.empty() && result.back() == '\0') {
            result.pop_back();
        }
    }
    return result;
}

cv::Point2f Normalize(const cv::Point2f& value) {
    const float length = std::sqrt(value.dot(value));
    if (length <= 1e-6f) {
        return cv::Point2f(1.0f, 0.0f);
    }
    return value * (1.0f / length);
}

cv::Point2f Rotate90(const cv::Point2f& value) {
    return cv::Point2f(-value.y, value.x);
}

cv::Point2f ScanDirectionVector(LineScanDirection direction) {
    switch (direction) {
    case LineScanDirection::RightToLeft:
        return cv::Point2f(-1.0f, 0.0f);
    case LineScanDirection::TopToBottom:
        return cv::Point2f(0.0f, 1.0f);
    case LineScanDirection::BottomToTop:
        return cv::Point2f(0.0f, -1.0f);
    case LineScanDirection::LeftToRight:
    default:
        return cv::Point2f(1.0f, 0.0f);
    }
}

cv::Point2f ResolveArrowScanDirection(const cv::Point2f& axisDirection, LineScanDirection desiredDirection) {
    const cv::Point2f normal = Rotate90(axisDirection);
    const cv::Point2f desired = ScanDirectionVector(desiredDirection);
    return normal.dot(desired) >= 0.0f ? normal : -normal;
}

LineEdgePolarity CycleEdgePolarity(LineEdgePolarity value, bool increment) {
    switch (value) {
    case LineEdgePolarity::Positive:
        return increment ? LineEdgePolarity::Negative : LineEdgePolarity::Any;
    case LineEdgePolarity::Negative:
        return increment ? LineEdgePolarity::Any : LineEdgePolarity::Positive;
    case LineEdgePolarity::Any:
    default:
        return increment ? LineEdgePolarity::Positive : LineEdgePolarity::Negative;
    }
}

LineSelectionMode CycleSelectionMode(LineSelectionMode value, bool increment) {
    switch (value) {
    case LineSelectionMode::Strongest:
        return increment ? LineSelectionMode::First : LineSelectionMode::Last;
    case LineSelectionMode::First:
        return increment ? LineSelectionMode::Last : LineSelectionMode::Strongest;
    case LineSelectionMode::Last:
    default:
        return increment ? LineSelectionMode::Strongest : LineSelectionMode::First;
    }
}

LineScanDirection CycleScanDirection(LineScanDirection value, bool increment) {
    switch (value) {
    case LineScanDirection::LeftToRight:
        return increment ? LineScanDirection::TopToBottom : LineScanDirection::RightToLeft;
    case LineScanDirection::TopToBottom:
        return increment ? LineScanDirection::BottomToTop : LineScanDirection::LeftToRight;
    case LineScanDirection::BottomToTop:
        return increment ? LineScanDirection::RightToLeft : LineScanDirection::TopToBottom;
    case LineScanDirection::RightToLeft:
    default:
        return increment ? LineScanDirection::LeftToRight : LineScanDirection::BottomToTop;
    }
}

LineFitMode CycleFitMode(LineFitMode value, bool increment) {
    switch (value) {
    case LineFitMode::LeastSquares:
        return increment ? LineFitMode::RobustHuber : LineFitMode::WeightTukey;
    case LineFitMode::RobustHuber:
        return increment ? LineFitMode::RobustWelsch : LineFitMode::LeastSquares;
    case LineFitMode::RobustWelsch:
        return increment ? LineFitMode::WeightTukey : LineFitMode::RobustHuber;
    case LineFitMode::WeightTukey:
    default:
        return increment ? LineFitMode::LeastSquares : LineFitMode::RobustWelsch;
    }
}

cv::Scalar MakeColor(int b, int g, int r) {
    return cv::Scalar(static_cast<double>(b), static_cast<double>(g), static_cast<double>(r));
}

bool PointInRect(const cv::Point& point, const cv::Rect& rect) {
    return rect.contains(point);
}

float DistancePointToLine(const cv::Point2f& point, const cv::Point2f& start, const cv::Point2f& end) {
    const cv::Point2f segment = end - start;
    const float segmentLength = std::sqrt(segment.dot(segment));
    if (segmentLength <= 1e-6f) {
        return static_cast<float>(cv::norm(point - start));
    }

    return static_cast<float>(std::fabs((point - start).cross(segment)) / segmentLength);
}

float Cross(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x * b.y - a.y * b.x;
}

bool IsSamePoint(const cv::Point2f& a, const cv::Point2f& b) {
    return cv::norm(a - b) <= 1e-3f;
}

bool IntersectLineWithSegment(
    const cv::Point2f& linePoint,
    const cv::Point2f& lineDirection,
    const cv::Point2f& segmentStart,
    const cv::Point2f& segmentEnd,
    cv::Point2f& intersection,
    float& lineT) {
    const cv::Point2f segmentDirection = segmentEnd - segmentStart;
    const float denominator = Cross(lineDirection, segmentDirection);
    if (std::fabs(denominator) <= 1e-6f) {
        return false;
    }

    const cv::Point2f delta = segmentStart - linePoint;
    lineT = Cross(delta, segmentDirection) / denominator;
    const float segmentT = Cross(delta, lineDirection) / denominator;
    if (segmentT < -1e-4f || segmentT > 1.0f + 1e-4f) {
        return false;
    }

    intersection = linePoint + lineDirection * lineT;
    return true;
}

}  // namespace

DemoCanvasApp::DemoCanvasApp(std::vector<std::string> sampleImages)
    : sampleImages_(std::move(sampleImages)) {
    BuildLayout();
}

int DemoCanvasApp::Run() {
    cv::namedWindow(windowName_, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(windowName_, &DemoCanvasApp::OnMouseThunk, this);

    if (!sampleImages_.empty()) {
        LoadNextSample();
    }

    for (;;) {
        if (cv::getWindowProperty(windowName_, cv::WND_PROP_VISIBLE) < 1.0) {
            break;
        }
        Render();
        const int key = cv::waitKeyEx(16);
        if (key < 0) {
            continue;
        }
        if (key == 27) {
            break;
        }
        HandleKey(key);
    }

    cv::destroyWindow(windowName_);
    return 0;
}

void DemoCanvasApp::OnMouseThunk(int event, int x, int y, int flags, void* userdata) {
    if (userdata == nullptr) {
        return;
    }

    static_cast<DemoCanvasApp*>(userdata)->OnMouse(event, x, y, flags);
}

void DemoCanvasApp::OnMouse(int event, int x, int y, int flags) {
    const cv::Point point(x, y);
    if (!sourceImage_.empty() && viewport_.imageRect.contains(point)) {
        const cv::Point2f imagePoint = ScreenToImage(point);
        mouseImagePixel_.x = std::max(0, std::min(sourceImage_.cols - 1, static_cast<int>(std::floor(imagePoint.x))));
        mouseImagePixel_.y = std::max(0, std::min(sourceImage_.rows - 1, static_cast<int>(std::floor(imagePoint.y))));
        hasMouseImagePixel_ = true;
    } else {
        hasMouseImagePixel_ = false;
    }

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (PointInRect(point, sidebarRect_)) {
            ApplySidebarClick(point);
            return;
        }

        cv::Point2f imagePoint;
        if (!HitTestImagePoint(x, y, imagePoint)) {
            return;
        }

        if (createMode_ == CreateMode::ArrowCaliper) {
            if (dragMode_ == DragMode::None) {
                StartArrowAxisCreate(imagePoint);
            } else if (dragMode_ == DragMode::CreatingArrowLength || dragMode_ == DragMode::CreatingArrowWidth) {
                dragStartImagePoint_ = imagePoint;
            }
            return;
        }

        if (createMode_ == CreateMode::RotatedRectangle) {
            StartFrameCreate(imagePoint);
            return;
        }

        if (hasRoi_ && roiKind_ == RoiKind::ArrowCaliper && arrowRoi_.axisReady && arrowRoi_.lengthReady) {
            bool isStartHandle = false;
            if (HitTestArrowEndpoint(imagePoint, isStartHandle)) {
                dragMode_ = isStartHandle ? DragMode::ResizingArrowStart : DragMode::ResizingArrowEnd;
                dragStartImagePoint_ = imagePoint;
                ClearResult();
                return;
            }
            if (HitTestArrowLengthHandle(imagePoint)) {
                dragMode_ = DragMode::ResizingArrowLength;
                dragStartImagePoint_ = imagePoint;
                ClearResult();
                return;
            }
            if (HitTestArrowAxis(imagePoint) || HitTestArrowCaliperBody(imagePoint)) {
                dragMode_ = DragMode::MovingArrowGroup;
                dragStartImagePoint_ = imagePoint;
                ClearResult();
                return;
            }
        }

        if (hasRoi_ && roiKind_ == RoiKind::RotatedFrame) {
            int signX = 0;
            int signY = 0;
            if (HitTestResizeHandle(imagePoint, frameRoi_, signX, signY)) {
                dragMode_ = DragMode::ResizingFrame;
                resizeState_.originalFrame = frameRoi_;
                resizeState_.signX = signX;
                resizeState_.signY = signY;
                const cv::Point2f localPoint = ToLocal(frameRoi_, imagePoint);
                resizeState_.startLocalX = localPoint.x;
                resizeState_.startLocalY = localPoint.y;
                ClearResult();
                return;
            }

            if (HitTestInsideFrame(imagePoint, frameRoi_)) {
                dragMode_ = DragMode::MovingFrame;
                dragOffset_ = imagePoint - frameRoi_.center;
                ClearResult();
                return;
            }
        }

        if (sourceImage_.empty() || !imagePanelRect_.contains(point)) {
            return;
        }

        dragMode_ = DragMode::PanningImageView;
        dragStartScreenPoint_ = point;
        panStartOffset_ = imageOffset_;
        return;
    }

    if (event == cv::EVENT_MOUSEMOVE) {
        cv::Point2f imagePoint;
        if (!HitTestImagePoint(x, y, imagePoint)) {
            return;
        }

        if (dragMode_ == DragMode::CreatingArrowAxis) {
            UpdateArrowAxisCreate(imagePoint);
            return;
        }
        if (dragMode_ == DragMode::CreatingArrowLength) {
            UpdateArrowLengthCreate(imagePoint);
            return;
        }
        if (dragMode_ == DragMode::CreatingArrowWidth) {
            UpdateArrowWidthCreate(imagePoint);
            return;
        }
        if (dragMode_ == DragMode::CreatingFrame) {
            UpdateFrameCreate(imagePoint);
            return;
        }
        if (dragMode_ == DragMode::ResizingArrowStart && roiKind_ == RoiKind::ArrowCaliper) {
            if (cv::norm(imagePoint - arrowRoi_.axisEnd) >= kMinAxisLength) {
                arrowRoi_.axisStart = imagePoint;
                SyncArrowScanDirectionToAxis();
                SyncArrowCaliperWidthToAxis();
                hasResult_ = false;
            }
            return;
        }
        if (dragMode_ == DragMode::ResizingArrowEnd && roiKind_ == RoiKind::ArrowCaliper) {
            if (cv::norm(imagePoint - arrowRoi_.axisStart) >= kMinAxisLength) {
                arrowRoi_.axisEnd = imagePoint;
                SyncArrowScanDirectionToAxis();
                SyncArrowCaliperWidthToAxis();
                hasResult_ = false;
            }
            return;
        }
        if (dragMode_ == DragMode::ResizingArrowLength && roiKind_ == RoiKind::ArrowCaliper) {
            UpdateArrowLengthCreate(imagePoint);
            hasResult_ = false;
            return;
        }
        if (dragMode_ == DragMode::MovingArrowGroup && roiKind_ == RoiKind::ArrowCaliper) {
            const cv::Point2f delta = imagePoint - dragStartImagePoint_;
            arrowRoi_.axisStart += delta;
            arrowRoi_.axisEnd += delta;
            dragStartImagePoint_ = imagePoint;
            hasResult_ = false;
            return;
        }
        if (dragMode_ == DragMode::PanningImageView) {
            imageOffset_ = panStartOffset_ + cv::Point2f(
                static_cast<float>(point.x - dragStartScreenPoint_.x),
                static_cast<float>(point.y - dragStartScreenPoint_.y));
            ClampImageOffset(viewport_.imageRect.size());
            return;
        }
        if (dragMode_ == DragMode::MovingFrame && roiKind_ == RoiKind::RotatedFrame) {
            frameRoi_.center = imagePoint - dragOffset_;
            return;
        }
        if (dragMode_ == DragMode::ResizingFrame && roiKind_ == RoiKind::RotatedFrame) {
            const cv::Point2f currentLocal = ToLocal(resizeState_.originalFrame, imagePoint);
            const float deltaX = currentLocal.x - resizeState_.startLocalX;
            const float deltaY = currentLocal.y - resizeState_.startLocalY;
            frameRoi_ = resizeState_.originalFrame;
            frameRoi_.width = std::max(kMinFrameSize, resizeState_.originalFrame.width + deltaX * 2.0f * resizeState_.signX);
            frameRoi_.height = std::max(kMinFrameSize, resizeState_.originalFrame.height + deltaY * 2.0f * resizeState_.signY);
            return;
        }
    }

    if (event == cv::EVENT_LBUTTONUP) {
        if (dragMode_ == DragMode::CreatingArrowAxis) {
            if (ComputeArrowAxisLength() >= kMinAxisLength) {
                StartArrowLengthCreate();
            } else {
                ClearRoi();
            }
            return;
        }
        if (dragMode_ == DragMode::CreatingArrowLength) {
            StartArrowWidthCreate();
            return;
        }
        if (dragMode_ == DragMode::CreatingArrowWidth) {
            FinishArrowWidthCreate();
            return;
        }
        if (dragMode_ == DragMode::CreatingFrame) {
            FinishFrameCreate();
            return;
        }
        dragMode_ = DragMode::None;
    }

    if (event == cv::EVENT_MOUSEWHEEL) {
        const int wheel = cv::getMouseWheelDelta(flags);
        if (wheel == 0) {
            return;
        }

        const bool ctrlPressed = (flags & cv::EVENT_FLAG_CTRLKEY) != 0;
        if (ctrlPressed && roiKind_ == RoiKind::RotatedFrame) {
            frameRoi_.angleDegrees += wheel > 0 ? kRotationStep : -kRotationStep;
            ClearResult();
            return;
        }

        if (!sourceImage_.empty() && imagePanelRect_.contains(point)) {
            ZoomAtScreenPoint(point, wheel > 0 ? kZoomStep : (1.0f / kZoomStep));
        }
    }
}

void DemoCanvasApp::HandleKey(int key) {
    switch (key) {
    case 'o':
    case 'O':
        OpenImageWithDialog();
        return;
    case 'n':
    case 'N':
        LoadNextSample();
        return;
    case 'r':
    case 'R':
        ResetCreateMode(CreateMode::ArrowCaliper);
        return;
    case 't':
    case 'T':
        ResetCreateMode(CreateMode::RotatedRectangle);
        return;
    case 'd':
    case 'D':
        DetectLine();
        return;
    case 'c':
    case 'C':
        ClearResult();
        return;
    case 'x':
    case 'X':
        ClearRoi();
        return;
    case 'q':
    case 'Q':
        if (roiKind_ == RoiKind::RotatedFrame) {
            frameRoi_.angleDegrees -= kRotationStep;
            ClearResult();
        }
        return;
    case 'e':
    case 'E':
        if (roiKind_ == RoiKind::RotatedFrame) {
            frameRoi_.angleDegrees += kRotationStep;
            ClearResult();
        }
        return;
    case 2490368:
        params_.edgeThreshold += 1.0f;
        ClearResult();
        return;
    case 2621440:
        params_.edgeThreshold = std::max(0.0f, params_.edgeThreshold - 1.0f);
        ClearResult();
        return;
    default:
        return;
    }
}

void DemoCanvasApp::Render() {
    cv::Mat canvas(windowSize_, CV_8UC3, MakeColor(34, 34, 34));
    DrawSidebar(canvas);
    DrawImagePanel(canvas);
    DrawStatus(canvas);
    cv::imshow(windowName_, canvas);
}

void DemoCanvasApp::BuildLayout() {
    sidebarRect_ = cv::Rect(0, 0, kSidebarWidth, windowSize_.height);
    imagePanelRect_ = cv::Rect(kSidebarWidth, 0, windowSize_.width - kSidebarWidth, windowSize_.height - kStatusHeight);
    statusRect_ = cv::Rect(kSidebarWidth, windowSize_.height - kStatusHeight, windowSize_.width - kSidebarWidth, kStatusHeight);

    int top = kMargin;
    buttons_ = {
        {ButtonId::OpenImage, "Open Image", cv::Rect(kMargin, top, kSidebarWidth - kMargin * 2, kButtonHeight)},
        {ButtonId::NextSample, "Next Image", cv::Rect(kMargin, top += 42, kSidebarWidth - kMargin * 2, kButtonHeight)},
        {ButtonId::CreateRect, "Create Arrow Calipers", cv::Rect(kMargin, top += 58, kSidebarWidth - kMargin * 2, kButtonHeight)},
        {ButtonId::CreateRotated, "Create Rotated ROI", cv::Rect(kMargin, top += 42, kSidebarWidth - kMargin * 2, kButtonHeight)},
        {ButtonId::Detect, "Detect Line", cv::Rect(kMargin, top += 58, kSidebarWidth - kMargin * 2, kButtonHeight)},
        {ButtonId::ClearRoi, "Clear Calipers/ROI", cv::Rect(kMargin, top += 42, kSidebarWidth - kMargin * 2, kButtonHeight)},
        {ButtonId::ClearResult, "Clear Result", cv::Rect(kMargin, top += 42, kSidebarWidth - kMargin * 2, kButtonHeight)},
    };

    top += 55;
    const int rowWidth = kSidebarWidth - kMargin * 2;
    const int valueWidth = 124;
    const int buttonWidth = 30;
    const int spacing = 4;
    parameterControls_.clear();
    const std::array<std::string, 9> labels = {
        "Caliper Count", "Sample Step", "Filter Size", "Edge Threshold", "Edge Polarity", "EdgeType Selection", "Scan Direction", "Fit Mode", "ExcludPts Num"
    };
    for (const std::string& label : labels) {
        ParamControl control;
        control.label = label;
        control.rowRect = cv::Rect(kMargin, top, rowWidth, kParamHeight);
        control.minusRect = cv::Rect(kMargin, top + 14, buttonWidth, buttonWidth - 2);
        control.valueRect = cv::Rect(kMargin + buttonWidth + spacing, top + 14, valueWidth, buttonWidth - 2);
        control.plusRect = cv::Rect(control.valueRect.x + control.valueRect.width + spacing, top + 14, buttonWidth, buttonWidth - 2);
        parameterControls_.push_back(control);
        top += 58;
    }
}

void DemoCanvasApp::DrawSidebar(cv::Mat& canvas) {
    cv::rectangle(canvas, sidebarRect_, MakeColor(44, 44, 44), cv::FILLED);
    cv::putText(canvas, "Linefind Demo", cv::Point(18, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, MakeColor(255, 255, 255), 2, cv::LINE_AA);

    for (const Button& button : buttons_) {
        cv::rectangle(canvas, button.rect, MakeColor(70, 70, 70), cv::FILLED);
        cv::rectangle(canvas, button.rect, MakeColor(115, 115, 115), 1);
        cv::putText(canvas, button.label, cv::Point(button.rect.x + 10, button.rect.y + 22), cv::FONT_HERSHEY_SIMPLEX, 0.48, MakeColor(240, 240, 240), 1, cv::LINE_AA);
    }

    const std::array<std::string, 9> values = {
        std::to_string(params_.caliperCount),
        std::to_string(static_cast<int>(params_.sampleStep)),
        std::to_string(params_.filterSize),
        std::to_string(static_cast<int>(params_.edgeThreshold)),
        ToDisplayString(params_.edgePolarity),
        ToDisplayString(params_.selectionMode),
        ToDisplayString(params_.scanDirection),
        ToDisplayString(params_.fitMode),
        std::to_string(params_.excludPts),
    };

    for (size_t index = 0; index < parameterControls_.size(); ++index) {
        const ParamControl& control = parameterControls_[index];
        cv::putText(canvas, control.label, cv::Point(control.rowRect.x, control.rowRect.y + 10), cv::FONT_HERSHEY_SIMPLEX, 0.45, MakeColor(220, 220, 220), 1, cv::LINE_AA);
        cv::rectangle(canvas, control.minusRect, MakeColor(80, 80, 80), cv::FILLED);
        cv::rectangle(canvas, control.valueRect, MakeColor(58, 58, 58), cv::FILLED);
        cv::rectangle(canvas, control.plusRect, MakeColor(80, 80, 80), cv::FILLED);
        cv::rectangle(canvas, control.minusRect, MakeColor(115, 115, 115), 1);
        cv::rectangle(canvas, control.valueRect, MakeColor(115, 115, 115), 1);
        cv::rectangle(canvas, control.plusRect, MakeColor(115, 115, 115), 1);
        cv::putText(canvas, "<", cv::Point(control.minusRect.x + 9, control.minusRect.y + 18), cv::FONT_HERSHEY_SIMPLEX, 0.65, MakeColor(255, 255, 255), 1, cv::LINE_AA);
        cv::putText(canvas, ">", cv::Point(control.plusRect.x + 9, control.plusRect.y + 18), cv::FONT_HERSHEY_SIMPLEX, 0.65, MakeColor(255, 255, 255), 1, cv::LINE_AA);
        cv::putText(canvas, values[index], cv::Point(control.valueRect.x + 8, control.valueRect.y + 18), cv::FONT_HERSHEY_SIMPLEX, 0.46, MakeColor(240, 240, 240), 1, cv::LINE_AA);
    }

    std::stringstream stream(BuildHelpText());
    std::string line;
    int y = windowSize_.height - 150;
    while (std::getline(stream, line, '\n')) {
        cv::putText(canvas, line, cv::Point(18, y), cv::FONT_HERSHEY_SIMPLEX, 0.38, MakeColor(185, 185, 185), 1, cv::LINE_AA);
        y += 18;
    }
}

void DemoCanvasApp::DrawImagePanel(cv::Mat& canvas) {
    cv::rectangle(canvas, imagePanelRect_, MakeColor(18, 18, 18), cv::FILLED);
    viewport_.panelRect = imagePanelRect_;
    viewport_.imageRect = cv::Rect();
    viewport_.drawRect = cv::Rect();
    viewport_.scale = 1.0f;
    viewport_.baseScale = 1.0f;

    if (sourceImage_.empty()) {
        cv::putText(canvas, "Open an image or load a sample to start.", cv::Point(imagePanelRect_.x + 40, imagePanelRect_.y + 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, MakeColor(200, 200, 200), 2, cv::LINE_AA);
        return;
    }

    const float scaleX = static_cast<float>(imagePanelRect_.width - kMargin * 2) / static_cast<float>(sourceImage_.cols);
    const float scaleY = static_cast<float>(imagePanelRect_.height - kMargin * 2) / static_cast<float>(sourceImage_.rows);
    const float baseScale = std::min(scaleX, scaleY);
    const float scale = baseScale * zoomFactor_;
    const cv::Size drawSize(
        std::max(1, cvRound(static_cast<float>(sourceImage_.cols) * scale)),
        std::max(1, cvRound(static_cast<float>(sourceImage_.rows) * scale)));

    if (viewNeedsReset_) {
        imageOffset_.x = static_cast<float>(imagePanelRect_.x + (imagePanelRect_.width - drawSize.width) / 2);
        imageOffset_.y = static_cast<float>(imagePanelRect_.y + (imagePanelRect_.height - drawSize.height) / 2);
        viewNeedsReset_ = false;
    }

    ClampImageOffset(drawSize);
    const int drawX = cvRound(imageOffset_.x);
    const int drawY = cvRound(imageOffset_.y);

    viewport_.scale = scale;
    viewport_.baseScale = baseScale;
    viewport_.imageRect = cv::Rect(drawX, drawY, drawSize.width, drawSize.height);
    viewport_.drawRect = viewport_.imageRect & imagePanelRect_;

    cv::Mat resizedImage;
    //cv::resize(sourceImage_, resizedImage, drawSize);
    cv::resize(sourceImage_, resizedImage, drawSize, 0.0, 0.0, cv::INTER_NEAREST);
    if (viewport_.drawRect.width <= 0 || viewport_.drawRect.height <= 0) {
        return;
    }

    const cv::Rect srcRect(
        viewport_.drawRect.x - viewport_.imageRect.x,
        viewport_.drawRect.y - viewport_.imageRect.y,
        viewport_.drawRect.width,
        viewport_.drawRect.height);
    cv::Mat canvasRoi = canvas(viewport_.drawRect);
    resizedImage(srcRect).copyTo(canvasRoi);

    DrawPixelGrid(canvasRoi);
    DrawResultOverlay(canvasRoi);
}

void DemoCanvasApp::DrawStatus(cv::Mat& canvas) {
    cv::rectangle(canvas, statusRect_, MakeColor(28, 28, 28), cv::FILLED);
    cv::putText(canvas, BuildStatusText(), cv::Point(statusRect_.x + 16, statusRect_.y + 30), cv::FONT_HERSHEY_SIMPLEX, kStatusFontScale, MakeColor(230, 230, 230), 1, cv::LINE_AA);
    const std::vector<std::pair<std::string, std::string>> detailColumns = BuildStatusDetailColumns();
    if (!detailColumns.empty()) {
        const int startX = statusRect_.x + 16;
        const int labelY = statusRect_.y + 58;
        const int valueY = statusRect_.y + 82;
        const int separatorGap = 18;
        int cursorX = startX;

        for (size_t index = 0; index < detailColumns.size(); ++index) {
            const cv::Size labelSize = cv::getTextSize(detailColumns[index].first, cv::FONT_HERSHEY_SIMPLEX, kStatusFontScale, 1, nullptr);
            const cv::Size valueSize = cv::getTextSize(detailColumns[index].second, cv::FONT_HERSHEY_SIMPLEX, kStatusFontScale, 1, nullptr);
            const int blockWidth = std::max(labelSize.width, valueSize.width);

            cv::putText(canvas, detailColumns[index].first, cv::Point(cursorX, labelY), cv::FONT_HERSHEY_SIMPLEX, kStatusFontScale, MakeColor(205, 205, 205), 1, cv::LINE_AA);
            cv::putText(canvas, detailColumns[index].second, cv::Point(cursorX, valueY), cv::FONT_HERSHEY_SIMPLEX, kStatusFontScale, MakeColor(230, 230, 230), 1, cv::LINE_AA);

            cursorX += blockWidth + 24;
            if (index + 1 < detailColumns.size()) {
                cv::putText(canvas, "|", cv::Point(cursorX, labelY), cv::FONT_HERSHEY_SIMPLEX, kStatusFontScale, MakeColor(160, 160, 160), 1, cv::LINE_AA);
                cv::putText(canvas, "|", cv::Point(cursorX, valueY), cv::FONT_HERSHEY_SIMPLEX, kStatusFontScale, MakeColor(160, 160, 160), 1, cv::LINE_AA);
                cursorX += separatorGap;
            }
        }
    }
    if (!imagePath_.empty()) {
        cv::putText(canvas, imagePath_, cv::Point(statusRect_.x + 16, statusRect_.y + 110), cv::FONT_HERSHEY_SIMPLEX, 0.42, MakeColor(170, 170, 170), 1, cv::LINE_AA);
    }
    if (hasMouseImagePixel_) {
        std::ostringstream stream;
        stream << "X," << mouseImagePixel_.x << " Y," << mouseImagePixel_.y;
        const std::string text = stream.str();
        const double fontScale = 0.36;
        const int baseline = 0;
        const cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, const_cast<int*>(&baseline));
        const int textX = statusRect_.x + statusRect_.width - textSize.width - 5;
        const int textY = statusRect_.y + 15;
        cv::putText(canvas, text, cv::Point(textX, textY), cv::FONT_HERSHEY_SIMPLEX, fontScale, MakeColor(230, 230, 230), 1, cv::LINE_AA);
    }
}

void DemoCanvasApp::DrawPixelGrid(cv::Mat& imageCanvas) const {
    if (sourceImage_.empty() || viewport_.scale < kPixelGridMinScale || viewport_.drawRect.width <= 0 || viewport_.drawRect.height <= 0) {
        return;
    }

    const int imageLeft = viewport_.drawRect.x - viewport_.imageRect.x;
    const int imageTop = viewport_.drawRect.y - viewport_.imageRect.y;
    const int imageRight = imageLeft + viewport_.drawRect.width;
    const int imageBottom = imageTop + viewport_.drawRect.height;

    const int startColumn = std::max(0, static_cast<int>(std::floor(static_cast<float>(imageLeft) / viewport_.scale)));
    const int endColumn = std::min(sourceImage_.cols, static_cast<int>(std::ceil(static_cast<float>(imageRight) / viewport_.scale)));
    const int startRow = std::max(0, static_cast<int>(std::floor(static_cast<float>(imageTop) / viewport_.scale)));
    const int endRow = std::min(sourceImage_.rows, static_cast<int>(std::ceil(static_cast<float>(imageBottom) / viewport_.scale)));

    const cv::Scalar gridColor = MakeColor(40, 40, 40);
    for (int column = startColumn; column <= endColumn; ++column) {
        const int x = cvRound(static_cast<float>(column) * viewport_.scale) - imageLeft;
        cv::line(imageCanvas, cv::Point(x, 0), cv::Point(x, imageCanvas.rows), gridColor, 1, cv::LINE_AA);
    }

    for (int row = startRow; row <= endRow; ++row) {
        const int y = cvRound(static_cast<float>(row) * viewport_.scale) - imageTop;
        cv::line(imageCanvas, cv::Point(0, y), cv::Point(imageCanvas.cols, y), gridColor, 1, cv::LINE_AA);
    }
}

void DemoCanvasApp::DrawResultOverlay(cv::Mat& imageCanvas) const {
    if (hasRoi_) {
        if (roiKind_ == RoiKind::ArrowCaliper) {
            DrawArrowCaliperPreview(imageCanvas);
        } else if (roiKind_ == RoiKind::RotatedFrame) {
            DrawFrame(imageCanvas, frameRoi_, MakeColor(255, 191, 0), 2);
            DrawRotatedFrameCalipers(imageCanvas);
        }
    }

    if (hasResult_) {
        for (const LineEdgePoint& edgePoint : result_.excludedEdgePoints) {
            const cv::Point point = ImageToScreen(edgePoint.point) - viewport_.drawRect.tl();
            cv::line(imageCanvas, point + cv::Point(-4, 0), point + cv::Point(4, 0), MakeColor(0, 0, 255), 1, cv::LINE_AA);
            cv::line(imageCanvas, point + cv::Point(0, -4), point + cv::Point(0, 4), MakeColor(0, 0, 255), 1, cv::LINE_AA);
        }

        for (const LineEdgePoint& edgePoint : result_.edgePoints) {
            const cv::Point point = ImageToScreen(edgePoint.point) - viewport_.drawRect.tl();
            cv::line(imageCanvas, point + cv::Point(-4, 0), point + cv::Point(4, 0), MakeColor(0, 255, 0), 1, cv::LINE_AA);
            cv::line(imageCanvas, point + cv::Point(0, -4), point + cv::Point(0, 4), MakeColor(0, 255, 0), 1, cv::LINE_AA);
        }

        if (result_.success) {
            cv::Point2f displayStart = result_.lineStart;
            cv::Point2f displayEnd = result_.lineEnd;
            ComputeDisplayedResultLine(displayStart, displayEnd);
            const cv::Point start = ImageToScreen(displayStart) - viewport_.drawRect.tl();
            const cv::Point end = ImageToScreen(displayEnd) - viewport_.drawRect.tl();
            cv::line(imageCanvas, start, end, MakeColor(0, 255, 0), 1, cv::LINE_AA);
        }
    }
}

void DemoCanvasApp::DrawFrame(cv::Mat& imageCanvas, const LineDetectionFrame& frame, const cv::Scalar& color, int thickness) const {
    const std::vector<cv::Point2f> corners = frame.GetCorners();
    for (size_t index = 0; index < corners.size(); ++index) {
        const cv::Point start = ImageToScreen(corners[index]) - viewport_.drawRect.tl();
        const cv::Point end = ImageToScreen(corners[(index + 1) % corners.size()]) - viewport_.drawRect.tl();
        cv::line(imageCanvas, start, end, color, thickness, cv::LINE_AA);
        cv::circle(imageCanvas, start, 4, MakeColor(255, 255, 255), cv::FILLED, cv::LINE_AA);
    }
}

void DemoCanvasApp::DrawRotatedFrameCalipers(cv::Mat& imageCanvas) const {
    const int caliperCount = std::max(2, params_.caliperCount);
    const cv::Point2f scanDir = Normalize(frameRoi_.GetScanDirection(params_.scanDirection));
    const cv::Point2f arrangeDir = Normalize(frameRoi_.GetArrangeDirection(params_.scanDirection));
    const float arrangeLength = frameRoi_.GetArrangeLength(params_.scanDirection);
    const float scanLength = frameRoi_.GetScanLength(params_.scanDirection);
    const float spacing = arrangeLength / static_cast<float>(caliperCount);
    const float caliperWidth = std::max(1.0f, arrangeLength / static_cast<float>(caliperCount));
    const float halfScan = scanLength * 0.5f;
    const float halfWidth = caliperWidth * 0.5f;

    for (int index = 0; index < caliperCount; ++index) {
        const cv::Point2f center = frameRoi_.center + arrangeDir * ((static_cast<float>(index) + 0.5f) * spacing - arrangeLength * 0.5f);
        const cv::Point2f topLeft = center - arrangeDir * halfWidth - scanDir * halfScan;
        const cv::Point2f topRight = center + arrangeDir * halfWidth - scanDir * halfScan;
        const cv::Point2f bottomRight = center + arrangeDir * halfWidth + scanDir * halfScan;
        const cv::Point2f bottomLeft = center - arrangeDir * halfWidth + scanDir * halfScan;
        const std::array<cv::Point, 4> corners = {
            ImageToScreen(topLeft) - viewport_.drawRect.tl(),
            ImageToScreen(topRight) - viewport_.drawRect.tl(),
            ImageToScreen(bottomRight) - viewport_.drawRect.tl(),
            ImageToScreen(bottomLeft) - viewport_.drawRect.tl(),
        };
        for (size_t cornerIndex = 0; cornerIndex < corners.size(); ++cornerIndex) {
            cv::line(imageCanvas, corners[cornerIndex], corners[(cornerIndex + 1) % corners.size()], MakeColor(255, 215, 0), 1, cv::LINE_AA);
        }
    }

    const float scanArrowLength = std::min(36.0f, std::max(18.0f, scanLength * 0.22f));
    const cv::Point2f scanArrowBase = frameRoi_.center + arrangeDir * (arrangeLength * 0.12f);
    const cv::Point2f scanArrowTip = scanArrowBase + scanDir * scanArrowLength;
    const cv::Point2f scanArrowWingDir = Rotate90(scanDir) * 5.0f;
    const cv::Point scanBasePoint = ImageToScreen(scanArrowBase) - viewport_.drawRect.tl();
    const cv::Point scanTipPoint = ImageToScreen(scanArrowTip) - viewport_.drawRect.tl();
    const cv::Point scanWing1 =
        ImageToScreen(scanArrowTip - scanDir * 7.0f + scanArrowWingDir) - viewport_.drawRect.tl();
    const cv::Point scanWing2 =
        ImageToScreen(scanArrowTip - scanDir * 7.0f - scanArrowWingDir) - viewport_.drawRect.tl();
    cv::line(imageCanvas, scanBasePoint, scanTipPoint, MakeColor(0, 165, 255), 2, cv::LINE_AA);
    cv::line(imageCanvas, scanTipPoint, scanWing1, MakeColor(0, 165, 255), 2, cv::LINE_AA);
    cv::line(imageCanvas, scanTipPoint, scanWing2, MakeColor(0, 165, 255), 2, cv::LINE_AA);
}

void DemoCanvasApp::DrawArrowCaliperPreview(cv::Mat& imageCanvas) const {
    if (!arrowRoi_.axisReady) {
        return;
    }

    const cv::Point start = ImageToScreen(arrowRoi_.axisStart) - viewport_.drawRect.tl();
    const cv::Point end = ImageToScreen(arrowRoi_.axisEnd) - viewport_.drawRect.tl();
    cv::line(imageCanvas, start, end, MakeColor(255, 0, 0), 2, cv::LINE_AA);
    cv::circle(imageCanvas, start, cvRound(kArrowHandleOuterRadius), MakeColor(255, 255, 255), cv::FILLED, cv::LINE_AA);
    cv::circle(imageCanvas, start, cvRound(kArrowHandleOuterRadius), MakeColor(255, 0, 0), 2, cv::LINE_AA);
    cv::circle(imageCanvas, start, cvRound(kArrowHandleInnerRadius), MakeColor(255, 0, 0), cv::FILLED, cv::LINE_AA);
    cv::circle(imageCanvas, end, cvRound(kArrowHandleOuterRadius), MakeColor(255, 255, 255), cv::FILLED, cv::LINE_AA);
    cv::circle(imageCanvas, end, cvRound(kArrowHandleOuterRadius), MakeColor(255, 0, 0), 2, cv::LINE_AA);
    cv::circle(imageCanvas, end, cvRound(kArrowHandleInnerRadius), MakeColor(255, 0, 0), cv::FILLED, cv::LINE_AA);

    const cv::Point2f axisDir = Normalize(arrowRoi_.axisEnd - arrowRoi_.axisStart);
    const cv::Point2f arrowBase = (arrowRoi_.axisStart + arrowRoi_.axisEnd) * 0.5f;
    const cv::Point2f arrowTip = arrowBase + axisDir * kArrowHeadLength;
    const cv::Point2f arrowWingDir = Rotate90(axisDir) * 6.0f;
    const cv::Point basePoint = ImageToScreen(arrowBase) - viewport_.drawRect.tl();
    const cv::Point tipPoint = ImageToScreen(arrowTip) - viewport_.drawRect.tl();
    const cv::Point wing1 = ImageToScreen(arrowBase - axisDir * 7.0f + arrowWingDir) - viewport_.drawRect.tl();
    const cv::Point wing2 = ImageToScreen(arrowBase - axisDir * 7.0f - arrowWingDir) - viewport_.drawRect.tl();
    cv::line(imageCanvas, basePoint, tipPoint, MakeColor(255, 0, 0), 2, cv::LINE_AA);
    cv::line(imageCanvas, tipPoint, wing1, MakeColor(255, 0, 0), 2, cv::LINE_AA);
    cv::line(imageCanvas, tipPoint, wing2, MakeColor(255, 0, 0), 2, cv::LINE_AA);

    if (!arrowRoi_.lengthReady) {
        return;
    }

    const std::pair<cv::Point2f, cv::Point2f> lengthHandles = BuildArrowLengthHandlePoints();
    const cv::Point lengthHandleA = ImageToScreen(lengthHandles.first) - viewport_.drawRect.tl();
    const cv::Point lengthHandleB = ImageToScreen(lengthHandles.second) - viewport_.drawRect.tl();
    cv::circle(imageCanvas, lengthHandleA, cvRound(kArrowLengthHandleRadius), MakeColor(255, 255, 255), cv::FILLED, cv::LINE_AA);
    cv::circle(imageCanvas, lengthHandleA, cvRound(kArrowLengthHandleRadius), MakeColor(255, 0, 0), 2, cv::LINE_AA);
    cv::circle(imageCanvas, lengthHandleB, cvRound(kArrowLengthHandleRadius), MakeColor(255, 255, 255), cv::FILLED, cv::LINE_AA);
    cv::circle(imageCanvas, lengthHandleB, cvRound(kArrowLengthHandleRadius), MakeColor(255, 0, 0), 2, cv::LINE_AA);

    const cv::Point2f scanDir = ResolveArrowScanDirection(axisDir, params_.scanDirection);
    const float scanArrowLength = std::min(28.0f, std::max(16.0f, arrowRoi_.caliperLength * 0.28f));
    const cv::Point2f scanArrowBase = arrowBase + axisDir * 14.0f;
    const cv::Point2f scanArrowTip = scanArrowBase + scanDir * scanArrowLength;
    const cv::Point2f scanArrowWingDir = Rotate90(scanDir) * 5.0f;
    const cv::Point scanBasePoint = ImageToScreen(scanArrowBase) - viewport_.drawRect.tl();
    const cv::Point scanTipPoint = ImageToScreen(scanArrowTip) - viewport_.drawRect.tl();
    const cv::Point scanWing1 =
        ImageToScreen(scanArrowTip - scanDir * 7.0f + scanArrowWingDir) - viewport_.drawRect.tl();
    const cv::Point scanWing2 =
        ImageToScreen(scanArrowTip - scanDir * 7.0f - scanArrowWingDir) - viewport_.drawRect.tl();
    cv::line(imageCanvas, scanBasePoint, scanTipPoint, MakeColor(0, 165, 255), 2, cv::LINE_AA);
    cv::line(imageCanvas, scanTipPoint, scanWing1, MakeColor(0, 165, 255), 2, cv::LINE_AA);
    cv::line(imageCanvas, scanTipPoint, scanWing2, MakeColor(0, 165, 255), 2, cv::LINE_AA);

    const std::vector<LineCaliper> calipers = BuildArrowCalipers();
    for (const LineCaliper& caliper : calipers) {
        const std::vector<cv::Point2f> cornersFloat = BuildArrowCaliperCorners(caliper);
        const std::array<cv::Point, 4> corners = {
            ImageToScreen(cornersFloat[0]) - viewport_.drawRect.tl(),
            ImageToScreen(cornersFloat[1]) - viewport_.drawRect.tl(),
            ImageToScreen(cornersFloat[2]) - viewport_.drawRect.tl(),
            ImageToScreen(cornersFloat[3]) - viewport_.drawRect.tl(),
        };
        for (size_t index = 0; index < corners.size(); ++index) {
            cv::line(imageCanvas, corners[index], corners[(index + 1) % corners.size()], MakeColor(0, 215, 255), 1, cv::LINE_AA);
        }
    }
}

bool DemoCanvasApp::LoadImageFromPath(const std::string& path) {
    const cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        sourceImage_.release();
        grayImage_.release();
        return false;
    }
    sourceImage_ = image;
    cv::cvtColor(sourceImage_, grayImage_, cv::COLOR_BGR2GRAY);
    imagePath_ = path;
    ClearResult();
    ResetImageView();
    return true;
}

bool DemoCanvasApp::OpenImageWithDialog() {
    wchar_t filename[MAX_PATH] = L"";
    OPENFILENAMEW dialog = {};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = nullptr;
    dialog.lpstrFile = filename;
    dialog.nMaxFile = MAX_PATH;
    dialog.lpstrFilter = L"Images\0*.bmp;*.jpg;*.jpeg;*.png;*.tif;*.tiff\0All Files\0*.*\0";
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

    if (!GetOpenFileNameW(&dialog)) {
        return false;
    }

    return LoadImageFromPath(WideToUtf8(filename));
}

bool DemoCanvasApp::LoadNextSample() {
    if (sampleImages_.empty()) {
        return false;
    }

    for (size_t attempt = 0; attempt < sampleImages_.size(); ++attempt) {
        sampleIndex_ = (sampleIndex_ + 1) % static_cast<int>(sampleImages_.size());
        if (LoadImageFromPath(sampleImages_[sampleIndex_])) {
            return true;
        }
    }
    return false;
}

void DemoCanvasApp::DetectLine() {
    if (sourceImage_.empty() || grayImage_.empty() || !hasRoi_) {
        return;
    }

    if (roiKind_ == RoiKind::ArrowCaliper) {
        result_ = detector_.DetectGray(grayImage_, BuildArrowCalipers(), params_);
    } else if (roiKind_ == RoiKind::RotatedFrame) {
        result_ = detector_.DetectGray(grayImage_, frameRoi_, params_);
    }
    hasResult_ = true;
}

void DemoCanvasApp::ClearResult() {
    hasResult_ = false;
    result_ = LineDetectionResult();
}

void DemoCanvasApp::ClearRoi() {
    hasRoi_ = false;
    roiKind_ = RoiKind::None;
    frameRoi_ = LineDetectionFrame();
    arrowRoi_ = ArrowCaliperRoi();
    dragMode_ = DragMode::None;
    createMode_ = CreateMode::None;
    ClearResult();
}

void DemoCanvasApp::ResetCreateMode(CreateMode mode) {
    createMode_ = mode;
    dragMode_ = DragMode::None;
    ClearResult();
    if (mode == CreateMode::ArrowCaliper) {
        roiKind_ = RoiKind::ArrowCaliper;
        hasRoi_ = false;
        arrowRoi_ = ArrowCaliperRoi();
    } else if (mode == CreateMode::RotatedRectangle) {
        roiKind_ = RoiKind::RotatedFrame;
        hasRoi_ = false;
        frameRoi_ = LineDetectionFrame();
    }
}

void DemoCanvasApp::StartArrowAxisCreate(const cv::Point2f& imagePoint) {
    dragMode_ = DragMode::CreatingArrowAxis;
    dragStartImagePoint_ = imagePoint;
    arrowRoi_.axisStart = imagePoint;
    arrowRoi_.axisEnd = imagePoint;
    arrowRoi_.axisReady = true;
    arrowRoi_.lengthReady = false;
    arrowRoi_.caliperLength = 60.0f;
    arrowRoi_.caliperWidth = 20.0f;
    hasRoi_ = true;
    roiKind_ = RoiKind::ArrowCaliper;
    SyncArrowScanDirectionToAxis();
    SyncArrowCaliperWidthToAxis();
}

void DemoCanvasApp::UpdateArrowAxisCreate(const cv::Point2f& imagePoint) {
    arrowRoi_.axisEnd = imagePoint;
    SyncArrowScanDirectionToAxis();
    SyncArrowCaliperWidthToAxis();
}

void DemoCanvasApp::StartArrowLengthCreate() {
    dragMode_ = DragMode::CreatingArrowLength;
    arrowRoi_.lengthReady = false;
}

void DemoCanvasApp::UpdateArrowLengthCreate(const cv::Point2f& imagePoint) {
    const cv::Point2f axisDir = Normalize(arrowRoi_.axisEnd - arrowRoi_.axisStart);
    const cv::Point2f normal = Rotate90(axisDir);
    const cv::Point2f midpoint = (arrowRoi_.axisStart + arrowRoi_.axisEnd) * 0.5f;
    const float halfLength = std::fabs((imagePoint - midpoint).dot(normal));
    const float halfWidth = std::fabs((imagePoint - midpoint).dot(axisDir));
    arrowRoi_.caliperLength = std::max(kMinCaliperLength, halfLength * 2.0f);
    arrowRoi_.caliperWidth = std::max(kMinCaliperWidth, halfWidth * 2.0f);
    SyncArrowCaliperWidthToAxis();
    arrowRoi_.lengthReady = true;
}

void DemoCanvasApp::StartArrowWidthCreate() {
    dragMode_ = DragMode::CreatingArrowWidth;
    SyncArrowCaliperWidthToAxis();
}

void DemoCanvasApp::UpdateArrowWidthCreate(const cv::Point2f& imagePoint) {
    const cv::Point2f axisDir = Normalize(arrowRoi_.axisEnd - arrowRoi_.axisStart);
    const cv::Point2f midpoint = (arrowRoi_.axisStart + arrowRoi_.axisEnd) * 0.5f;
    const float halfWidth = std::fabs((imagePoint - midpoint).dot(axisDir));
    arrowRoi_.caliperWidth = std::max(kMinCaliperWidth, halfWidth * 2.0f);
    SyncArrowCaliperWidthToAxis();
}

void DemoCanvasApp::FinishArrowWidthCreate() {
    dragMode_ = DragMode::None;
    createMode_ = CreateMode::None;
    arrowRoi_.lengthReady = true;
    SyncArrowCaliperWidthToAxis();
}

void DemoCanvasApp::StartFrameCreate(const cv::Point2f& imagePoint) {
    dragMode_ = DragMode::CreatingFrame;
    dragStartImagePoint_ = imagePoint;
    frameRoi_ = LineDetectionFrame{imagePoint, kMinFrameSize, kMinFrameSize, 0.0f};
    hasRoi_ = true;
    roiKind_ = RoiKind::RotatedFrame;
    ClearResult();
}

void DemoCanvasApp::UpdateFrameCreate(const cv::Point2f& imagePoint) {
    const cv::Point2f topLeft(std::min(dragStartImagePoint_.x, imagePoint.x), std::min(dragStartImagePoint_.y, imagePoint.y));
    const cv::Point2f bottomRight(std::max(dragStartImagePoint_.x, imagePoint.x), std::max(dragStartImagePoint_.y, imagePoint.y));
    frameRoi_.center = (topLeft + bottomRight) * 0.5f;
    frameRoi_.width = std::max(kMinFrameSize, bottomRight.x - topLeft.x);
    frameRoi_.height = std::max(kMinFrameSize, bottomRight.y - topLeft.y);
    frameRoi_.angleDegrees = 0.0f;
}

void DemoCanvasApp::FinishFrameCreate() {
    dragMode_ = DragMode::None;
    createMode_ = CreateMode::None;
}

bool DemoCanvasApp::HitTestImagePoint(int x, int y, cv::Point2f& imagePoint) const {
    if (!viewport_.imageRect.contains(cv::Point(x, y))) {
        return false;
    }
    imagePoint = ScreenToImage(cv::Point(x, y));
    return true;
}

bool DemoCanvasApp::HitTestInsideFrame(const cv::Point2f& imagePoint, const LineDetectionFrame& frame) const {
    const cv::Point2f localPoint = ToLocal(frame, imagePoint);
    return std::fabs(localPoint.x) <= frame.width * 0.5f && std::fabs(localPoint.y) <= frame.height * 0.5f;
}

bool DemoCanvasApp::HitTestResizeHandle(const cv::Point2f& imagePoint, const LineDetectionFrame& frame, int& signX, int& signY) const {
    const std::vector<cv::Point2f> corners = frame.GetCorners();
    const std::array<std::pair<int, int>, 4> signs = {
        std::make_pair(-1, -1), std::make_pair(1, -1), std::make_pair(1, 1), std::make_pair(-1, 1)
    };

    for (size_t index = 0; index < corners.size(); ++index) {
        if (cv::norm(corners[index] - imagePoint) <= kResizeHandleRadius / viewport_.scale) {
            signX = signs[index].first;
            signY = signs[index].second;
            return true;
        }
    }

    return false;
}

bool DemoCanvasApp::HitTestArrowEndpoint(const cv::Point2f& imagePoint, bool& isStartHandle) const {
    if (!arrowRoi_.axisReady) {
        return false;
    }

    const float hitRadius = kArrowHandleOuterRadius / viewport_.scale;
    if (cv::norm(imagePoint - arrowRoi_.axisStart) <= hitRadius) {
        isStartHandle = true;
        return true;
    }
    if (cv::norm(imagePoint - arrowRoi_.axisEnd) <= hitRadius) {
        isStartHandle = false;
        return true;
    }

    return false;
}

bool DemoCanvasApp::HitTestArrowLengthHandle(const cv::Point2f& imagePoint) const {
    if (!arrowRoi_.axisReady || !arrowRoi_.lengthReady) {
        return false;
    }

    const std::pair<cv::Point2f, cv::Point2f> handles = BuildArrowLengthHandlePoints();
    const float hitRadius = kArrowLengthHandleRadius / viewport_.scale;
    return cv::norm(imagePoint - handles.first) <= hitRadius || cv::norm(imagePoint - handles.second) <= hitRadius;
}

bool DemoCanvasApp::HitTestArrowAxis(const cv::Point2f& imagePoint) const {
    if (!arrowRoi_.axisReady) {
        return false;
    }

    const cv::Point2f axis = arrowRoi_.axisEnd - arrowRoi_.axisStart;
    const float axisLengthSquared = axis.dot(axis);
    if (axisLengthSquared <= 1e-6f) {
        return false;
    }

    const float projection = (imagePoint - arrowRoi_.axisStart).dot(axis) / axisLengthSquared;
    if (projection < 0.0f || projection > 1.0f) {
        return false;
    }

    return DistancePointToLine(imagePoint, arrowRoi_.axisStart, arrowRoi_.axisEnd) <= kArrowAxisHitPadding / viewport_.scale;
}

bool DemoCanvasApp::HitTestArrowCaliperBody(const cv::Point2f& imagePoint) const {
    if (!arrowRoi_.axisReady || !arrowRoi_.lengthReady) {
        return false;
    }

    for (const LineCaliper& caliper : BuildArrowCalipers()) {
        const cv::Point2f arrangeDir = Normalize(caliper.arrangeDirection);
        const cv::Point2f scanDir = Normalize(caliper.scanDirection);
        const cv::Point2f delta = imagePoint - caliper.center;
        const float localX = delta.dot(arrangeDir);
        const float localY = delta.dot(scanDir);
        if (std::fabs(localX) <= caliper.width * 0.5f && std::fabs(localY) <= caliper.scanLength * 0.5f) {
            return true;
        }
    }

    return false;
}

cv::Point2f DemoCanvasApp::ScreenToImage(const cv::Point& screenPoint) const {
    const float localX = static_cast<float>(screenPoint.x - viewport_.imageRect.x);
    const float localY = static_cast<float>(screenPoint.y - viewport_.imageRect.y);
    return cv::Point2f(localX / viewport_.scale, localY / viewport_.scale);
}

cv::Point DemoCanvasApp::ImageToScreen(const cv::Point2f& imagePoint) const {
    const int x = viewport_.imageRect.x + cvRound(imagePoint.x * viewport_.scale);
    const int y = viewport_.imageRect.y + cvRound(imagePoint.y * viewport_.scale);
    return cv::Point(x, y);
}

cv::Point2f DemoCanvasApp::ToLocal(const LineDetectionFrame& frame, const cv::Point2f& imagePoint) const {
    const cv::Point2f delta = imagePoint - frame.center;
    return cv::Point2f(delta.dot(frame.XDirection()), delta.dot(frame.YDirection()));
}

void DemoCanvasApp::ApplySidebarClick(const cv::Point& point) {
    for (const Button& button : buttons_) {
        if (!button.rect.contains(point)) {
            continue;
        }

        switch (button.id) {
        case ButtonId::OpenImage:
            OpenImageWithDialog();
            return;
        case ButtonId::NextSample:
            LoadNextSample();
            return;
        case ButtonId::CreateRect:
            ResetCreateMode(CreateMode::ArrowCaliper);
            return;
        case ButtonId::CreateRotated:
            ResetCreateMode(CreateMode::RotatedRectangle);
            return;
        case ButtonId::Detect:
            DetectLine();
            return;
        case ButtonId::ClearRoi:
            ClearRoi();
            return;
        case ButtonId::ClearResult:
            ClearResult();
            return;
        }
    }

    for (size_t index = 0; index < parameterControls_.size(); ++index) {
        const ParamControl& control = parameterControls_[index];
        if (control.minusRect.contains(point)) {
            UpdateParameterFromControl(index, false);
            return;
        }
        if (control.plusRect.contains(point) || control.valueRect.contains(point)) {
            UpdateParameterFromControl(index, true);
            return;
        }
    }
}

void DemoCanvasApp::UpdateParameterFromControl(size_t paramIndex, bool increment) {
    switch (paramIndex) {
    case 0:
        params_.caliperCount = std::max(2, params_.caliperCount + (increment ? 1 : -1));
        SyncArrowCaliperWidthToAxis();
        break;
    case 1: {
        params_.sampleStep = std::max(1, params_.sampleStep + (increment ? 1 : -1));
        break;
    }
    case 2: {
        params_.filterSize = std::max(1, params_.filterSize + (increment ? 2 : -2));
        if (params_.filterSize % 2 == 0) {
            params_.filterSize += increment ? 1 : -1;
        }
        break;
    }
    case 3:
        params_.edgeThreshold = std::max(0, params_.edgeThreshold + (increment ? 2 : -2));
        break;
    case 4: {
        params_.edgePolarity = CycleEdgePolarity(params_.edgePolarity, increment);
        break;
    }
    case 5: {
        params_.selectionMode = CycleSelectionMode(params_.selectionMode, increment);
        break;
    }
    case 6: {
        if (roiKind_ == RoiKind::ArrowCaliper && arrowRoi_.axisReady) {
            SyncArrowScanDirectionToAxis();
            const std::pair<LineScanDirection, LineScanDirection> validDirections = GetArrowValidScanDirections();
            params_.scanDirection =
                (params_.scanDirection == validDirections.first) ? validDirections.second : validDirections.first;
        } else {
            params_.scanDirection = CycleScanDirection(params_.scanDirection, increment);
        }
        break;
    }
    case 7: {
        params_.fitMode = CycleFitMode(params_.fitMode, increment);
        break;
    }
    case 8:
        params_.excludPts = std::max(0, params_.excludPts + (increment ? 1 : -1));
        break;
    default:
        break;
    }
    ClearResult();
}

void DemoCanvasApp::ResetImageView() {
    zoomFactor_ = 1.0f;
    imageOffset_ = cv::Point2f(0.0f, 0.0f);
    viewNeedsReset_ = true;
    hasMouseImagePixel_ = false;
}

void DemoCanvasApp::ClampImageOffset(const cv::Size& drawSize) {
    const float panelLeft = static_cast<float>(imagePanelRect_.x);
    const float panelTop = static_cast<float>(imagePanelRect_.y);
    const float panelRight = static_cast<float>(imagePanelRect_.x + imagePanelRect_.width);
    const float panelBottom = static_cast<float>(imagePanelRect_.y + imagePanelRect_.height);

    if (drawSize.width <= imagePanelRect_.width) {
        imageOffset_.x = panelLeft + static_cast<float>(imagePanelRect_.width - drawSize.width) * 0.5f;
    } else {
        imageOffset_.x = std::min(panelLeft, std::max(panelRight - static_cast<float>(drawSize.width), imageOffset_.x));
    }

    if (drawSize.height <= imagePanelRect_.height) {
        imageOffset_.y = panelTop + static_cast<float>(imagePanelRect_.height - drawSize.height) * 0.5f;
    } else {
        imageOffset_.y = std::min(panelTop, std::max(panelBottom - static_cast<float>(drawSize.height), imageOffset_.y));
    }
}

void DemoCanvasApp::ZoomAtScreenPoint(const cv::Point& screenPoint, float zoomMultiplier) {
    if (sourceImage_.empty() || viewport_.scale <= 1e-6f || viewport_.baseScale <= 1e-6f) {
        return;
    }

    const float oldScale = viewport_.scale;
    const float newZoomFactor = std::max(kMinZoomFactor, std::min(kMaxZoomFactor, zoomFactor_ * zoomMultiplier));
    if (std::fabs(newZoomFactor - zoomFactor_) <= 1e-6f) {
        return;
    }

    const float newScale = viewport_.baseScale * newZoomFactor;
    const cv::Point2f anchorImagePoint = ScreenToImage(screenPoint);
    zoomFactor_ = newZoomFactor;
    imageOffset_.x = static_cast<float>(screenPoint.x) - anchorImagePoint.x * newScale;
    imageOffset_.y = static_cast<float>(screenPoint.y) - anchorImagePoint.y * newScale;

    const cv::Size drawSize(
        std::max(1, cvRound(static_cast<float>(sourceImage_.cols) * newScale)),
        std::max(1, cvRound(static_cast<float>(sourceImage_.rows) * newScale)));
    ClampImageOffset(drawSize);
}

std::pair<LineScanDirection, LineScanDirection> DemoCanvasApp::GetArrowValidScanDirections() const {
    if (!arrowRoi_.axisReady) {
        return std::make_pair(LineScanDirection::TopToBottom, LineScanDirection::BottomToTop);
    }

    const cv::Point2f axisDir = Normalize(arrowRoi_.axisEnd - arrowRoi_.axisStart);
    const cv::Point2f normal = Rotate90(axisDir);
    if (std::fabs(normal.y) >= std::fabs(normal.x)) {
        return normal.y >= 0.0f
            ? std::make_pair(LineScanDirection::TopToBottom, LineScanDirection::BottomToTop)
            : std::make_pair(LineScanDirection::BottomToTop, LineScanDirection::TopToBottom);
    }

    return normal.x >= 0.0f
        ? std::make_pair(LineScanDirection::LeftToRight, LineScanDirection::RightToLeft)
        : std::make_pair(LineScanDirection::RightToLeft, LineScanDirection::LeftToRight);
}

void DemoCanvasApp::SyncArrowScanDirectionToAxis() {
    if (roiKind_ != RoiKind::ArrowCaliper || !arrowRoi_.axisReady) {
        return;
    }

    const std::pair<LineScanDirection, LineScanDirection> validDirections = GetArrowValidScanDirections();
    if (params_.scanDirection != validDirections.first && params_.scanDirection != validDirections.second) {
        params_.scanDirection = validDirections.first;
    }
}

void DemoCanvasApp::SyncArrowCaliperWidthToAxis() {
    if (roiKind_ != RoiKind::ArrowCaliper || !arrowRoi_.axisReady) {
        return;
    }

    arrowRoi_.caliperWidth = std::min(ComputeMaxArrowCaliperWidth(), std::max(kMinCaliperWidth, arrowRoi_.caliperWidth));
}

std::vector<LineCaliper> DemoCanvasApp::BuildArrowCalipers() const {
    std::vector<LineCaliper> calipers;
    if (!arrowRoi_.axisReady || !arrowRoi_.lengthReady) {
        return calipers;
    }

    const int caliperCount = std::max(2, params_.caliperCount);
    const cv::Point2f axisDir = Normalize(arrowRoi_.axisEnd - arrowRoi_.axisStart);
    const cv::Point2f scanDir = ResolveArrowScanDirection(axisDir, params_.scanDirection);
    const float axisLength = ComputeArrowAxisLength();
    const float caliperWidth = ComputeArrowCaliperWidth();
    const cv::Point2f start = arrowRoi_.axisStart;
    const float gap = caliperCount > 1
        ? std::max(0.0f, (axisLength - caliperWidth * static_cast<float>(caliperCount)) / static_cast<float>(caliperCount - 1))
        : 0.0f;

    calipers.reserve(caliperCount);
    for (int index = 0; index < caliperCount; ++index) {
        const float offset = caliperWidth * 0.5f + static_cast<float>(index) * (caliperWidth + gap);
        LineCaliper caliper;
        caliper.center = start + axisDir * offset;
        caliper.scanDirection = scanDir;
        caliper.arrangeDirection = axisDir;
        caliper.scanLength = arrowRoi_.caliperLength;
        caliper.width = caliperWidth;
        calipers.push_back(caliper);
    }

    return calipers;
}

std::vector<cv::Point2f> DemoCanvasApp::BuildArrowCaliperCorners(const LineCaliper& caliper) const {
    const cv::Point2f scanDir = Normalize(caliper.scanDirection);
    const cv::Point2f arrangeDir = Normalize(caliper.arrangeDirection);
    const float halfScan = caliper.scanLength * 0.5f;
    const float halfWidth = caliper.width * 0.5f;
    return {
        caliper.center - arrangeDir * halfWidth - scanDir * halfScan,
        caliper.center + arrangeDir * halfWidth - scanDir * halfScan,
        caliper.center + arrangeDir * halfWidth + scanDir * halfScan,
        caliper.center - arrangeDir * halfWidth + scanDir * halfScan,
    };
}

std::vector<cv::Point2f> DemoCanvasApp::BuildActiveRoiBoundary() const {
    if (!hasRoi_) {
        return {};
    }

    if (roiKind_ == RoiKind::RotatedFrame) {
        return frameRoi_.GetCorners();
    }

    if (roiKind_ == RoiKind::ArrowCaliper && arrowRoi_.axisReady && arrowRoi_.lengthReady) {
        const cv::Point2f axisDir = Normalize(arrowRoi_.axisEnd - arrowRoi_.axisStart);
        const cv::Point2f normal = Rotate90(axisDir);
        const float halfLength = ComputeArrowHalfLength();
        return {
            arrowRoi_.axisStart - normal * halfLength,
            arrowRoi_.axisEnd - normal * halfLength,
            arrowRoi_.axisEnd + normal * halfLength,
            arrowRoi_.axisStart + normal * halfLength,
        };
    }

    return {};
}

bool DemoCanvasApp::ComputeDisplayedResultLine(cv::Point2f& start, cv::Point2f& end) const {
    if (!hasResult_ || !result_.success) {
        return false;
    }

    const cv::Point2f lineStart = result_.lineStart;
    const cv::Point2f lineEnd = result_.lineEnd;
    const cv::Point2f lineDirection = lineEnd - lineStart;
    if (cv::norm(lineDirection) <= 1e-6f) {
        start = lineStart;
        end = lineEnd;
        return true;
    }

    const std::vector<cv::Point2f> boundary = BuildActiveRoiBoundary();
    if (boundary.size() < 2) {
        start = lineStart;
        end = lineEnd;
        return true;
    }

    struct IntersectionHit {
        cv::Point2f point;
        float t = 0.0f;
    };

    std::vector<IntersectionHit> hits;
    hits.reserve(boundary.size());
    for (size_t index = 0; index < boundary.size(); ++index) {
        cv::Point2f intersection;
        float lineT = 0.0f;
        if (!IntersectLineWithSegment(
                lineStart,
                lineDirection,
                boundary[index],
                boundary[(index + 1) % boundary.size()],
                intersection,
                lineT)) {
            continue;
        }

        const auto existing = std::find_if(hits.begin(), hits.end(), [&](const IntersectionHit& hit) {
            return IsSamePoint(hit.point, intersection);
        });
        if (existing == hits.end()) {
            hits.push_back({intersection, lineT});
        }
    }

    if (hits.size() < 2) {
        start = lineStart;
        end = lineEnd;
        return true;
    }

    std::sort(hits.begin(), hits.end(), [](const IntersectionHit& left, const IntersectionHit& right) {
        return left.t < right.t;
    });
    start = hits.front().point;
    end = hits.back().point;
    return true;
}

std::pair<cv::Point2f, cv::Point2f> DemoCanvasApp::BuildArrowLengthHandlePoints() const {
    const cv::Point2f axisDir = Normalize(arrowRoi_.axisEnd - arrowRoi_.axisStart);
    const cv::Point2f normal = Rotate90(axisDir);
    const cv::Point2f midpoint = (arrowRoi_.axisStart + arrowRoi_.axisEnd) * 0.5f;
    const float halfLength = ComputeArrowHalfLength();
    return std::make_pair(midpoint + normal * halfLength, midpoint - normal * halfLength);
}

float DemoCanvasApp::ComputeArrowAxisLength() const {
    return static_cast<float>(cv::norm(arrowRoi_.axisEnd - arrowRoi_.axisStart));
}

float DemoCanvasApp::ComputeArrowHalfLength() const {
    return arrowRoi_.caliperLength * 0.5f;
}

float DemoCanvasApp::ComputeArrowCaliperWidth() const {
    return std::min(ComputeMaxArrowCaliperWidth(), std::max(kMinCaliperWidth, arrowRoi_.caliperWidth));
}

float DemoCanvasApp::ComputeMaxArrowCaliperWidth() const {
    const int caliperCount = std::max(2, params_.caliperCount);
    return std::max(kMinCaliperWidth, ComputeArrowAxisLength() / static_cast<float>(caliperCount));
}

std::string DemoCanvasApp::BuildStatusText() const {
    if (sourceImage_.empty()) {
        return "No image loaded";
    }
    if (!hasRoi_) {
        return "Image loaded. Create arrow calipers or a rotated ROI.";
    }
    if (createMode_ == CreateMode::ArrowCaliper && dragMode_ == DragMode::CreatingArrowAxis) {
        return "Arrow calipers: drag the main axis.";
    }
    if (createMode_ == CreateMode::ArrowCaliper && dragMode_ == DragMode::CreatingArrowLength) {
        return "Arrow calipers: drag to set scan length and width, then release.";
    }
    if (createMode_ == CreateMode::ArrowCaliper && dragMode_ == DragMode::CreatingArrowWidth) {
        return "Arrow calipers: drag along the main axis to set each caliper width, then release.";
    }
    if (dragMode_ == DragMode::ResizingArrowStart || dragMode_ == DragMode::ResizingArrowEnd) {
        return "Arrow calipers: drag an endpoint handle to change the axis length.";
    }
    if (dragMode_ == DragMode::ResizingArrowLength) {
        return "Arrow calipers: drag a side handle to change scan length and width.";
    }
    if (!hasResult_) {
        return "ROI ready. Run line detection.";
    }

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2);
    const int rawPoints = result_.rawPointCount > 0 ? result_.rawPointCount : static_cast<int>(result_.edgePoints.size());
    const int fitPoints = result_.fitPointCount > 0 ? result_.fitPointCount : static_cast<int>(result_.edgePoints.size());
    stream << (result_.success ? "Success" : "Failure")
           << " | " << result_.message
           << " | points: " << rawPoints << "->" << fitPoints
           << " | times:" << result_.elapsedMilliseconds << " ms";
    return stream.str();
}

std::string DemoCanvasApp::BuildStatusDetailText() const {
    if (!hasResult_) {
        return std::string();
    }

    std::ostringstream stream;
    const std::vector<std::pair<std::string, std::string>> columns = BuildStatusDetailColumns();
    for (size_t index = 0; index < columns.size(); ++index) {
        if (index > 0) {
            stream << " | ";
        }
        stream << columns[index].first << ": " << columns[index].second;
    }
    return stream.str();
}

std::vector<std::pair<std::string, std::string>> DemoCanvasApp::BuildStatusDetailColumns() const {
    if (!hasResult_) {
        return {};
    }

    cv::Point2f displayStart = result_.lineStart;
    cv::Point2f displayEnd = result_.lineEnd;
    ComputeDisplayedResultLine(displayStart, displayEnd);

    std::ostringstream startXStream;
    std::ostringstream startYStream;
    std::ostringstream endXStream;
    std::ostringstream endYStream;
    std::ostringstream angleStream;
    std::ostringstream fitErrorStream;
    std::ostringstream fitErrorRmseStream;

    startXStream << std::fixed << std::setprecision(2) << displayStart.x;
    startYStream << std::fixed << std::setprecision(2) << displayStart.y;
    endXStream << std::fixed << std::setprecision(2) << displayEnd.x;
    endYStream << std::fixed << std::setprecision(2) << displayEnd.y;
    angleStream << std::fixed << std::setprecision(4) << result_.angleDegrees;
    fitErrorStream << std::fixed << std::setprecision(8) << result_.fitError;
    fitErrorRmseStream << std::fixed << std::setprecision(8) << result_.fitErrorRmse;

    return {
        {"startX", startXStream.str()},
        {"startY", startYStream.str()},
        {"endX", endXStream.str()},
        {"endY", endYStream.str()},
        {"angle", angleStream.str()},
        {"fitError(MAE)", fitErrorStream.str()},
        {"fitError(RMSE)", fitErrorRmseStream.str()},
    };
}

std::string DemoCanvasApp::BuildHelpText() const {
    //return "Shortcuts:\nO open image, N next sample\nR arrow calipers, T rotated ROI\nArrow mode: drag axis, then drag length\nQ/E rotate only for rotated ROI\nD detect, C clear result, X clear ROI";
    return "";
}
