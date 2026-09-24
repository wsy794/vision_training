#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include "common/utils.hpp"

using namespace cv;

enum class TrackState { DETECTED, LOST };

struct Target {
    int id = -1;
    Point2d center{-1, -1};
    double radius = 0;
    TrackState state = TrackState::LOST;
    int lostFrames = 0;
};

const int MAX_LOST_FRAMES = 15;

Point2d detectRCenter(const Mat& frame, double& outRadius) {
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    // R 标是橙色的，所以直接在 HSV 里提取橙色
    Mat mask;
    // 这里使用和扇叶一样的橙色范围，确保能抓到 R 标
    inRange(hsv, Scalar(0, 100, 100), Scalar(25, 255, 255), mask);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    std::vector<std::vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Point2d imgCenter(frame.cols / 2.0, frame.rows / 2.0);
    double bestArea = 0;
    Point2d best(-1, -1);
    outRadius = 0;

    for (const auto& cnt : contours) {
        double area = contourArea(cnt);
        // 1. 过滤面积：太小是噪点，太大是扇叶本身（R标通常不大）
        if (area < 80 || area > 5000) continue;

        double peri = arcLength(cnt, true);
        if (peri < 1e-6) continue;
        
        // 2. 要求轮廓足够圆
        double circularity = 4 * M_PI * area / (peri * peri);
        if (circularity < 0.5) continue;

        Point2d c = vutils::contourCenter(cnt);

        // 3. 核心约束：R标一定在画面正中心附近，不可能跑到边缘！
        // 限制在画面中间 30% 的区域内，如果你发现太严格可以改成 0.5
        if (vutils::dist(c, imgCenter) > frame.cols * 0.3) continue;

        if (area > bestArea) {
            bestArea = area;
            best = c;
            outRadius = std::sqrt(area / M_PI);
        }
    }
    return best;
}
std::vector<std::pair<Point2d, double>>
detectTargets(const Mat& frame, const Point2d& rCenter, double rRadius,
              const Scalar& hsvLow, const Scalar& hsvHigh) {
    std::vector<std::pair<Point2d, double>> result;

    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    Mat mask;
    inRange(hsv, hsvLow, hsvHigh, mask);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    std::vector<std::vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& cnt : contours) {
        double area = contourArea(cnt);
        if (area < 300) continue;

        Point2d c = vutils::contourCenter(cnt);
        double d = vutils::dist(c, rCenter);
        if (d < rRadius * 0.5 || d > rRadius * 3.0) continue;

        double radius = std::sqrt(area / M_PI);
        result.emplace_back(c, radius);
    }
    return result;
}

int main(int argc, char** argv) {
    std::string videoPath = "resources/task_3.mp4";
    std::string outDir = "result/task3_windmill/task_3";
    if (argc > 1) videoPath = argv[1];
    if (argc > 2) outDir = argv[2];

    vutils::ensureDir(outDir);

    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << videoPath << std::endl;
        return 1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    int width  = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int totalFrames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    std::cout << "Video: " << width << "x" << height
              << ", FPS=" << fps
              << ", frames=" << totalFrames << std::endl;

    VideoWriter writer(outDir + "/recognition_overlay.mp4",
                       VideoWriter::fourcc('m', 'p', '4', 'v'),
                       fps, Size(width, height));
    VideoWriter writerBin(outDir + "/binary_process.mp4",
                          VideoWriter::fourcc('m', 'p', '4', 'v'),
                          fps, Size(width, height));

    Scalar hsvLowRed1(0, 50, 50),   hsvHighRed1(25, 255, 255);
    Scalar hsvLowRed2(170, 100, 100), hsvHighRed2(179, 255, 255);
    Scalar hsvLowCyan(80, 100, 100),  hsvHighCyan(100, 255, 255);

    Target tracked;
    int nextId = 0;

    Mat frame;
    int frameIdx = 0;

    while (cap.read(frame)) {
        if (frame.empty()) break;

        double rRadius = 0;
Point2d rCenter = detectRCenter(frame, rRadius);

// 👇 新增：R标中心连续性约束
static Point2d lastRCenter(-1, -1);
static int rCenterLostFrames = 0;
if (rCenter.x > 0) {
    lastRCenter = rCenter;
    rCenterLostFrames = 0;
} else {
    rCenterLostFrames++;
    // 如果连续丢失不超过30帧，就沿用上一帧的中心点
    if (rCenterLostFrames < 30 && lastRCenter.x > 0) {
        rCenter = lastRCenter; 
    }
}
// 👆 新增结束

        std::vector<std::pair<Point2d, double>> targets;
        if (rCenter.x > 0 && rRadius > 0) {
            auto red1 = detectTargets(frame, rCenter, rRadius, hsvLowRed1, hsvHighRed1);
            auto red2 = detectTargets(frame, rCenter, rRadius, hsvLowRed2, hsvHighRed2);
            auto cyan = detectTargets(frame, rCenter, rRadius, hsvLowCyan, hsvHighCyan);
            targets.insert(targets.end(), red1.begin(), red1.end());
            targets.insert(targets.end(), red2.begin(), red2.end());
            targets.insert(targets.end(), cyan.begin(), cyan.end());
        }

        Mat overlay = frame.clone();
        Mat binaryVis = Mat::zeros(frame.size(), CV_8UC3);

        if (rCenter.x > 0 && rRadius > 0) {
            circle(overlay, rCenter, static_cast<int>(rRadius), Scalar(255, 255, 0), 2);
            circle(overlay, rCenter, 3, Scalar(0, 0, 255), -1);
            vutils::drawLabel(overlay, "R-Center", Point(rCenter.x + 5, rCenter.y - 5), Scalar(255, 255, 0));
        }

        bool matched = false;
        if (tracked.id >= 0 && tracked.center.x > 0) {
            double bestDist = 1e9;
            int bestIdx = -1;
            for (size_t i = 0; i < targets.size(); i++) {
                double d = vutils::dist(targets[i].first, tracked.center);
                if (d < bestDist) {
                    bestDist = d;
                    bestIdx = static_cast<int>(i);
                }
            }
            if (bestIdx >= 0 && bestDist < 150.0) {
                tracked.center = targets[bestIdx].first;
                tracked.radius = targets[bestIdx].second;
                tracked.state = TrackState::DETECTED;
                tracked.lostFrames = 0;
                matched = true;
            }
        }

        if (!matched) {
            if (tracked.id < 0 || tracked.lostFrames >= MAX_LOST_FRAMES) {
                if (!targets.empty()) {
                    double bestScore = -1e9;
                    int bestIdx = 0;
                    for (size_t i = 0; i < targets.size(); i++) {
                        double d = vutils::dist(targets[i].first, rCenter);
                        double score = targets[i].second * 100 - d * 0.1;
                        if (score > bestScore) {
                            bestScore = score;
                            bestIdx = static_cast<int>(i);
                        }
                    }
                    tracked.id = nextId++;
                    tracked.center = targets[bestIdx].first;
                    tracked.radius = targets[bestIdx].second;
                    tracked.state = TrackState::DETECTED;
                    tracked.lostFrames = 0;
                } else {
                    if (tracked.id >= 0) {
                        tracked.lostFrames++;
                        tracked.state = TrackState::LOST;
                    }
                }
            } else {
                tracked.lostFrames++;
                tracked.state = TrackState::LOST;
            }
        }

        if (tracked.id >= 0) {
            if (tracked.center.x > 0) {
                circle(overlay, tracked.center, static_cast<int>(tracked.radius), Scalar(0, 255, 0), 2);
                circle(overlay, tracked.center, 4, Scalar(0, 255, 0), -1);
            }
            if (rCenter.x > 0 && tracked.center.x > 0) {
                line(overlay, rCenter, tracked.center, Scalar(0, 165, 255), 2);
            }

            char buf[128];
            const char* stateStr = (tracked.state == TrackState::DETECTED) ? "detected" : "lost";
            snprintf(buf, sizeof(buf), "ID=%d  State=%s", tracked.id, stateStr);
            Scalar color = (tracked.state == TrackState::DETECTED) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
            vutils::drawLabel(overlay, buf, Point(20, 40), color, 0.8, 2);
        }

        {
            Mat hsv;
            cvtColor(frame, hsv, COLOR_BGR2HSV);
            Mat m1, m2, m3;
            inRange(hsv, hsvLowRed1, hsvHighRed1, m1);
            inRange(hsv, hsvLowRed2, hsvHighRed2, m2);
            inRange(hsv, hsvLowCyan, hsvHighCyan, m3);
            Mat all = m1 | m2 | m3;
            Mat allBGR;
            cvtColor(all, allBGR, COLOR_GRAY2BGR);
            binaryVis = allBGR;
            if (rCenter.x > 0) {
                circle(binaryVis, rCenter, 3, Scalar(0, 255, 255), -1);
            }
        }

        writer.write(overlay);
        writerBin.write(binaryVis);
        frameIdx++;
    }
    cap.release();
    writer.release();
    writerBin.release();

    std::cout << "Task 3 done. Output: " << outDir << std::endl;
    return 0;
}