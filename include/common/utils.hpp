#pragma once
#include <limits>
#include <new>
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>

namespace vutils { // 👈 使用 vutils，避开 OpenCV 的 cv::utils

    void ensureDir(const std::string& path);

    void drawLabel(cv::Mat& img, const std::string& text,
                   const cv::Point& org, const cv::Scalar& color,
                   double scale = 0.6, int thickness = 1);

    double dist(const cv::Point2d& a, const cv::Point2d& b);

    cv::Point2d contourCenter(const std::vector<cv::Point>& cnt);
}