#include "common/utils.hpp"
#include <iostream>

namespace vutils {

void ensureDir(const std::string& path) {
    std::filesystem::create_directories(path);
}

void drawLabel(cv::Mat& img, const std::string& text,
               const cv::Point& org, const cv::Scalar& color,
               double scale, int thickness) {
    int baseline = 0;
    cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                  scale, thickness, &baseline);
    cv::rectangle(img,
                  cv::Point(org.x, org.y - sz.height - 4),
                  cv::Point(org.x + sz.width + 4, org.y + baseline),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX,
                scale, color, thickness);
}

double dist(const cv::Point2d& a, const cv::Point2d& b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

cv::Point2d contourCenter(const std::vector<cv::Point>& cnt) {
    cv::Moments m = cv::moments(cnt);
    if (m.m00 == 0) return cv::Point2d(-1, -1);
    return cv::Point2d(m.m10 / m.m00, m.m01 / m.m00);
}

}