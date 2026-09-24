#include <limits>
#include <new>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "common/utils.hpp"

using namespace cv;

int main() {
    vutils::ensureDir("result/task1_images"); // 👈 改成 vutils

    // ===== 1. 读图与颜色转换 =====
    Mat img = imread("resources/test_image.jpg");
    if (img.empty()) {
        std::cerr << "Cannot read resources/test_image.jpg" << std::endl;
        return 1;
    }
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    imwrite("result/task1_images/gray.png", gray);

    Mat hsv, hsvCh[3];
    cvtColor(img, hsv, COLOR_BGR2HSV);
    split(hsv, hsvCh);
    imwrite("result/task1_images/hsv_h.png", hsvCh[0]);
    imwrite("result/task1_images/hsv_s.png", hsvCh[1]);
    imwrite("result/task1_images/hsv_v.png", hsvCh[2]);

    // ===== 2. 滤波对比 =====
    Mat meanImg, gaussianImg, medianImg;
    blur(img, meanImg, Size(5, 5));
    GaussianBlur(img, gaussianImg, Size(5, 5), 1.5);
    medianBlur(img, medianImg, 5);
    imwrite("result/task1_images/mean_filter.png", meanImg);
    imwrite("result/task1_images/gaussian_filter.png", gaussianImg);
    imwrite("result/task1_images/median_filter.png", medianImg);

    // ===== 3. 红色提取 =====
    Mat maskLow, maskHigh, redMask;
    inRange(hsv, Scalar(0, 100, 100), Scalar(10, 255, 255), maskLow);
    inRange(hsv, Scalar(170, 100, 100), Scalar(179, 255, 255), maskHigh);
    bitwise_or(maskLow, maskHigh, redMask);
    imwrite("result/task1_images/red_mask.png", redMask);

    // ===== 4. 形态学与轮廓 =====
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat eroded, dilated, opened, closed;
    erode(redMask, eroded, kernel);
    dilate(redMask, dilated, kernel);
    morphologyEx(redMask, opened, MORPH_OPEN, kernel);
    morphologyEx(redMask, closed, MORPH_CLOSE, kernel);
    imwrite("result/task1_images/erode.png", eroded);
    imwrite("result/task1_images/dilate.png", dilated);
    imwrite("result/task1_images/open.png", opened);
    imwrite("result/task1_images/close.png", closed);

    std::vector<std::vector<Point>> contours;
    findContours(closed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat result = img.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area < 500.0) continue;
        Rect box = boundingRect(contours[i]);
        double ratio = static_cast<double>(box.width) / box.height;
        if (ratio < 0.2 || ratio > 5.0) continue;
        rectangle(result, box, Scalar(0, 0, 255), 2);
        drawContours(result, contours, static_cast<int>(i), Scalar(0, 255, 0), 2);
    }
    imwrite("result/task1_images/contours_boxes.png", result);

    // ===== 5. 绘制与变换 =====
    Mat drawing = img.clone();
    circle(drawing, Point(drawing.cols / 2, drawing.rows / 2), 100, Scalar(255, 0, 0), 3);
    rectangle(drawing, Rect(50, 50, 200, 150), Scalar(0, 255, 0), 3);
    putText(drawing, "OpenCV Demo", Point(50, 250), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
    imwrite("result/task1_images/drawing.png", drawing);

    Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    Mat rotMat = getRotationMatrix2D(center, 35.0, 1.0);
    Mat rotated;
    warpAffine(img, rotated, rotMat, img.size());
    imwrite("result/task1_images/rotated_35deg.png", rotated);

    Mat cropped = img(Rect(0, 0, img.cols / 2, img.rows / 2)).clone();
    imwrite("result/task1_images/crop_top_left.png", cropped);

    std::cout << "Task 1 done. 16 images saved." << std::endl;
    return 0;
}