#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "common/utils.hpp"

using namespace cv;

// 角速度模型: omega(t) = b + A * sin(Omega * t + phi)
struct AngularVelocityResidual {
    AngularVelocityResidual(double t, double omega_obs)
        : t_(t), omega_obs_(omega_obs) {}

    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        const T& A     = params[0];
        const T& b     = params[1];
        const T& Omega = params[2];
        const T& phi   = params[3];

        T omega_pred = b + A * ceres::sin(Omega * T(t_) + phi);
        residual[0] = T(omega_obs_) - omega_pred;
        return true;
    }

    double t_, omega_obs_;
};

int main(int argc, char** argv) {
    std::string videoPath = "resources/task_2.mp4";
    if (argc > 1) videoPath = argv[1];

    vutils::ensureDir("result/task2_fit");

    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << videoPath << std::endl;
        return 1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    int width  = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    std::cout << "Video: " << width << "x" << height
              << ", FPS=" << fps
              << ", frames=" << totalFrames << std::endl;

    // 已知旋转中心
    Point2d center(480.0, 360.0);

    std::vector<double> times;
    std::vector<double> anglesWrapped;
    std::vector<double> anglesUnwrapped;
    std::vector<Point2d> targetCenters;

    VideoWriter writer("result/task2_fit/tracking_overlay.mp4",
                       VideoWriter::fourcc('m', 'p', '4', 'v'),
                       fps, Size(width, height));

    Mat frame;
    int frameIdx = 0;

    while (cap.read(frame)) {
        if (frame.empty()) break;

        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // 青色范围（可根据实际视频微调）
        Mat mask;
        inRange(hsv, Scalar(80, 100, 100), Scalar(100, 255, 255), mask);

        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);

        std::vector<std::vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Point2d targetCenter(-1, -1);
        double maxArea = 0;
        for (const auto& cnt : contours) {
            double area = contourArea(cnt);
            if (area > maxArea) {
                maxArea = area;
                targetCenter = vutils::contourCenter(cnt);
            }
        }

        double t = frameIdx / fps;
        double angle = 0;
        if (targetCenter.x > 0) {
            angle = std::atan2(center.y - targetCenter.y,
                               targetCenter.x - center.x);
            targetCenters.push_back(targetCenter);
        } else {
            targetCenters.push_back(Point2d(-1, -1));
        }

        times.push_back(t);
        anglesWrapped.push_back(angle);

        // 可视化
        Mat overlay = frame.clone();
        circle(overlay, center, 6, Scalar(0, 0, 255), -1);
        if (targetCenter.x > 0) {
            circle(overlay, targetCenter, 6, Scalar(0, 255, 0), -1);
            line(overlay, center, targetCenter, Scalar(255, 0, 0), 2);

            char buf[128];
            snprintf(buf, sizeof(buf), "t=%.2fs theta=%.3f", t, angle);
            vutils::drawLabel(overlay, buf, Point(20, 40), Scalar(0, 255, 255));
        } else {
            vutils::drawLabel(overlay, "TARGET LOST", Point(20, 40), Scalar(0, 0, 255));
        }
        writer.write(overlay);
        frameIdx++;
    }
    cap.release();
    writer.release();

    // 展开角度
    size_t N = anglesWrapped.size();
    anglesUnwrapped.resize(N);
    if (N > 0) anglesUnwrapped[0] = anglesWrapped[0];
    for (size_t i = 1; i < N; i++) {
        double diff = anglesWrapped[i] - anglesWrapped[i - 1];
        while (diff >  M_PI) diff -= 2 * M_PI;
        while (diff < -M_PI) diff += 2 * M_PI;
        anglesUnwrapped[i] = anglesUnwrapped[i - 1] + diff;
    }

    // 数值中心差分求角速度
    std::vector<double> omegaObs, tOmega;
    for (size_t i = 1; i + 1 < N; i++) {
        double dt = times[i + 1] - times[i - 1];
        double dtheta = anglesUnwrapped[i + 1] - anglesUnwrapped[i - 1];
        omegaObs.push_back(dtheta / dt);
        tOmega.push_back(times[i]);
    }

    // ===== Ceres 拟合 =====
    double params[4] = {0.5, 1.0, 1.0, 0.0};  // A, b, Omega, phi

    ceres::Problem problem;
    for (size_t i = 0; i < omegaObs.size(); i++) {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<AngularVelocityResidual, 1, 4>(
                new AngularVelocityResidual(tOmega[i], omegaObs[i]));
        problem.AddResidualBlock(cost, nullptr, params);
    }

    problem.SetParameterLowerBound(params, 0, 0.0);
    problem.SetParameterLowerBound(params, 2, 0.01);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 500;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    std::cout << "A     = " << params[0] << " rad/s" << std::endl;
    std::cout << "b     = " << params[1] << " rad/s" << std::endl;
    std::cout << "Omega = " << params[2] << " rad/s" << std::endl;
    std::cout << "phi   = " << params[3] << " rad" << std::endl;

    // 计算 RMSE
    double sse = 0;
    std::vector<double> omegaFit;
    for (size_t i = 0; i < omegaObs.size(); i++) {
        double pred = params[1] +
                      params[0] * std::sin(params[2] * tOmega[i] + params[3]);
        omegaFit.push_back(pred);
        sse += (omegaObs[i] - pred) * (omegaObs[i] - pred);
    }
    double rmse = std::sqrt(sse / omegaObs.size());
    std::cout << "RMSE = " << rmse << " rad/s" << std::endl;

    // ===== 绘制拟合对比图 =====
    const int plotW = 1200, plotH = 600;
    Mat plot(plotH, plotW, CV_8UC3, Scalar(255, 255, 255));

    double minT = tOmega.front(), maxT = tOmega.back();
    double minO = *std::min_element(omegaObs.begin(), omegaObs.end());
    double maxO = *std::max_element(omegaObs.begin(), omegaObs.end());
    minO = std::min(minO, *std::min_element(omegaFit.begin(), omegaFit.end()));
    maxO = std::max(maxO, *std::max_element(omegaFit.begin(), omegaFit.end()));
    double margin = (maxO - minO) * 0.15 + 1e-6;
    minO -= margin; maxO += margin;

    auto toPlot = [&](double t, double o) {
        int px = static_cast<int>((t - minT) / (maxT - minT + 1e-9) *
                                  (plotW - 120) + 60);
        int py = static_cast<int>((maxO - o) / (maxO - minO + 1e-9) *
                                  (plotH - 120) + 60);
        return Point(px, py);
    };

    line(plot, Point(60, plotH - 60), Point(plotW - 60, plotH - 60), Scalar(0, 0, 0), 2);
    line(plot, Point(60, 60), Point(60, plotH - 60), Scalar(0, 0, 0), 2);

    for (size_t i = 0; i < omegaObs.size(); i++) {
        circle(plot, toPlot(tOmega[i], omegaObs[i]), 2, Scalar(255, 0, 0), -1);
    }
    for (size_t i = 0; i + 1 < omegaObs.size(); i++) {
        line(plot, toPlot(tOmega[i], omegaFit[i]), toPlot(tOmega[i + 1], omegaFit[i + 1]), Scalar(0, 0, 255), 2);
    }
    vutils::drawLabel(plot, "omega(t) = b + A*sin(Omega*t + phi)", Point(80, 40), Scalar(0, 0, 0), 0.7, 2);
    vutils::drawLabel(plot, "Blue: observation   Red: fit", Point(80, plotH - 20), Scalar(0, 0, 0), 0.5, 1);
    imwrite("result/task2_fit/fit_comparison.png", plot);
    imwrite("result/task2_fit/angular_velocity.png", plot);

    // ===== 残差图 =====
    Mat residualPlot(plotH, plotW, CV_8UC3, Scalar(255, 255, 255));
    double maxRes = 0;
    for (size_t i = 0; i < omegaObs.size(); i++) {
        maxRes = std::max(maxRes, std::abs(omegaObs[i] - omegaFit[i]));
    }
    maxRes = std::max(maxRes, 1e-6);

    line(residualPlot, Point(60, plotH / 2), Point(plotW - 60, plotH / 2), Scalar(0, 0, 0), 1);
    line(residualPlot, Point(60, 60), Point(60, plotH - 60), Scalar(0, 0, 0), 2);

    for (size_t i = 0; i + 1 < omegaObs.size(); i++) {
        double r1 = omegaObs[i] - omegaFit[i];
        double r2 = omegaObs[i + 1] - omegaFit[i + 1];
        auto toRes = [&](double t, double r) {
            int px = static_cast<int>((t - minT) / (maxT - minT + 1e-9) * (plotW - 120) + 60);
            int py = static_cast<int>(plotH / 2 - r / maxRes * (plotH / 2 - 60));
            return Point(px, py);
        };
        line(residualPlot, toRes(tOmega[i], r1), toRes(tOmega[i + 1], r2), Scalar(0, 0, 255), 1);
    }
    vutils::drawLabel(residualPlot, "Residuals (rad/s)", Point(80, 40), Scalar(0, 0, 0), 0.7, 2);
    imwrite("result/task2_fit/residuals.png", residualPlot);

    // ===== 写结果说明 =====
    std::ofstream md("result/task2_fit_result.md");
    md << "# 任务2：合成旋转视频的参数拟合\n\n";
    md << "## 模型\n\n";
    md << "角速度模型：`omega(t) = b + A*sin(Omega*t + phi)`\n\n";
    md << "## 估计参数\n\n";
    md << "| 参数 | 值 | 单位 |\n";
    md << "|------|----|----|\n";
    md << "| A | " << params[0] << " | rad/s |\n";
    md << "| b | " << params[1] << " | rad/s |\n";
    md << "| Omega | " << params[2] << " | rad/s |\n";
    md << "| phi | " << params[3] << " | rad |\n\n";
    md << "速度变化周期 T = 2*pi/Omega = " << (2 * M_PI / params[2]) << " s\n\n";
    md << "## 误差指标\n\n";
    md << "- RMSE = " << rmse << " rad/s\n";
    md << "- 有效样本数 = " << omegaObs.size() << "\n";
    md << "- 参与计算帧范围：1 ~ " << (totalFrames - 2) << "\n\n";
    md << "## 方法\n\n";
    md << "1. 在 HSV 空间用青色阈值提取目标，取最大轮廓中心。\n";
    md << "2. 计算 `theta = atan2(c_y - y, x - c_x)`，并展开为连续角。\n";
    md << "3. 中心差分求角速度。\n";
    md << "4. 用 Ceres Solver 自动求导拟合，约束 A>0、Omega>0。\n";
    md.close();

    std::cout << "Task 2 done." << std::endl;
    return 0;
}