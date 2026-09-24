# 第二次培训作业

## 一、环境依赖

· 操作系统：Ubuntu 22.04
· 依赖库：OpenCV 4.5.4, Eigen 3.4.0, Ceres Solver 2.0.0
· 构建工具：CMake 3.22, C++ 标准 C++17

## 二、构建与运行命令

1. 编译工程

```bash
cmake -S . -B build
cmake --build build -j4
```

2. 运行命令

```bash
./build/task1_image
./build/task2_fit
./build/task3_windmill resources/task_3.mp4 result/task3_windmill/task_3
./build/task3_windmill resources/task_4.mp4 result/task3_windmill/task_4
```

## 三、输入输出路径

· 输入素材：resources/ 目录下（test_image.jpg、task_2.mp4、task_3.mp4、task_4.mp4）
· 输出结果：result/ 目录下

## 四、任务1 参数与分析

· 滤波参数：均值(5x5)，高斯(5x5, sigma=1.5)，中值(ksize=5)
· 红色提取阈值：HSV 双区间 H[0,10] 和 H[170,179]，S≥100，V≥100
· 形态学操作：5x5 矩形核，实现腐蚀、膨胀、开运算、闭运算
· 轮廓筛选：面积≥500，长宽比在 [0.2, 5.0] 之间
· 筛选后轮廓面积：（这里填你终端里打印的面积数据）

## 五、任务说明文件索引

· 任务2参数与误差指标：详见 result/task2_fit_result.md
· 任务3锁定、丢失与重选规则：详见 result/task3_tracking_result.md