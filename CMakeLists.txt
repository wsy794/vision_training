cmake_minimum_required(VERSION 3.16)
project(vision_training LANGUAGES CXX)

# 设定 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 👇 这就是终极杀手锏！
# 强制在编译任何文件之前，先包含 limits 和 new。
# 这样 OpenCV 内部的 Bug 就彻底失效了！
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include limits -include new -include cstddef")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

add_library(common STATIC
    src/common/utils.cpp
)
target_include_directories(common PUBLIC include)
target_link_libraries(common PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Ceres::ceres
)

# 任务1
add_executable(task1_image src/task1_image/main.cpp)
target_link_libraries(task1_image PRIVATE common)

add_executable(task2_fit src/task2_fit/main.cpp)
target_link_libraries(task2_fit PRIVATE common)
add_executable(task3_windmill src/task3_windmill/main.cpp)
target_link_libraries(task3_windmill PRIVATE common)