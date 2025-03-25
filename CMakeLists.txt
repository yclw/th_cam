cmake_minimum_required(VERSION 3.5)
project(quick_discovery)

# 优化
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -O3 -march=native -fopenmp")

# opencv
set(OpenCV_DIR /home/yc-mini/opencv/opencv-4.10.0/install/lib/cmake/opencv4) # 改成自己的opencv路径（如果安装到系统就不用加）
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)
include_directories(hikSDK/include)
include_directories(mvSDK/include)
include_directories(dahSDK/include)
include_directories(cameras)

add_executable(quick_discovery
    test/main.cpp
    src/armor_detector.cpp
    src/number_classifier.cpp
)

target_link_directories(quick_discovery PUBLIC hikSDK/lib/amd64 mvSDK/lib/amd64 dahSDK/lib/x86_64)

target_link_libraries(quick_discovery
  ${OpenCV_LIBS}
  TBB::tbb
  FormatConversion
  MediaProcess
  MvCameraControl
  MVRender
  MvUsb3vTL
  MVSDK
  gxiapi
)