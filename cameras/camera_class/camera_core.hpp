#ifndef CAMERA_CORE_HPP
#define CAMERA_CORE_HPP
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <memory>
#include <atomic>
#include <chrono>
#include "armor_detector.hpp"

class Camera
{
public:
    virtual ~Camera(){}

    // 停止抓取图像
    virtual void stopGrabbing() = 0;

    // 获取丢帧数
    virtual int getLostFrames() const = 0;

    // 开始抓取图像
    virtual void operator()() = 0;

    std::string name;
};
#endif // CAMERA_CORE_HPP