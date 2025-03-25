#ifndef VIRTUAL_CAMERA_HPP
#define VIRTUAL_CAMERA_HPP

#include "camera_core.hpp"

class VirtualCamera:public Camera
{
public:
    VirtualCamera(const std::string &name, std::shared_ptr<std::pair<cv::Mat, bool>> sharedFrame, std::mutex &mtx, std::condition_variable &cv, cv::Mat frame, int sleep = 5)
        : name(name), sharedFrame(sharedFrame), mtx(mtx), cv(cv), frame(frame.clone()), sleep(sleep), lostFrames(0), stop(false) {}

    ~VirtualCamera() override {}

    void operator()() override
    {
        while (!stop.load())
        {
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (sharedFrame->second)
                    lostFrames++;
                sharedFrame->first = generateNewFrame();
                sharedFrame->second = true;
            }
            cv.notify_all();

            // 模拟帧生成速度
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep)); // 默认200帧/秒
        }
    }

    int getLostFrames() const override
    {
        return lostFrames.load();
    }

    // 设置帧生成速度
    void setSleep(int sleep)
    {
        this->sleep = sleep;
    }

    // 停止采集
    void stopGrabbing() override {
        stop.store(true);
    }

private:
    cv::Mat generateNewFrame()
    {
        return frame.clone();
    }

    cv::Mat frame;

    std::string name;
    std::shared_ptr<std::pair<cv::Mat, bool>> sharedFrame;
    std::mutex &mtx;
    std::condition_variable &cv;
    int sleep;

    std::atomic<int> lostFrames; // 丢弃帧计数器
    std::atomic<bool> stop;
};

#endif // VIRTUAL_CAMERA_HPP