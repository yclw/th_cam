#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <string.h>

#include "cameras.hpp"
#include "armor_detector_consumer.hpp"


int all_cameras_test()
{
    auto hikCameras = HikCamera::getHikCameraList();
    auto mindCameras = MindCamera::getMindCameraList();
    auto dahCameras = DahCamera::getDahCameraList();

    if (hikCameras.size() == 0 && mindCameras.size() == 0 && dahCameras.size() == 0)
    {
        std::cout << "No camera found" << std::endl;
        return -1;
    }

    std::unordered_map<std::string,std::shared_ptr<std::pair<cv::Mat, bool>>> sharedFrames;
    std::vector<std::shared_ptr<Camera>> cameras;
    std::vector<std::thread> producerThreads;
    std::mutex mtx;
    std::condition_variable cv;

    // 启动所有hik相机
    for (size_t i = 0; i < hikCameras.size(); i++)
    {
        auto sharedFrame = std::make_shared<std::pair<cv::Mat, bool>>(cv::Mat(), false);
        std::string name = "hik_" + std::to_string(i);
        sharedFrames[name] = sharedFrame;
        cameras.push_back(std::make_shared<HikCamera>(sharedFrame, mtx, cv, hikCameras[i]));
        producerThreads.emplace_back(std::ref(*cameras.back()));
    }

    // 启动所有mind相机
    for (int i = 0; i < mindCameras.size(); ++i) {
        auto sharedFrame = std::make_shared<std::pair<cv::Mat, bool>>(cv::Mat(), false);
        std::string name = "mind_" + std::to_string(i);
        sharedFrames[name] = sharedFrame;
        cameras.push_back(std::make_shared<MindCamera>(sharedFrame, mtx, cv, mindCameras[i]));
        producerThreads.emplace_back(std::ref(*cameras.back()));
    }

    // 启动所有dah相机
    for (int i = 0; i < dahCameras.size(); ++i) {
        auto sharedFrame = std::make_shared<std::pair<cv::Mat, bool>>(cv::Mat(), false);
        std::string name = "dah_" + std::to_string(i);
        sharedFrames[name] = sharedFrame;
        cameras.push_back(std::make_shared<DahCamera>(sharedFrame, mtx, cv, dahCameras[i]));
        producerThreads.emplace_back(std::ref(*cameras.back()));
    }

    // 启动两个图像处理线程
    LightParams light_params;
    ArmorParams armor_params;
    armor_params.max_angle = 100.0;
    std::vector<std::thread> detectorThreads;
    std::vector<std::shared_ptr<DetectorConsumer>> detectorConsumers;
    std::vector<Detector> detectors;
    for (int i = 0; i < 2; ++i) {
        detectors.emplace_back(light_params, armor_params, 120, EnemyColor::BLUE);
        detectorConsumers.push_back(std::make_shared<DetectorConsumer>(sharedFrames, mtx, cv, detectors.back()));
        detectorThreads.emplace_back(std::ref(*detectorConsumers.back()));
    }

    // 主线程统计帧率
    int previousFrameCount = 0;
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1)); // 每秒统计一次
        // 统计处理帧率
        int currentFrameCount = 0;
        for (auto &detectorConsumer : detectorConsumers)
            currentFrameCount += detectorConsumer->getFrameCount();
        int processedFrames = currentFrameCount - previousFrameCount;
        previousFrameCount = currentFrameCount;
        std::cout << "FPS: " << processedFrames << " frames/second" << std::endl;

        // 统计丢帧总数
        for (size_t i = 0; i < cameras.size(); i++)
            std::cout << "camera " << i << " lost frames: " << cameras[i]->getLostFrames() << std::endl;
        
    }
    return 0;
}

int th_virtual_camera_test()
{
    std::unordered_map<std::string,std::shared_ptr<std::pair<cv::Mat, bool>>> sharedFrames;
    std::vector<std::shared_ptr<Camera>> cameras;
    std::vector<std::thread> producerThreads;
    std::mutex mtx;
    std::condition_variable cv;

    cv::Mat img = cv::imread("./1.png");
    if (img.empty())
    {
        throw std::runtime_error("image is empty");
        return -1;
    }
    // cv::resize(img, img, cv::Size(640, 480));

    for (size_t i = 0; i < 3; i++)
    {
        auto sharedFrame = std::make_shared<std::pair<cv::Mat, bool>>(cv::Mat(), false);
        std::string name = "vir_" + std::to_string(i);
        sharedFrames[name] = sharedFrame;
        cameras.push_back(std::make_shared<VirtualCamera>(name, sharedFrame, mtx, cv, img, 4));
        producerThreads.emplace_back(std::ref(*cameras.back()));
    }

    // 启动两个图像处理线程
    LightParams light_params;
    ArmorParams armor_params;
    armor_params.max_angle = 100.0;
    std::vector<std::thread> detectorThreads;
    std::vector<std::shared_ptr<DetectorConsumer>> detectorConsumers;
    std::vector<Detector> detectors;
    for (int i = 0; i < 2; ++i) {
        detectors.emplace_back(light_params, armor_params, 120, EnemyColor::BLUE);
        detectorConsumers.push_back(std::make_shared<DetectorConsumer>(sharedFrames, mtx, cv, detectors.back()));
        detectorThreads.emplace_back(std::ref(*detectorConsumers.back()));
    }

    // 主线程统计帧率
    int previousFrameCount = 0;
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1)); // 每秒统计一次
        // 统计处理帧率
        int currentFrameCount = 0;
        for (auto &detectorConsumer : detectorConsumers){
            detectorConsumer->getResults();
            currentFrameCount += detectorConsumer->getFrameCount();
        }
            
        int processedFrames = currentFrameCount - previousFrameCount;
        previousFrameCount = currentFrameCount;
        std::cout << "FPS: " << processedFrames << " frames/second" << std::endl;

        // 统计丢帧总数
        for (size_t i = 0; i < cameras.size(); i++)
            std::cout << "camera " << i << " lost frames: " << cameras[i]->getLostFrames() << std::endl;
        
    }

}

int main()
{
    // th_virtual_camera_test();
    all_cameras_test();
    return 0;
}
