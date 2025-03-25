#ifndef MIND_CAMERA_HPP
#define MIND_CAMERA_HPP

#include "camera_core.hpp"
#include "CameraApi.h"


class MindCamera:public Camera
{
public:
    MindCamera(std::shared_ptr<std::pair<cv::Mat, bool>> sharedFrame, std::mutex &mtx, std::condition_variable &cv, tSdkCameraDevInfo &device)
        : sharedFrame(sharedFrame), mtx(mtx), cv(cv), lostFrames(0), stop(false)
    {
        // 初始化相机
        CameraInit(&device, -1, -1, &camera_handle_);
        // CameraGetCapability(camera_handle_, &t_capability_);

        // 设置自动曝光模式和参数
        CameraSetAeState(camera_handle_, false);
        configureCamera();
    }

    MindCamera(std::shared_ptr<std::pair<cv::Mat, bool>> sharedFrame, std::mutex &mtx, std::condition_variable &cv)
        : sharedFrame(sharedFrame), mtx(mtx), cv(cv), lostFrames(0), stop(false)
    {
        // 获取设备列表并选择第一个设备
        auto devices = getMindCameraList();
        if (devices.size()==0)
        {
            throw std::runtime_error("No camera found.");
        }
        auto device = devices[0];

        // 初始化相机
        CameraInit(&device, -1, -1, &camera_handle_);
        // CameraGetCapability(camera_handle_, &t_capability_);

        // 设置自动曝光模式和参数
        CameraSetAeState(camera_handle_, false);
        configureCamera();
    }

    ~MindCamera() override
    {
        CameraUnInit(camera_handle_);
        std::cout << "MindCamera destroyed." << std::endl;
    }

    static std::vector<tSdkCameraDevInfo> getMindCameraList()
    {
        CameraSdkInit(1);
        
        // 枚举设备，并建立设备列表
        int i_camera_counts = 4;
        int ret = -1;
        tSdkCameraDevInfo device_list[4];
        ret = CameraEnumerateDevice(device_list, &i_camera_counts);
        std::cout << "Found camera count = " << i_camera_counts << std::endl;

        std::vector<tSdkCameraDevInfo> devices;
        for (int i = 0; i < i_camera_counts; i++)
        {
            devices.push_back(device_list[i]);
        }
        return devices;
    }

    void operator()() override
    {
        int nRet = CAMERA_STATUS_SUCCESS;
        CameraPlay(camera_handle_);
        CameraSetIspOutFormat(camera_handle_, CAMERA_MEDIA_TYPE_RGB8);
        tSdkFrameHead out_frame_head;
        uint8_t * out_frame_buffer;
        while (!stop.load())
        {
            nRet = CameraGetImageBuffer(camera_handle_, &out_frame_head, &out_frame_buffer, 1000);
            if (nRet == CAMERA_STATUS_SUCCESS)
            {

                processFrame(out_frame_head, out_frame_buffer);
                CameraReleaseImageBuffer(camera_handle_, out_frame_buffer);
                fail_count_ = 0;
            }
            else
            {
                handleGrabFailure();
                fail_count_++;
            }
        }
    }

    // 停止采集
    void stopGrabbing() override {
        stop.store(true);
    }

    int getLostFrames() const override
    {
        return lostFrames.load();
    }

private:
    void configureCamera() {
        std::cout << "Configuring camera..." << std::endl;
        // 设置曝光时间
        CameraSetExposureTime(camera_handle_, exposure_time);
        // 设置增益
        CameraSetAnalogGain(camera_handle_, analog_gain);
        CameraSetGain(camera_handle_, r_gain_, g_gain_, b_gain_);
        // 设置饱和度
        CameraSetSaturation(camera_handle_, saturation);
        // 设置gamma
        CameraSetGamma(camera_handle_, gamma);
    }

    void processFrame(tSdkFrameHead &out_frame_head,uint8_t *out_frame_buffer){
        image_data_.resize(out_frame_head.iHeight * out_frame_head.iWidth * 3);
        CameraImageProcess(camera_handle_, out_frame_buffer, image_data_.data(), &out_frame_head);
        cv::Mat img = cv::Mat(out_frame_head.iHeight, out_frame_head.iWidth, CV_8UC3, image_data_.data());

        {
            std::lock_guard<std::mutex> lock(mtx);
            if (sharedFrame->second)
                lostFrames++;
            sharedFrame->first = img.clone();
            sharedFrame->second = true;
        }
        cv.notify_all();
    }

    void handleGrabFailure() {
        std::cout << "Get buffer failed, retrying..." << std::endl;
        if (fail_count_ > 5) {
            std::cout << "Camera failed after several attempts." << std::endl;
        }
    }
    
    // 相机句柄，用于相机的控制和通信
    CameraHandle camera_handle_;
    // 默认曝光时间
    double exposure_time = 5000;
    // 默认增益
    int analog_gain = 64;
    // 默认饱和度
    int saturation = 128;
    // 默认rgb三通道的增益
    int r_gain_ = 100;
    int g_gain_ = 100;
    int b_gain_ = 100;
    // 默认伽马值
    int gamma = 100;
    // 存储图像数据的缓冲区
    std::vector<uint8_t> image_data_;
    // 标记图像采集是否完成
    std::atomic<bool> stop;
    // 记录相机连接失败的次数
    int fail_count_ = 0;

    std::shared_ptr<std::pair<cv::Mat, bool>> sharedFrame;
    std::mutex &mtx;
    std::condition_variable &cv;

    std::atomic<int> lostFrames; // 丢弃帧计数器
};


#endif // MIND_CAMERA_HPP