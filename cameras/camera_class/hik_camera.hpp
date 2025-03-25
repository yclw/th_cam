#ifndef HIK_CAMERA_HPP
#define HIK_CAMERA_HPP

#include "camera_core.hpp"

#include "MvCameraControl.h"


class HikCamera:public Camera
{
public:
    // 构造函数，接受设备信息作为参数
    HikCamera(std::shared_ptr<std::pair<cv::Mat, bool>> sharedFrame, std::mutex &mtx, std::condition_variable &cv, const MV_CC_DEVICE_INFO &device)
        : sharedFrame(sharedFrame), mtx(mtx), cv(cv), lostFrames(0), stop(false)
    {
        // 创建相机句柄并打开设备
        MV_CC_CreateHandle(&camera_handle_, &device);
        MV_CC_OpenDevice(camera_handle_);
        MV_CC_GetImageInfo(camera_handle_, &img_info_);

        // 预分配图像缓冲区
        image_data_.resize(img_info_.nWidthValue * img_info_.nHeightValue * 3);
        convert_param_.nWidth = img_info_.nWidthValue;
        convert_param_.nHeight = img_info_.nHeightValue;
        convert_param_.enDstPixelType = PixelType_Gvsp_RGB8_Packed;
        
        // 设置相机参数
        configureCamera();
    }

    // 构造函数，无设备信息时自动获取设备列表并选择第一个设备
    HikCamera(std::shared_ptr<std::pair<cv::Mat, bool>> sharedFrame, std::mutex &mtx, std::condition_variable &cv)
        : sharedFrame(sharedFrame), mtx(mtx), cv(cv), lostFrames(0), stop(false)
    {
        // 获取设备列表并选择第一个设备
        auto devices = getHikCameraList();
        if (devices.empty()) {
            throw std::runtime_error("No camera found.");
        }
        auto device = devices[0];

        // 创建相机句柄并打开设备
        MV_CC_CreateHandle(&camera_handle_, &devices[0]);
        MV_CC_OpenDevice(camera_handle_);
        MV_CC_GetImageInfo(camera_handle_, &img_info_);

        // 预分配图像缓冲区
        image_data_.resize(img_info_.nWidthValue * img_info_.nHeightValue * 3);
        convert_param_.nWidth = img_info_.nWidthValue;
        convert_param_.nHeight = img_info_.nHeightValue;
        convert_param_.enDstPixelType = PixelType_Gvsp_RGB8_Packed;

        // 设置相机参数
        configureCamera();
    }

    ~HikCamera() override
    {
        // 停止捕获并关闭设备
        stop = true;
        MV_CC_StopGrabbing(camera_handle_);
        MV_CC_CloseDevice(camera_handle_);
        MV_CC_DestroyHandle(camera_handle_);
    }

    // 获取设备列表
    static std::vector<MV_CC_DEVICE_INFO> getHikCameraList()
    {
        MV_CC_DEVICE_INFO_LIST device_list;
        int nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
        std::cout << "Open Device Num: " << device_list.nDeviceNum << std::endl;
        std::vector<MV_CC_DEVICE_INFO> devices;
        for (int i = 0; i < device_list.nDeviceNum; ++i) {
            devices.push_back(*device_list.pDeviceInfo[i]);
        }
        return devices;
    }

    // 采集线程
    void operator()() override
    {
        int nRet = MV_OK;
        MV_CC_StartGrabbing(camera_handle_);
        while (!stop.load()) {
            nRet = MV_CC_GetImageBuffer(camera_handle_, &out_frame, 1000);
            if (nRet == MV_OK) {
                fail_count_ = 0;
                processFrame(out_frame);
                MV_CC_FreeImageBuffer(camera_handle_, &out_frame);
            } else {
                fail_count_++;
                handleGrabFailure();
            }
        }
    }

    // 停止采集
    void stopGrabbing() override {
        stop.store(true);
    }

    // 获取丢帧数
    int getLostFrames() const override
    {
        return lostFrames.load();
    }

private:
    void configureCamera() {
        MV_CC_SetFloatValue(camera_handle_, "ExposureTime", exposure_time);
        MV_CC_SetFloatValue(camera_handle_, "Gain", gain);
    }

    void processFrame(MV_FRAME_OUT& frame) {

        // 格式转换
        image_data_.resize(frame.stFrameInfo.nWidth * frame.stFrameInfo.nHeight * 3);
        convert_param_.pDstBuffer = image_data_.data();
        convert_param_.nDstBufferSize = image_data_.size();
        convert_param_.pSrcData = frame.pBufAddr;
        convert_param_.nSrcDataLen = frame.stFrameInfo.nFrameLen;
        convert_param_.enSrcPixelType = frame.stFrameInfo.enPixelType;
        MV_CC_ConvertPixelType(camera_handle_, &convert_param_);
        
        cv::Mat img(frame.stFrameInfo.nHeight, frame.stFrameInfo.nWidth, CV_8UC3, image_data_.data());

        // 更新共享帧
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (sharedFrame->second) {
                lostFrames++;
            }
            sharedFrame->first = img.clone();
            sharedFrame->second = true;
        }
        cv.notify_all();
    }

    void handleGrabFailure() {
        std::cout << "Get buffer failed, retrying..." << std::endl;
        MV_CC_StopGrabbing(camera_handle_);
        MV_CC_StartGrabbing(camera_handle_);
        if (fail_count_ > 5) {
            std::cout << "Camera failed after several attempts." << std::endl;
        }
    }

    void* camera_handle_;
    MV_IMAGE_BASIC_INFO img_info_;
    double exposure_time = 5000;
    double gain = 16;
    MV_CC_PIXEL_CONVERT_PARAM convert_param_;
    std::vector<uint8_t> image_data_;
    MV_FRAME_OUT out_frame = {0};
    cv::Mat frame;
    std::shared_ptr<std::pair<cv::Mat, bool>> sharedFrame;
    std::mutex& mtx;
    std::condition_variable& cv;
    std::atomic<bool> stop;
    std::atomic<int> lostFrames;
    int fail_count_ = 0;
};

#endif // HIK_CAMERA_HPP