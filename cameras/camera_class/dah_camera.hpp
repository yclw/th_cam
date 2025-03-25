#ifndef DAH_CAMERA_HPP
#define DAH_CAMERA_HPP


#include "camera_core.hpp"
#include "GxIAPI.h"

class DahCamera:public Camera
{
public:
    DahCamera(std::shared_ptr<std::pair<cv::Mat, bool>> sharedFrame, std::mutex &mtx, std::condition_variable &cv, GX_DEVICE_BASE_INFO &device)
        : sharedFrame(sharedFrame), mtx(mtx), cv(cv), lostFrames(0), stop(false)
    {
        // 打开相机
        stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
        stOpenParam.openMode = GX_OPEN_SN;
        stOpenParam.pszContent = device.szSN;
        GXOpenDevice(&stOpenParam, &camera_handle_);

        // 设置采集队列的缓冲区数量
        GXSetAcqusitionBufferNumber(camera_handle_, nBufferNum);

        // 设置触发模式为关闭
        GXSetEnum(camera_handle_,GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);

        // 设置相机参数
        configureCamera();
    }

    DahCamera(std::shared_ptr<std::pair<cv::Mat, bool>> sharedFrame, std::mutex &mtx, std::condition_variable &cv)
        : sharedFrame(sharedFrame), mtx(mtx), cv(cv), lostFrames(0)
    {
        // 获取设备列表并选择第一个设备
        auto devices = getDahCameraList();
        if (devices.size()==0)
        {
            throw std::runtime_error("No camera found.");
        }
        auto device = devices[0];

        // 打开相机
        stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
        stOpenParam.openMode = GX_OPEN_SN;
        stOpenParam.pszContent = device.szSN;
        GXOpenDevice(&stOpenParam, &camera_handle_);

        // 设置采集队列的缓冲区数量
        GXSetAcqusitionBufferNumber(camera_handle_, nBufferNum);

        // 设置触发模式为关闭
        GXSetEnum(camera_handle_,GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);

        // 设置相机参数
        configureCamera();
    }

    ~DahCamera() override
    {
        // 停止捕获并关闭设备
        stop = false;
        GXStreamOff(camera_handle_);
        GXCloseDevice(camera_handle_);
        GXCloseLib();
        std::cout << "HikCamera destroyed." << std::endl;
    }

    // 获取相机列表
    static std::vector<GX_DEVICE_BASE_INFO> getDahCameraList()
    {
        uint32_t nDeviceNum = 0;
        GXInitLib();
        GXUpdateDeviceList(&nDeviceNum, 1000);
        std::cout << "Device number: " << nDeviceNum << std::endl;

        std::vector<GX_DEVICE_BASE_INFO> device_list(nDeviceNum);

        if (nDeviceNum == 0) return device_list;

        size_t nSize = nDeviceNum * sizeof(GX_DEVICE_BASE_INFO);
        // 获取所有设备的基础信息
        GXGetAllDeviceBaseInfo(device_list.data(), &nSize);

        return device_list;
    }


    // 启动相机
    void operator()() override
    {
        int nRet = GX_STATUS_SUCCESS;
        GXStreamOn(camera_handle_);
        while (!stop.load())
        {
            nRet = GXDQBuf(camera_handle_, &pFrameBuffer, 1000);
            if (nRet == GX_STATUS_SUCCESS)
            {
                // 处理帧数据
                processFrame(pFrameBuffer);
                nRet = GXQBuf(camera_handle_, pFrameBuffer);
                fail_count_ = 0;
            }
            else
            {
                // 处理相机捕获失败
                handleGrabFailure();
                fail_count_++;
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
        // !!!!!!!
        // 设置曝光时间
        // GXSetFloat(camera_handle_, GX_FLOAT_EXPOSURE_TIME,exposure_time);
    }

    void processFrame(PGX_FRAME_BUFFER &frame)
    {
        if (frame->nStatus == GX_FRAME_STATUS_SUCCESS)
        {
            cv::Mat img = cv::Mat(frame->nHeight, frame->nWidth, CV_8UC3, frame->pImgBuf);
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (sharedFrame->second)
                    lostFrames++;
                sharedFrame->first = img.clone();
                sharedFrame->second = true;
            }
            cv.notify_all();
        }
    }

    void handleGrabFailure() {
        std::cout << "Get buffer failed, retrying..." << std::endl;
        GXStreamOff(camera_handle_);
        GXStreamOn(camera_handle_);
        if (fail_count_ > 5) {
            std::cout << "Camera failed after several attempts." << std::endl;
        }
    }

    // 相机句柄，用于相机的控制和通信
    GX_DEV_HANDLE camera_handle_ = nullptr;
    // 相机启动参数
    GX_OPEN_PARAM stOpenParam;
    // 设置采集队列的缓冲区数量
    uint64_t nBufferNum = 1;
    //定义 GXDQBuf 的传入参数
    PGX_FRAME_BUFFER pFrameBuffer;
    // 默认曝光时间为5000微秒
    float exposure_time = 5000;
    // 存储图像数据的缓冲区
    std::vector<uint8_t> image_data_;
    // 标记图像采集是否完成
    bool over = false;
    // 记录相机连接失败的次数
    int fail_count_ = 0;

    std::shared_ptr<std::pair<cv::Mat, bool>> sharedFrame;
    std::mutex &mtx;
    std::condition_variable &cv;
    std::atomic<bool> stop;

    std::atomic<int> lostFrames; // 丢弃帧计数器
};

#endif // DAH_CAMERA_HPP