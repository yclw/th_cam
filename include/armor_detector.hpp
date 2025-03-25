#ifndef ARMOR_DETECTOR_HPP
#define ARMOR_DETECTOR_HPP

// std
#include <cmath>
#include <string>
#include <vector>
// #include <iostream>
#include <numeric>
#include <algorithm>
#include <execution>

// cv
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// project
#include "number_classifier.hpp"

// Armor type
enum class ArmorType
{
    SMALL,
    LARGE,
    INVALID
};

// Color type
enum class EnemyColor
{
    RED = 0,
    BLUE = 1,
    WHITE = 2
};

///////////////
//           //
// parameter //
//           //
///////////////

// Light parameter
class LightParams
{
public:
    LightParams(double min_ratio = 0.08, double max_ratio = 0.4, double max_angle = 40.0, int color_diff_thresh = 50) : min_ratio(min_ratio), max_ratio(max_ratio), max_angle(max_angle), color_diff_thresh(color_diff_thresh) {}

    // width / height
    double min_ratio;
    double max_ratio;

    // vertical angle
    double max_angle;

    // judge color
    int color_diff_thresh;
};

// Armor parameter
class ArmorParams
{
public:
    ArmorParams(double min_light_ratio = 0.6, double min_small_center_distance = 0.8, double max_small_center_distance = 3.2,
                double min_large_center_distance = 3.2, double max_large_center_distance = 5.0, double max_angle = 15.0,
                double max_angle_difference = 15.0) : min_light_ratio(min_light_ratio), min_small_center_distance(min_small_center_distance), max_small_center_distance(max_small_center_distance),
                                                      min_large_center_distance(min_large_center_distance), max_large_center_distance(max_large_center_distance), max_angle(max_angle),
                                                      max_angle_difference(max_angle_difference) {}
    // width / height
    double min_light_ratio;
    // light pairs distance
    double min_small_center_distance;
    double max_small_center_distance;
    double min_large_center_distance;
    double max_large_center_distance;
    // horizontal angle
    double max_angle;
    // max angle difference between two light
    double max_angle_difference;
};

///////////////////
//               //
// light & armor //
//               //
///////////////////

struct Light : public cv::RotatedRect
{
    Light() = default;
    explicit Light(const std::vector<cv::Point> &contour)
        : RotatedRect(cv::minAreaRect(contour))
    {

        center = std::accumulate(
            contour.begin(),
            contour.end(),
            cv::Point2f(0, 0),
            [n = static_cast<float>(contour.size())](const cv::Point2f &a, const cv::Point &b)
            {
                return a + cv::Point2f(b.x, b.y) / n;
            });

        cv::Point2f p[4];

        this->points(p); // 001  (取外接旋转矩形的4个角)

        std::sort(p, p + 4, [](const cv::Point2f &a, const cv::Point2f &b)
                  { return a.y < b.y; });
        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;

        length = cv::norm(top - bottom);
        width = cv::norm(p[0] - p[1]);

        axis = top - bottom;
        axis = axis / cv::norm(axis); // 002 (倾斜方向的单位向量)

        // Calculate the tilt angle
        tilt_angle = std::atan2(-(top.y - bottom.y), (top.x - bottom.x));
        tilt_angle = tilt_angle / CV_PI * 180; // 003 (倾斜方向(单位度))
    }
    EnemyColor color = EnemyColor::WHITE;
    cv::Point2f top, bottom, center;
    cv::Point2f axis;
    double length;
    double width;
    float tilt_angle;
};

// Struct used to store the armor
struct Armor
{
    static constexpr const int N_LANDMARKS = 6;
    static constexpr const int N_LANDMARKS_2 = N_LANDMARKS * 2;
    Armor() = default;
    Armor(const Light &l1, const Light &l2)
    {
        if (l1.center.x < l2.center.x)
        {
            left_light = l1, right_light = l2;
        }
        else
        {
            left_light = l2, right_light = l1;
        }

        center = (left_light.center + right_light.center) / 2;
    }

    // Build the points in the object coordinate system, start from bottom left in
    // clockwise order
    template <typename PointType>
    static inline std::vector<PointType> buildObjectPoints(const double &w,
                                                           const double &h) noexcept
    {
        if constexpr (N_LANDMARKS == 4)
        {
            return {PointType(0, w / 2, -h / 2),
                    PointType(0, w / 2, h / 2),
                    PointType(0, -w / 2, h / 2),
                    PointType(0, -w / 2, -h / 2)};
        }
        else
        {
            return {PointType(0, w / 2, -h / 2),
                    PointType(0, w / 2, 0),
                    PointType(0, w / 2, h / 2),
                    PointType(0, -w / 2, h / 2),
                    PointType(0, -w / 2, 0),
                    PointType(0, -w / 2, -h / 2)};
        }
    }

    // Landmarks start from bottom left in clockwise order
    std::vector<cv::Point2f> landmarks() const
    {
        if constexpr (N_LANDMARKS == 4)
        {
            return {left_light.bottom, left_light.top, right_light.top, right_light.bottom};
        }
        else
        {
            return {left_light.bottom,
                    left_light.center,
                    left_light.top,
                    right_light.top,
                    right_light.center,
                    right_light.bottom};
        }
    }
    // Light pairs part
    Light left_light, right_light;
    cv::Point2f center;
    ArmorType type;

    // Number part
    cv::Mat number_img;
    int number;
    float confidence;
    std::string classfication_result;
    cv::Point2f lights_vertices[4];
};

//////////////
//          //
//  result  //
//          //
//////////////
struct Result
{
    int x;
    int y;
    int number;
};

//////////////
//          //
// function //
//          //
//////////////

class Detector
{
public:
    Detector(LightParams light_params, ArmorParams armor_params, int binary_thres, EnemyColor ourColor,
             std::string model_path = "./model/lenet.onnx",
             std::string label_path = "./model/label.txt") : light_params(light_params), armor_params(armor_params), binary_thres(binary_thres), ourColor(ourColor), is_drawArmor(is_drawArmor)
    {
        this->model = new NumberClassifier(model_path, label_path);
    }

    void detect(const cv::Mat &input, std::vector<Result> &results);

private:
    NumberClassifier *model;
    int binary_thres;

    LightParams light_params;
    ArmorParams armor_params;

    bool is_drawArmor;

    EnemyColor ourColor;

    void preprocessImage(const cv::Mat &rgb_img, cv::Mat &binary_img);
    bool isLight(const Light &light);
    void findLights(cv::InputArray rgb_img, cv::InputArray binary_img, std::vector<Light> &lights);
    bool containLight(const int i, const int j, const std::vector<Light> &lights);
    ArmorType isArmor(const Light &light_1, const Light &light_2);
    void matchLights(const std::vector<Light> &lights, cv::InputArray img, std::vector<Armor> &armors, std::vector<Result> &results);
    cv::Mat extractNumber(cv::InputArray src, Armor &armor);
};

#endif // ARMOR_DETECTOR_HPP