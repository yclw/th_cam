#include "armor_detector.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
// #include <opencv2/ximgproc/fast_line_detector.hpp>

void Detector::detect(const cv::Mat &input, std::vector<Result> &results)
{
  cv::Mat _binary_img(input.rows, input.cols, CV_8UC1);
  std::vector<Light> _lights;
  std::vector<Armor> _armors;
  preprocessImage(input, _binary_img);
  findLights(input, _binary_img, _lights);
  matchLights(_lights, input, _armors, results);
}

void Detector::preprocessImage(const cv::Mat &rgb_img, cv::Mat &binary_img)
{

  // uchar* p = rgb_img.data;
  // uchar* pRed = binary_img.data;

  // cv::parallel_for_(cv::Range(0, rgb_img.rows), [&](const cv::Range& range) {
  //     for (int i = range.start; i < range.end; i++) {
  //         uchar* rowPtr = p + i * rgb_img.cols * 3;
  //         uchar* rowRedPtr = pRed + i * rgb_img.cols;

  //         for (int j = 0; j < rgb_img.cols; j++) {
  //             rowRedPtr[j] = rowPtr[j * 3 + (ourColor==EnemyColor::BLUE ? 2:0)] >= binary_thres ? 255 : 0;
  //         }
  //     }
  // });
  cv::UMat rgb_img_umat = rgb_img.getUMat(cv::ACCESS_READ);
  cv::UMat gray_img_umat;
  cv::extractChannel(rgb_img_umat, gray_img_umat, ourColor == EnemyColor::BLUE ? 2 : 0);
  cv::threshold(gray_img_umat, binary_img, binary_thres, 255, cv::THRESH_BINARY);
}

bool Detector::isLight(const Light &light)
{
  // The ratio of light (short side / long side)
  float ratio = light.width / light.length;
  bool ratio_ok = light_params.min_ratio < ratio && ratio < light_params.max_ratio;

  bool angle_ok = (light.tilt_angle < (light_params.max_angle + 90.0)) && (light.tilt_angle > (90.0 - light_params.max_angle));

  bool is_light = ratio_ok && angle_ok;

  return is_light;
}

void Detector::findLights(cv::InputArray rgb_img, cv::InputArray binary_img, std::vector<Light> &lights)
{

  const cv::Mat rgb = rgb_img.getMat();
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  // return; // test

  for (const auto &contour : contours)
  {
    if (contour.size() < 6)
      continue;

    Light light = Light(contour);
    if (isLight(light))
    {
      int sum_r = 0, sum_b = 0;
      for (const auto &point : contour)
      {
        sum_r += rgb.at<cv::Vec3b>(point.y, point.x)[2];
        sum_b += rgb.at<cv::Vec3b>(point.y, point.x)[0];
      }
      if (std::abs(sum_r - sum_b) / static_cast<int>(contour.size()) >
          light_params.color_diff_thresh)
      {
        light.color = sum_r > sum_b ? EnemyColor::RED : EnemyColor::BLUE;
      }
      if (light.color != ourColor)
      {
        lights.emplace_back(light);
      }
    }
  }
  std::sort(lights.begin(), lights.end(), [](const Light &l1, const Light &l2)
            { return l1.center.x < l2.center.x; });
}

bool Detector::containLight(const int i, const int j, const std::vector<Light> &lights)
{
  const Light &light_1 = lights.at(i), light_2 = lights.at(j);
  auto points = std::vector<cv::Point2f>{light_1.top, light_1.bottom, light_2.top, light_2.bottom};
  auto bounding_rect = cv::boundingRect(points);
  double avg_length = (light_1.length + light_2.length) / 2.0;
  double avg_width = (light_1.width + light_2.width) / 2.0;
  // Only check lights in between
  for (int k = i + 1; k < j; k++)
  {
    const Light &test_light = lights.at(k);

    // 防止数字干扰
    if (test_light.width > 2 * avg_width)
    {
      continue;
    }
    // 防止红点准星或弹丸干扰
    if (test_light.length < 0.5 * avg_length)
    {
      continue;
    }

    if (bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom) ||
        bounding_rect.contains(test_light.center))
    {
      return true;
    }
  }
  return false;
}

ArmorType Detector::isArmor(const Light &light_1, const Light &light_2)
{
  // Ratio of the length of 2 lights (short side / long side)
  float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                             : light_2.length / light_1.length;
  bool light_ratio_ok = light_length_ratio > armor_params.min_light_ratio;

  // Distance between the center of 2 lights (unit : light length)
  float avg_light_length = (light_1.length + light_2.length) / 2;
  float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
  bool center_distance_ok = (armor_params.min_small_center_distance <= center_distance &&
                             center_distance < armor_params.max_small_center_distance) ||
                            (armor_params.min_large_center_distance <= center_distance &&
                             center_distance < armor_params.max_large_center_distance);

  // Angle of light center connection
  cv::Point2f diff = light_1.center - light_2.center;
  float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
  float angle_difference = std::abs(light_1.tilt_angle - light_2.tilt_angle);
  bool angle_ok = angle < armor_params.max_angle && angle_difference < armor_params.max_angle_difference;

  bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

  // Judge armor type
  ArmorType type;
  if (is_armor)
  {
    type = center_distance > armor_params.min_large_center_distance ? ArmorType::LARGE
                                                                    : ArmorType::SMALL;
  }
  else
  {
    type = ArmorType::INVALID;
  }

  return type;
}

void Detector::matchLights(const std::vector<Light> &lights, cv::InputArray img, std::vector<Armor> &armors, std::vector<Result> &results)
{
  for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++)
  {
    double max_iter_width = light_1->length * armor_params.max_large_center_distance;
    double min_iter_width = light_1->width;
    for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++)
    {
      double distance_1_2 = light_2->center.x - light_1->center.x;
      if (distance_1_2 < min_iter_width)
        continue;
      if (distance_1_2 > max_iter_width)
        break;
      if (containLight(light_1 - lights.begin(), light_2 - lights.begin(), lights))
      {
        continue;
      }
      ArmorType type = isArmor(*light_1, *light_2);
      if (type != ArmorType::INVALID)
      {
        Armor armor = Armor(*light_1, *light_2);
        armor.type = type;
        armor.number_img = extractNumber(img, armor);
        auto classifyRet = model->classify(armor.number_img);
        armor.number = classifyRet[0];
        if (armor.number != 8)
        {
          armors.emplace_back(armor);
          results.emplace_back(Result{(int)armor.center.x, (int)armor.center.y, armor.number});
        }
      }
    }
  }
}

cv::Mat Detector::extractNumber(cv::InputArray src, Armor &armor)
{
  // Light length in image
  static const int light_length = 12;
  // Image size after warp
  static const int warp_height = 28;
  static const int small_armor_width = 32;
  static const int large_armor_width = 54;
  // Number ROI size
  static const cv::Size roi_size(28, 28);
  static const cv::Size input_size(28, 28);

  armor.lights_vertices[0] = armor.left_light.bottom;
  armor.lights_vertices[1] = armor.left_light.top;
  armor.lights_vertices[2] = armor.right_light.top;
  armor.lights_vertices[3] = armor.right_light.bottom;

  const int top_light_y = (warp_height - light_length) / 2 - 1;
  const int bottom_light_y = top_light_y + light_length;
  const int warp_width = armor.type == ArmorType::SMALL ? small_armor_width : large_armor_width;
  cv::Point2f target_vertices[4] = {
      cv::Point(0, bottom_light_y),
      cv::Point(0, top_light_y),
      cv::Point(warp_width - 1, top_light_y),
      cv::Point(warp_width - 1, bottom_light_y),
  };
  cv::UMat number_image;
  auto rotation_matrix = cv::getPerspectiveTransform(armor.lights_vertices, target_vertices);
  cv::warpPerspective(src, number_image, rotation_matrix, cv::Size(warp_width, warp_height));

  // Get ROI
  number_image = number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

  // Binarize
  cv::cvtColor(number_image, number_image, cv::COLOR_RGB2GRAY);
  cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  // std::cout<<number_image.size()<<std::endl;
  // cv::resize(number_image, number_image, input_size);

  cv::Mat number_image_mat = number_image.getMat(cv::ACCESS_RW).clone();

  return number_image_mat;
}