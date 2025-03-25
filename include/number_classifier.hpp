#ifndef ARMOR_DETECTOR_NUMBER_CLASSIFIER_HPP
#define ARMOR_DETECTOR_NUMBER_CLASSIFIER_HPP

// std
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <vector>
// third party
#include <opencv2/opencv.hpp>

// Class used to classify the number of the armor, based on the MLP model
class NumberClassifier
{
public:
  NumberClassifier(const std::string &model_path,
                   const std::string &label_path);

  // Classify the number of the armor
  std::vector<int> classify(cv::Mat &img_num);

  double threshold;

private:
  std::mutex mutex_;
  cv::dnn::Net net_;
  std::vector<std::string> class_names_;
  std::vector<std::string> ignore_classes_;
};

#endif // ARMOR_DETECTOR_NUMBER_CLASSIFIER_HPP
