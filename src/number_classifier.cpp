// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
// std
#include <algorithm>
#include <cstddef>
#include <execution>
#include <fstream>
#include <future>
#include <map>
#include <string>
#include <vector>

// project
#include "number_classifier.hpp"

NumberClassifier::NumberClassifier(const std::string &model_path,
                                   const std::string &label_path)
{
  net_ = cv::dnn::readNetFromONNX(model_path);
  std::ifstream label_file(label_path);
  std::string line;
  while (std::getline(label_file, line))
  {
    class_names_.push_back(line);
  }
}

std::vector<int> NumberClassifier::classify(cv::Mat &img_num)
{
  // Normalize
  cv::Mat input = img_num / 255.0;

  // Create blob from image
  cv::Mat blob;
  cv::dnn::blobFromImage(input, blob);

  // Set the input blob for the neural network
  mutex_.lock();
  net_.setInput(blob);

  // Forward pass the image blob through the model
  cv::Mat outputs = net_.forward().clone();
  mutex_.unlock();

  // Decode the output
  double confidence;
  cv::Point class_id_point;
  minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
  int label_id = class_id_point.x;

  std::vector<int> result;
  result.push_back(label_id);
  result.push_back((int)(confidence * 100));
  return result;
}
