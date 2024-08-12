// Copyright 2024 Ar-Ray-code
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef YOLOX_CPP__UTILS_HPP_
#define YOLOX_CPP__UTILS_HPP_


#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "core.hpp"
#include "coco_names.hpp"

namespace yolox_cpp
{
namespace utils
{

static std::vector<std::string> read_class_labels_file(const file_name_t & file_name)
{
  std::vector<std::string> class_names;
  std::ifstream ifs(file_name);
  std::string buff;
  if (ifs.fail()) {
    return class_names;
  }
  while (getline(ifs, buff)) {
    if (buff == "") {
      continue;
    }
    class_names.push_back(buff);
  }
  return class_names;
}

static void draw_objects(
  cv::Mat bgr,
  const std::vector<Object> & objects,
  const std::vector<std::string> & class_names = COCO_CLASSES)
{
  for (const Object & obj : objects) {
    const int color_index = obj.label % 80;
    cv::Scalar color = cv::Scalar(
      color_list[color_index][0], color_list[color_index][1], color_list[color_index][2]);
    float c_mean = cv::mean(color)[0];
    cv::Scalar txt_color;
    if (c_mean > 0.5) {
      txt_color = cv::Scalar(0, 0, 0);
    } else {
      txt_color = cv::Scalar(255, 255, 255);
    }

    cv::rectangle(bgr, obj.rect, color * 255, 2);

    char text[256];
    sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);      // NOLINT

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

    cv::Scalar txt_bk_color = color * 0.7 * 255;

    int x = obj.rect.x;
    int y = obj.rect.y + 1;
    if (y > bgr.rows) {
      y = bgr.rows;
    }

    cv::rectangle(
      bgr,
      cv::Rect(
        cv::Point(x, y),
        cv::Size(label_size.width, label_size.height + baseLine)),
      txt_bk_color, -1);

    cv::putText(
      bgr, text, cv::Point(x, y + label_size.height),
      cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
  }
}
}  // namespace utils
}  // namespace yolox_cpp

#endif  // YOLOX_CPP__UTILS_HPP_
