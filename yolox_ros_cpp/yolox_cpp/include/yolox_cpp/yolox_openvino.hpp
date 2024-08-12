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

#ifndef YOLOX_CPP__YOLOX_OPENVINO_HPP_
#define YOLOX_CPP__YOLOX_OPENVINO_HPP_

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "core.hpp"
#include "coco_names.hpp"

namespace yolox_cpp
{
class YoloXOpenVINO : public AbcYoloX
{
public:
  YoloXOpenVINO(
    file_name_t path_to_model,
    std::string device_name,
    float nms_th = 0.45,
    float conf_th = 0.3,
    std::string model_version = "0.1.1rc0",
    int num_classes = 80,
    bool p6 = false);
  std::vector<Object> inference(const cv::Mat & frame) override;

private:
  std::string device_name_;
  std::vector<float> blob_;
  ov::Shape input_shape_;
  ov::InferRequest infer_request_;
};
}  // namespace yolox_cpp

#endif  // YOLOX_CPP__YOLOX_OPENVINO_HPP_
