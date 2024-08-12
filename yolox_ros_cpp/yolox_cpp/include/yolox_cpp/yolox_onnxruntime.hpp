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

#ifndef YOLOX_CPP__YOLOX_ONNXRUNTIME_HPP_
#define YOLOX_CPP__YOLOX_ONNXRUNTIME_HPP_

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "onnxruntime/core/session/onnxruntime_cxx_api.h"  // NOLINT

#include "core.hpp"
#include "coco_names.hpp"

namespace yolox_cpp
{
class YoloXONNXRuntime : public AbcYoloX
{
public:
  YoloXONNXRuntime(
    file_name_t path_to_model,
    int intra_op_num_threads,
    int inter_op_num_threads = 1,
    bool use_cuda = true,
    int device_id = 0,
    bool use_parallel = false,
    float nms_th = 0.45,
    float conf_th = 0.3,
    std::string model_version = "0.1.1rc0",
    int num_classes = 80,
    bool p6 = false);
  std::vector<Object> inference(const cv::Mat & frame) override;

private:
  int intra_op_num_threads_ = 1;
  int inter_op_num_threads_ = 1;
  int device_id_ = 0;
  bool use_cuda_ = true;
  bool use_parallel_ = false;

  Ort::Session session_{nullptr};
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "Default"};

  Ort::Value input_tensor_{nullptr};
  Ort::Value output_tensor_{nullptr};
  std::string input_name_;
  std::string output_name_;
  std::vector<std::unique_ptr<uint8_t[]>> input_buffer_;
  std::vector<std::unique_ptr<uint8_t[]>> output_buffer_;
};
}  // namespace yolox_cpp

#endif  // YOLOX_CPP__YOLOX_ONNXRUNTIME_HPP_
