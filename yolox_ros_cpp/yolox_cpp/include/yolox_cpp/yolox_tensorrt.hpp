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

#ifndef YOLOX_CPP__YOLOX_TENSORRT_HPP_
#define YOLOX_CPP__YOLOX_TENSORRT_HPP_

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "cuda_runtime_api.h"  // NOLINT
#include "NvInfer.h"  // NOLINT

#include "core.hpp"
#include "coco_names.hpp"
#include "tensorrt_logging.h"  // NOLINT

namespace yolox_cpp
{
using namespace nvinfer1;  // NOLINT

#define CHECK(status) do {auto ret = (status); \
    if (ret != 0) {std::cerr << "Cuda failure: " << ret << std::endl; abort();}} while (0)                                               // NOLINT

class YoloXTensorRT : public AbcYoloX
{
public:
  YoloXTensorRT(
    file_name_t path_to_engine,
    int device = 0,
    float nms_th = 0.45,
    float conf_th = 0.3,
    std::string model_version = "0.1.1rc0",
    int num_classes = 80,
    bool p6 = false);
  ~YoloXTensorRT();
  std::vector<Object> inference(const cv::Mat & frame) override;

private:
  void doInference(float * input, float * output);

  int DEVICE_ = 0;
  Logger gLogger_;
  std::unique_ptr<IRuntime> runtime_;
  std::unique_ptr<ICudaEngine> engine_;
  std::unique_ptr<IExecutionContext> context_;
  int output_size_;
  const int inputIndex_ = 0;
  const int outputIndex_ = 1;
  void * inference_buffers_[2];
};
}  // namespace yolox_cpp

#endif  // YOLOX_CPP__YOLOX_TENSORRT_HPP_
