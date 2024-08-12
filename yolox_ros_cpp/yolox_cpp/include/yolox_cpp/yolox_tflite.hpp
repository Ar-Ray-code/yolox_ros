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

#ifndef YOLOX_CPP__YOLOX_TFLITE_HPP_
#define YOLOX_CPP__YOLOX_TFLITE_HPP_

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "tensorflow/lite/interpreter.h"  // NOLINT
#include "tensorflow/lite/kernels/register.h"  // NOLINT
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"  // NOLINT
#include "tensorflow/lite/model.h"  // NOLINT
#include "tensorflow/lite/optional_debug_tools.h"  // NOLINT
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"  // NOLINT
// #include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"  // NOLINT
// #include "tensorflow/lite/delegates/gpu/delegate.h"  // NOLINT

#include "core.hpp"
#include "coco_names.hpp"

namespace yolox_cpp
{
#define TFLITE_MINIMAL_CHECK(x) \
  if (!(x)) { \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1); \
  } // NOLINT

class YoloXTflite : public AbcYoloX
{
public:
  YoloXTflite(
    file_name_t path_to_model,
    int num_threads,
    float nms_th = 0.45,
    float conf_th = 0.3,
    std::string model_version = "0.1.1rc0",
    int num_classes = 80,
    bool p6 = false,
    bool is_nchw = true);
  ~YoloXTflite();
  std::vector<Object> inference(const cv::Mat & frame) override;

private:
  int doInference(float * input, float * output);

  int input_size_;
  int output_size_;
  bool is_nchw_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  TfLiteDelegate * delegate_;
};
}  // namespace yolox_cpp

#endif  // YOLOX_CPP__YOLOX_TFLITE_HPP_
