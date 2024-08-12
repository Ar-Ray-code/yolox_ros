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

#ifndef YOLOX_CPP__YOLOX_HPP_
#define YOLOX_CPP__YOLOX_HPP_

#include "config.h"  // NOLINT

#ifdef ENABLE_OPENVINO
    #include "yolox_openvino.hpp"
#endif

#ifdef ENABLE_TENSORRT
    #include "yolox_tensorrt.hpp"
#endif

#ifdef ENABLE_ONNXRUNTIME
    #include "yolox_onnxruntime.hpp"
#endif

#ifdef ENABLE_TFLITE
    #include "yolox_tflite.hpp"
#endif


#endif  // YOLOX_CPP__YOLOX_HPP_
