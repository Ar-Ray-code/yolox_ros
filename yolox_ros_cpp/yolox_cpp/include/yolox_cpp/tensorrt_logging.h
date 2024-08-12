// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


#ifndef YOLOX_CPP__TENSORRT_LOGGING_H_
#define YOLOX_CPP__TENSORRT_LOGGING_H_

#include "NvInferRuntimeCommon.h"  // NOLINT
#include <cassert>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

namespace yolox_cpp
{
  using Severity = nvinfer1::ILogger::Severity; // NOLINT

  class LogStreamConsumerBuffer: public std::stringbuf
  {
public:
    LogStreamConsumerBuffer(
      std::ostream & stream, const std::string & prefix,
      bool shouldLog) : mOutput(stream), mPrefix(prefix), mShouldLog(shouldLog)                                                                         // NOLINT
    {
    }

    LogStreamConsumerBuffer(LogStreamConsumerBuffer && other) noexcept : mOutput(other.mOutput),
    mPrefix(other.mPrefix), mShouldLog(other.mShouldLog)                                                                                               // NOLINT
    {
    }

    LogStreamConsumerBuffer(const LogStreamConsumerBuffer & other) = delete;
    LogStreamConsumerBuffer() = delete;
    LogStreamConsumerBuffer & operator = (const LogStreamConsumerBuffer &) = delete;
    LogStreamConsumerBuffer & operator = (LogStreamConsumerBuffer &&) = delete;

    ~LogStreamConsumerBuffer() override
    {
      if (pbase() != pptr()) {
        putOutput();
      }
    }

    int32_t sync() override
    {
      putOutput();
      return 0;
    }

    void putOutput()
    {
      if (mShouldLog) {
        // prepend timestamp
        std::time_t timestamp = std::time(nullptr);
        tm * tm_local = std::localtime(&timestamp);
        mOutput << "[";
        mOutput << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/";
        mOutput << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
        mOutput << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
        mOutput << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
        mOutput << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
        mOutput << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
        // std::stringbuf::str() gets the string contents of the buffer
        // insert the buffer contents pre-appended by the appropriate prefix into the stream
        mOutput << mPrefix << str();
      }
      // set the buffer to empty
      str("");
      // flush the stream
      mOutput.flush();
    }

    void setShouldLog(bool shouldLog)
    {
      mShouldLog = shouldLog;
    }

private:
    std::ostream & mOutput;
    std::string mPrefix;
    bool mShouldLog {};
  };


  class LogStreamConsumerBase
  {
public:
    LogStreamConsumerBase(std::ostream & stream, const std::string & prefix, bool shouldLog)
      : mBuffer(stream, prefix, shouldLog) {
    }

protected:
    LogStreamConsumerBuffer mBuffer;
  };

  class LogStreamConsumer: protected LogStreamConsumerBase, public std::ostream
  {
public:
    LogStreamConsumer(
      nvinfer1::ILogger::Severity reportableSeverity,
      nvinfer1::ILogger::Severity severity) : LogStreamConsumerBase(
        severityOstream(
          severity), severityPrefix(severity), severity <= reportableSeverity), std::ostream(
        &mBuffer),
      mShouldLog(severity <= reportableSeverity), mSeverity(severity)                                                                                                                                                                                                                                              // NOLINT
    {
    }

    LogStreamConsumer(
      LogStreamConsumer &&
      other) noexcept : LogStreamConsumerBase(
      severityOstream(other.mSeverity),
      severityPrefix(other.mSeverity), other.mShouldLog), std::ostream(&mBuffer), mShouldLog(
      other.mShouldLog), mSeverity(other.mSeverity)                                                                                                                                                                                                      // NOLINT
    {
    }

    LogStreamConsumer(const LogStreamConsumer & other) = delete;
    LogStreamConsumer() = delete;
    ~LogStreamConsumer() = default;
    LogStreamConsumer & operator = (const LogStreamConsumer &) = delete;
    LogStreamConsumer & operator = (LogStreamConsumer &&) = delete;

    void setReportableSeverity(Severity reportableSeverity)
    {
      mShouldLog = mSeverity <= reportableSeverity;
      mBuffer.setShouldLog(mShouldLog);
    }

private:
    static std::ostream & severityOstream(Severity severity)
    {
      return severity >= Severity::kINFO ? std::cout : std::cerr;
    }

    static std::string severityPrefix(Severity severity)
    {
      switch (severity) {
        case Severity::kINTERNAL_ERROR: return "[F] ";
        case Severity::kERROR: return "[E] ";
        case Severity::kWARNING: return "[W] ";
        case Severity::kINFO: return "[I] ";
        case Severity::kVERBOSE: return "[V] ";
        default: assert(0); return "";
      }
    }

    bool mShouldLog;
    Severity mSeverity;
  };


  class Logger: public nvinfer1::ILogger
  {
public:
    explicit Logger(Severity severity = Severity::kWARNING)
    : mReportableSeverity(severity) {
    }

    //!
    //! \enum TestResult
    //! \brief Represents the state of a given test
    //!
    enum class TestResult
    {
      kRUNNING,    //!< The test is running
      kPASSED,    //!< The test passed
      kFAILED,    //!< The test failed
      kWAIVED     //!< The test was waived
    };

    nvinfer1::ILogger & getTRTLogger() noexcept
    {
      return *this;
    }

    void log(Severity severity, const char * msg) noexcept override
    {
      LogStreamConsumer(
        mReportableSeverity, severity) << "[TRT] " << std::string(msg) << std::endl;
    }

    void setReportableSeverity(Severity severity) noexcept
    {
      mReportableSeverity = severity;
    }

    class TestAtom
    {
public:
      TestAtom(TestAtom &&) = default;

private:
      friend class Logger;

      TestAtom(bool started, const std::string & name, const std::string & cmdline)
        : mStarted(started),
        mName(name),
        mCmdline(cmdline)
      {
      }

      bool mStarted;
      std::string mName;
      std::string mCmdline;
    };


    static TestAtom defineTest(const std::string & name, const std::string & cmdline)
    {
      return TestAtom(false, name, cmdline);
    }

    static TestAtom defineTest(const std::string & name, int32_t argc, char const * const * argv)
    {
      // Append TensorRT version as info
      const std::string vname = name + " [TRT v" + std::to_string(NV_TENSORRT_VERSION) + "]";
      auto cmdline = genCmdlineString(argc, argv);
      return defineTest(vname, cmdline);
    }

    static void reportTestStart(TestAtom & testAtom)
    {
      reportTestResult(testAtom, TestResult::kRUNNING);
      assert(!testAtom.mStarted);
      testAtom.mStarted = true;
    }

    static void reportTestEnd(TestAtom const & testAtom, TestResult result)
    {
      assert(result != TestResult::kRUNNING);
      assert(testAtom.mStarted);
      reportTestResult(testAtom, result);
    }

    static int32_t reportPass(TestAtom const & testAtom)
    {
      reportTestEnd(testAtom, TestResult::kPASSED);
      return EXIT_SUCCESS;
    }

    static int32_t reportFail(TestAtom const & testAtom)
    {
      reportTestEnd(testAtom, TestResult::kFAILED);
      return EXIT_FAILURE;
    }

    static int32_t reportWaive(TestAtom const & testAtom)
    {
      reportTestEnd(testAtom, TestResult::kWAIVED);
      return EXIT_SUCCESS;
    }

    static int32_t reportTest(TestAtom const & testAtom, bool pass)
    {
      return pass ? reportPass(testAtom) : reportFail(testAtom);
    }

    Severity getReportableSeverity() const
    {
      return mReportableSeverity;
    }

private:
    static const char * severityPrefix(Severity severity)
    {
      switch (severity) {
        case Severity::kINTERNAL_ERROR: return "[F] ";
        case Severity::kERROR: return "[E] ";
        case Severity::kWARNING: return "[W] ";
        case Severity::kINFO: return "[I] ";
        case Severity::kVERBOSE: return "[V] ";
        default: assert(0); return "";
      }
    }

    static const char * testResultString(TestResult result)
    {
      switch (result) {
        case TestResult::kRUNNING: return "RUNNING";
        case TestResult::kPASSED: return "PASSED";
        case TestResult::kFAILED: return "FAILED";
        case TestResult::kWAIVED: return "WAIVED";
        default: assert(0); return "";
      }
    }

    static std::ostream & severityOstream(Severity severity)
    {
      return severity >= Severity::kINFO ? std::cout : std::cerr;
    }

    static void reportTestResult(TestAtom const & testAtom, TestResult result)
    {
      severityOstream(Severity::kINFO)
        << "&&&& " << testResultString(result) << " " << testAtom.mName << " # "
        << testAtom.mCmdline << std::endl;
    }

    static std::string genCmdlineString(int32_t argc, char const * const * argv)
    {
      std::stringstream ss;
      for (int32_t i = 0; i < argc; i++) {
        if (i > 0) {
          ss << " ";
        }
        ss << argv[i];
      }
      return ss.str();
    }

    Severity mReportableSeverity;
  };

  inline LogStreamConsumer LOG_VERBOSE(const Logger & logger)
  {
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kVERBOSE);
  }

  inline LogStreamConsumer LOG_INFO(const Logger & logger)
  {
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kINFO);
  }

  inline LogStreamConsumer LOG_WARN(const Logger & logger)
  {
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kWARNING);
  }

  inline LogStreamConsumer LOG_ERROR(const Logger & logger)
  {
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kERROR);
  }

  inline LogStreamConsumer LOG_FATAL(const Logger & logger)
  {
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kINTERNAL_ERROR);
  }

}  //  namespace yolox_cpp

#endif  // YOLOX_CPP__TENSORRT_LOGGING_H_
