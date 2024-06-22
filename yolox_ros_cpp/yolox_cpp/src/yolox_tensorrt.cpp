#include "yolox_cpp/yolox_tensorrt.hpp"

namespace yolox_cpp
{

    YoloXTensorRT::YoloXTensorRT(file_name_t path_to_engine, int device,
                                 float nms_th, float conf_th, std::string model_version,
                                 int num_classes, bool p6)
        : AbcYoloX(nms_th, conf_th, model_version, num_classes, p6),
          DEVICE_(device)
    {
        cudaSetDevice(this->DEVICE_);
        // create a model using the API directly and serialize it to a stream
        char *trtModelStream{nullptr};
        size_t size{0};

        std::ifstream file(path_to_engine, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
        else
        {
            std::cerr << "invalid arguments path_to_engine: " << path_to_engine << std::endl;
            return;
        }

        this->runtime_ = std::unique_ptr<IRuntime>(createInferRuntime(this->gLogger_));
        assert(this->runtime_ != nullptr);
        this->engine_ = std::unique_ptr<ICudaEngine>(this->runtime_->deserializeCudaEngine(trtModelStream, size));
        assert(this->engine_ != nullptr);
        this->context_ = std::unique_ptr<IExecutionContext>(this->engine_->createExecutionContext());
        assert(this->context_ != nullptr);
        delete[] trtModelStream;

        const auto input_name = this->engine_->getIOTensorName(this->inputIndex_);
        const auto input_dims = this->engine_->getTensorShape(input_name);
        this->input_h_ = input_dims.d[2];
        this->input_w_ = input_dims.d[3];
        std::cout << "INPUT_HEIGHT: " << this->input_h_ << std::endl;
        std::cout << "INPUT_WIDTH: " << this->input_w_ << std::endl;

        const auto output_name = this->engine_->getIOTensorName(this->outputIndex_);
        auto output_dims = this->engine_->getTensorShape(output_name);
        this->output_size_ = 1;
        for (int j = 0; j < output_dims.nbDims; ++j)
        {
            this->output_size_ *= output_dims.d[j];
        }

        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(this->engine_->getNbIOTensors() == 2);
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        assert(this->engine_->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
        assert(this->engine_->getTensorDataType(output_name) == nvinfer1::DataType::kFLOAT);

        // Create GPU buffers on device
        CHECK(cudaMalloc(&this->inference_buffers_[this->inputIndex_], 3 * this->input_h_ * this->input_w_ * sizeof(float)));
        CHECK(cudaMalloc(&this->inference_buffers_[this->outputIndex_], this->output_size_ * sizeof(float)));

        assert(this->context_->setInputShape(input_name, input_dims));
        assert(this->context_->allInputDimensionsSpecified());

        assert(this->context_->setTensorAddress(input_name, this->inference_buffers_[this->inputIndex_]));
        assert(this->context_->setTensorAddress(output_name, this->inference_buffers_[this->outputIndex_]));

        // Prepare GridAndStrides
        if (this->p6_)
        {
            generate_grids_and_stride(this->input_w_, this->input_h_, this->strides_p6_, this->grid_strides_);
        }
        else
        {
            generate_grids_and_stride(this->input_w_, this->input_h_, this->strides_, this->grid_strides_);
        }
    }

    YoloXTensorRT::~YoloXTensorRT()
    {
        CHECK(cudaFree(inference_buffers_[this->inputIndex_]));
        CHECK(cudaFree(inference_buffers_[this->outputIndex_]));
    }

    std::vector<Object> YoloXTensorRT::inference(const cv::Mat &frame)
    {
        // preprocess
        auto pr_img = static_resize(frame);
        float *input_blob = new float[pr_img.total() * 3];
        blobFromImage(pr_img, input_blob);

        // inference
        float *output_blob = new float[this->output_size_];
        this->doInference(input_blob, output_blob);

        float scale = std::min(this->input_w_ / (frame.cols * 1.0), this->input_h_ / (frame.rows * 1.0));

        std::vector<Object> objects;
        decode_outputs(output_blob, this->grid_strides_, objects, this->bbox_conf_thresh_, scale, frame.cols, frame.rows);

        delete input_blob;
        delete output_blob;
        return objects;
    }

    void YoloXTensorRT::doInference(float *input, float *output)
    {
        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(this->inference_buffers_[this->inputIndex_], input, 3 * this->input_h_ * this->input_w_ * sizeof(float), cudaMemcpyHostToDevice, stream));

        bool success = context_->enqueueV3(stream);
        if (!success)
            throw std::runtime_error("failed inference");

        CHECK(cudaMemcpyAsync(output, this->inference_buffers_[this->outputIndex_], this->output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream));

        CHECK(cudaStreamSynchronize(stream));

        // Release stream
        CHECK(cudaStreamDestroy(stream));
    }

} // namespace yolox_cpp
