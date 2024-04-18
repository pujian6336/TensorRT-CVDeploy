#include "trt_builder.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "buffers.h"
#include "calibrator.h"

void buildEngine(buildConf cfg)
{
    const char *onnx_file_path = cfg.onnx_model_path.c_str();

	auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        sample::gLogError << "Failed to create builder" << std::endl;
        return;
    }
    // 显性batch
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        sample::gLogError << "Failed to create network" << std::endl;
        return ;
    }

    // 创建onnxparser，用于解析onnx文件
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    // 调用onnxparser的parseFromFile方法解析onnx文件
    auto parsed = parser->parseFromFile(onnx_file_path, static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        sample::gLogError << "Failed to parse onnx file" << std::endl;
        return ;
    }
    // 获取输入节点维度信息
    nvinfer1::ITensor* inputTensor = network->getInput(0);
    nvinfer1::Dims inputDims = inputTensor->getDimensions();

    // ========== 3. 创建config配置：builder--->config ==========
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        sample::gLogError << "Failed to create config" << std::endl;
        return;
    }
    
    if (inputDims.d[0] == -1)
    {
        auto profile = builder->createOptimizationProfile();                                                           // 创建profile，用于设置输入的动态尺寸
        profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ 1, 3, cfg.input_height, cfg.input_width }); // 设置最小尺寸
        profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ cfg.batch_size, 3, cfg.input_height, cfg.input_width }); // 设置最优尺寸
        profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ cfg.batch_size, 3, cfg.input_height, cfg.input_width }); // 设置最大尺寸
        // 使用addOptimizationProfile方法添加profile，用于设置输入的动态尺寸
        config->addOptimizationProfile(profile);
    }
    
    if (cfg.mode == Mode::FP16)
    {
        if (builder->platformHasFastFp16())
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        else {
            sample::gLogError << "device don't support fp16" << std::endl;
            return;
        }
    }
    
    if (cfg.mode == Mode::INT8)
    {
        if (cfg.dataDir.empty())
        {
            sample::gLogError << "dataDir is empty!" << std::endl;
            return;
        }
        if (builder->platformHasFastInt8()) {
            auto calibrator = new Int8EntropyCalibrator2(cfg);
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            config->setInt8Calibrator(calibrator);
        }
        else
        {
            sample::gLogError << "device don't support int8." << std::endl;
            return;
        }
    }

    // 设置最大batchsize
    builder->setMaxBatchSize(cfg.batch_size);
    // 设置最大工作空间（新版本的TensorRT已经废弃了setWorkspaceSize）
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, cfg.maxWorkspaceSize);

    // 创建流，用于设置profile
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        sample::gLogError << "Failed to create stream" << std::endl;
        return;
    }
    config->setProfileStream(*profileStream);

    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
        sample::gLogError << "Failed to create engine" << std::endl;
        return;
    }

    std::ofstream engine_file(cfg.trt_model_save_path, std::ios::binary);
    assert(engine_file.is_open() && "Failed to open engine file");
    engine_file.write((char*)plan->data(), plan->size());
    engine_file.close();

    sample::gLogInfo << "Engine build success!" << std::endl;
    return;
}
