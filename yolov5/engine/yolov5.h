#pragma once
#include <NvInfer.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "utils/types.h"
#include "utils/config.h"

class YOLO
{
public:
	YOLO(Config cfg);
	~YOLO();

	virtual bool init();

	virtual void Run(const cv::Mat& img, std::vector<Detection> &res);
	virtual void Run(const std::vector<cv::Mat>& imgsBatch, std::vector<std::vector<Detection>>& res);

	virtual void warmUp(int epoch = 10);

public:
	virtual void preprocess(const cv::Mat& img);
	virtual void preprocess(const std::vector<cv::Mat>& imgsBatch);

	virtual bool infer();

	virtual void postprocess(std::vector<Detection>& res);
	virtual void postprocess(std::vector<std::vector<Detection>>& res);
	virtual void postprocess_cpu(std::vector<std::vector<Detection>>& res);

protected:
	std::unique_ptr<nvinfer1::IRuntime> m_runtime;
	std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
	std::unique_ptr<nvinfer1::IExecutionContext> m_context;

protected:
	Config m_cfg;

	bool m_usePlugin{ false };

	int m_topK; // 最大检出数

	// 模型输出信息
	nvinfer1::Dims m_output_dims;
	int m_classes_nums; // 类别数
	int m_total_objects; // 网络输出目标数（yolov5 80*80*3+40*40*3+20*20*3）
	int m_output_area; //

	std::vector<AffineMatrix> m_dst2src;

	cudaStream_t stream;

protected:

	// 网络输入与输出数据存储地址
	void* buffers[2]; 

	// 输出
	float* m_output_objects_device;
	float* m_output_objects_host;
	int m_output_objects_width; // 7:left, top, right, bottom, scores, label, keepflag(nms过滤后，是否保留标志);
};
