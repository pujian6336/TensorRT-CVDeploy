#pragma once
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "utils/config.h"


class YOLO_POSE
{
public:
	YOLO_POSE(Config cfg);
	~YOLO_POSE();
	virtual bool init();

	virtual void Run(const cv::Mat& img, std::vector<KeyPointResult>& res);
	virtual void Run(const std::vector<cv::Mat>& imgsBatch, std::vector<std::vector<KeyPointResult>>& res);

	virtual void warmUp(int epoch = 10);

public:
	virtual void preprocess(const cv::Mat& img);
	virtual void preprocess(const std::vector<cv::Mat>& imgsBatch);

	virtual bool infer();

	virtual void postprocess(std::vector<KeyPointResult> &res);
	virtual void postprocess(std::vector<std::vector<KeyPointResult>>& res);
	//virtual void postprocess_cpu(std::vector<std::vector<KeyPointResult>>& res);

protected:
	std::unique_ptr<nvinfer1::IRuntime> m_runtime;
	std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
	std::unique_ptr<nvinfer1::IExecutionContext> m_context;

protected:
	Config m_cfg;

	// 模型输出信息
	nvinfer1::Dims m_output_dims;
	
	int m_nkpts;
	int m_output_area;
	int m_total_objects;

	std::vector<AffineMatrix> m_dst2src;

	cudaStream_t m_stream;

protected:

	// 网络输入与输出数据存储地址
	void* m_buffers[2];

	// 输出
	float* m_output_device;
	float* m_output_host;
	int m_output_objects_width; // 7:left, top, right, bottom, scores, label, keepflag(nms过滤后，是否保留标志);

};
