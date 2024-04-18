#pragma once
#include <NvInfer.h>
#include "utils/config.h"
#include <opencv2/opencv.hpp>


class YOLO_SEG
{
public:
	YOLO_SEG(Config cfg);
	~YOLO_SEG();

	virtual bool init();

	virtual void Run(const cv::Mat& img, std::vector<Detection>& res, std::vector<cv::Mat>& masks);
	virtual void Run(const std::vector<cv::Mat>& imgsBatch, std::vector<std::vector<Detection>>& res, std::vector<std::vector<cv::Mat>>& masks);

	virtual void warmUp(int epoch = 10);

public:
	virtual void preprocess(const cv::Mat& img);
	virtual void preprocess(const std::vector<cv::Mat>& imgsBatch);

	virtual bool infer();

	virtual void postprocess(std::vector<Detection>& res, std::vector<cv::Mat>& masks);
	virtual void postprocess(std::vector<std::vector<Detection>>& res, std::vector<std::vector<cv::Mat>>& masks);

protected:
	std::unique_ptr<nvinfer1::IRuntime> m_runtime;
	std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
	std::unique_ptr<nvinfer1::IExecutionContext> m_context;

protected:
	Config m_cfg;
	bool m_usePlugin{ false };

	// 模型输出信息
	nvinfer1::Dims m_detect_dims;
	nvinfer1::Dims m_proto_dims;

	cudaStream_t m_stream;

	std::vector<AffineMatrix> m_dst2src;
	std::vector<cv::Mat> m_img;

	int m_classes_nums; // 类别数

	int m_total_objects; // 网络输出目标数（yolov5 80*80*3+40*40*3+20*20*3）
	int m_output_area;
	int m_proto_area;

	int m_topK; // 最大检出数

	int m_output_objects_width;

protected:
	// 网络输入与输出数据存储地址
	void* m_buffers[3];

	float* m_output_objects_device;
	float* m_output_objects_host;
	float* m_output_seg_host;

};
