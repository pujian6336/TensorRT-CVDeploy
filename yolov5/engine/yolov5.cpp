#include "logger.h"
#include "process/postprocess.h"
#include "process/preprocess.h"
#include "yolov5.h"
#include "utils/utils.h"
#include <fstream>
#include <vector>



YOLO::YOLO(Config cfg) : m_cfg(cfg)
{
	cuda_preprocess_init(m_cfg.src_height * m_cfg.src_width * m_cfg.batch_size);

	CUDA_CHECK(cudaStreamCreate(&stream));

	m_output_objects_device = nullptr;
	m_output_objects_host = nullptr;

	m_dst2src.resize(m_cfg.batch_size);

	m_output_objects_width = 7;
}

YOLO::~YOLO()
{
	cuda_preprocess_destroy();

	cudaStreamDestroy(stream);
	cudaFree(buffers[0]);
	cudaFree(buffers[1]);
}

bool YOLO::init()
{
	std::vector<unsigned char> engine_data = utils::load_engine_model(m_cfg.model_path);
	if (engine_data.empty())
	{
		sample::gLogError << "engine file is empty" << std::endl;
		return false;
	}

	// 1. init engine & context
	this->m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
	if (!m_runtime)
	{
		sample::gLogError << "runtime create failed" << std::endl;
		return false;
	}

	this->m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
	if (!m_engine)
	{
		sample::gLogError << "deserializeCudaEngine failed!" << std::endl;
		return false;
	}

	this->m_context = std::unique_ptr<nvinfer1::IExecutionContext>(this->m_engine->createExecutionContext());
	if (!m_context)
	{
		sample::gLogError << "context create failed!" << std::endl;
		return false;
	}

	// 2. get output's dim  
	int inputIndex = m_engine->getBindingIndex(m_cfg.input_output_nodeNames[0].c_str()); // 输入索引
	int outputIndex = m_engine->getBindingIndex(m_cfg.input_output_nodeNames[1].c_str()); //输出索引

	// 获取输入维度信息
	nvinfer1::Dims input_dims = m_context->getBindingDimensions(inputIndex);
	if (m_cfg.batch_size != input_dims.d[0])
	{
		this->m_context->setBindingDimensions(inputIndex, nvinfer1::Dims4(m_cfg.batch_size, 3, m_cfg.input_height, m_cfg.input_width));
	}

	// 获取输出维度信息
	m_output_dims = m_context->getBindingDimensions(outputIndex);
	if (m_output_dims.nbDims <0)
	{
		sample::gLogError << "output nodeName error!" << std::endl;
		return false;
	}
	if (m_output_dims.nbDims == 2) { m_usePlugin = true; }
	m_output_area = 1;
	for (size_t i = 1; i < m_output_dims.nbDims; i++)
	{
		if (m_output_dims.d[i] != 0)
		{
			m_output_area *= m_output_dims.d[i]; // 每个batch的输出维度信息
		}
	}
	if (!m_usePlugin)
	{
		m_total_objects = m_output_dims.d[1]; // 预测目标数量 80*80*3+40*40*3+20*20*3
		m_classes_nums = m_output_dims.d[2] - 5; // 类别数量

		CUDA_CHECK(cudaMalloc((void **) & m_output_objects_device, m_cfg.batch_size * (m_cfg.max_det * m_output_objects_width + 1) * sizeof(float)));
		m_output_objects_host = new float[m_cfg.batch_size * (m_cfg.max_det * m_output_objects_width + 1)];
		m_topK = m_cfg.max_det;
	}
	else
	{
		m_output_objects_host = new float[m_output_area * m_cfg.batch_size];
		m_topK = (m_output_dims.d[1] - 1) / m_output_objects_width;
	}

	CUDA_CHECK(cudaMalloc(&buffers[0], m_cfg.batch_size * m_cfg.input_height * m_cfg.input_width * 3 * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&buffers[1], m_cfg.batch_size * m_output_area * sizeof(float)));
	// 绑定输入输出数据地址
	m_context->setTensorAddress(m_cfg.input_output_nodeNames[0].c_str(), buffers[0]);
	m_context->setTensorAddress(m_cfg.input_output_nodeNames[1].c_str(), buffers[1]);

	return true;
}

void YOLO::Run(const cv::Mat& img, std::vector<Detection> &res)
{
	preprocess(img);
	infer();
	postprocess(res);
}

void YOLO::Run(const std::vector<cv::Mat>& imgsBatch, std::vector<std::vector<Detection>>& res)
{
	preprocess(imgsBatch);
	infer();
	postprocess(res);
}

void YOLO::preprocess(const cv::Mat& img)
{
	cuda_preprocess(img, (float*)buffers[0], m_cfg, m_dst2src[0], stream);
#if 0
	{
		float* phost = new float[3 * m_cfg.input_width * m_cfg.input_height];

		CUDA_CHECK(cudaMemcpyAsync(phost, buffers[0],
			sizeof(float) * 3 * m_cfg.input_width * m_cfg.input_height, cudaMemcpyDeviceToHost));
		cv::Mat r(m_cfg.input_width, m_cfg.input_height, CV_32FC1, phost);
		cv::Mat g(m_cfg.input_width, m_cfg.input_height, CV_32FC1, phost + m_cfg.input_width * m_cfg.input_height);
		cv::Mat b(m_cfg.input_width, m_cfg.input_height, CV_32FC1, phost + m_cfg.input_width * m_cfg.input_height * 2);
		std::vector<cv::Mat> bgr{ b, g, r };
		cv::Mat ret;
		cv::merge(bgr, ret);
		ret.convertTo(ret, CV_8UC3, 255, 0.0);
		cv::imshow("zxx", ret);
		cv::waitKey(100);
	}
#endif
}

void YOLO::preprocess(const std::vector<cv::Mat>& imgsBatch)
{
	m_dst2src.clear();
	cuda_batch_preprocess(imgsBatch, (float*)buffers[0], m_cfg, m_dst2src, stream);
#if 0
	{
		float* phost = new float[3 * m_cfg.input_width * m_cfg.input_height];
		float* buffer0 = reinterpret_cast<float*>(buffers[0]);

		for (int i = 0; i < m_cfg.batch_size; i++)
		{
			cudaMemcpyAsync(phost, buffer0 + i * 3 * m_cfg.input_width * m_cfg.input_height,
				sizeof(float) * 3 * m_cfg.input_width * m_cfg.input_height, cudaMemcpyDeviceToHost);
			cv::Mat r(m_cfg.input_width, m_cfg.input_height, CV_32FC1, phost);
			cv::Mat g(m_cfg.input_width, m_cfg.input_height, CV_32FC1, phost + m_cfg.input_width * m_cfg.input_height);
			cv::Mat b(m_cfg.input_width, m_cfg.input_height, CV_32FC1, phost + m_cfg.input_width * m_cfg.input_height * 2);
			std::vector<cv::Mat> bgr{ b, g, r };
			cv::Mat ret;
			cv::merge(bgr, ret);
			ret.convertTo(ret, CV_8UC3, 255, 0.0);
			cv::imshow("zxx", ret);
			cv::waitKey(0);
		}
	}
#endif
}

bool YOLO::infer()
{
	return m_context->enqueueV3(stream);	
}

void YOLO::postprocess(std::vector<Detection>& res)
{
	if (m_usePlugin)
	{
		CUDA_CHECK(cudaMemcpyAsync(m_output_objects_host, buffers[1], m_output_area * sizeof(float), cudaMemcpyDeviceToHost, stream));
		// 流同步
		CUDA_CHECK(cudaStreamSynchronize(stream));
		process(res, m_output_objects_host, m_cfg.input_width, m_cfg.input_height, m_topK, m_output_objects_width, m_dst2src[0]);
	}
	else
	{
		CUDA_CHECK(cudaMemsetAsync(m_output_objects_device, 0, sizeof(float) * (m_cfg.max_det * m_output_objects_width + 1), stream));
		cuda_decode((float*)buffers[1], m_output_objects_device, 1, m_total_objects, m_classes_nums, m_cfg.max_det, m_cfg.conf_threshold, stream);
		cuda_nms(m_output_objects_device, m_cfg.iou_threshold, m_cfg.max_det, m_output_objects_width, stream);

		CUDA_CHECK(cudaMemcpyAsync(m_output_objects_host, m_output_objects_device, m_cfg.batch_size * (m_cfg.max_det * m_output_objects_width + 1) * sizeof(float), cudaMemcpyDeviceToHost, stream));
		// 流同步
		CUDA_CHECK(cudaStreamSynchronize(stream));
		process(res, m_output_objects_host, m_cfg.input_width, m_cfg.input_height, m_topK, m_output_objects_width,  m_dst2src[0]);
	}
}

void YOLO::postprocess(std::vector<std::vector<Detection>>& res)
{
	if (m_usePlugin)
	{
		CUDA_CHECK(cudaMemcpyAsync(m_output_objects_host, buffers[1], m_cfg.batch_size * m_output_area * sizeof(float), cudaMemcpyDeviceToHost, stream));
		// 流同步
		CUDA_CHECK(cudaStreamSynchronize(stream));
		batch_process(res, m_output_objects_host, m_cfg.batch_size, m_cfg.input_width, m_cfg.input_height, m_topK, m_output_objects_width,  m_dst2src);
	}
	else
	{
		CUDA_CHECK(cudaMemsetAsync(m_output_objects_device, 0, m_cfg.batch_size * (m_cfg.max_det * m_output_objects_width + 1) * sizeof(float), stream));
		cuda_decode((float *)buffers[1], m_output_objects_device, m_cfg.batch_size, m_total_objects, m_classes_nums, m_cfg.max_det, m_cfg.conf_threshold, stream);
		cuda_nms_batch(m_output_objects_device, m_cfg.batch_size, m_cfg.iou_threshold, m_cfg.max_det, m_output_objects_width, stream);

		CUDA_CHECK(cudaMemcpyAsync(m_output_objects_host, m_output_objects_device, m_cfg.batch_size * (m_cfg.max_det * m_output_objects_width + 1) * sizeof(float), cudaMemcpyDeviceToHost, stream));
		// 流同步
		CUDA_CHECK(cudaStreamSynchronize(stream));
		batch_process(res, m_output_objects_host, m_cfg.batch_size, m_cfg.input_width, m_cfg.input_height, m_topK, m_output_objects_width,  m_dst2src);
	}
}

void YOLO::postprocess_cpu(std::vector<std::vector<Detection>>& res)
{
	CUDA_CHECK(cudaMemsetAsync(m_output_objects_device, 0, sizeof(float) * (m_cfg.max_det * m_output_objects_width + 1), stream));
	cuda_decode((float*)buffers[1], m_output_objects_device, m_cfg.batch_size, m_total_objects, m_classes_nums, m_cfg.max_det, m_cfg.conf_threshold, stream);
	CUDA_CHECK(cudaMemcpyAsync(m_output_objects_host, m_output_objects_device, m_cfg.batch_size * (m_cfg.max_det * m_output_objects_width + 1) * sizeof(float), cudaMemcpyDeviceToHost, stream));
	// 流同步
	CUDA_CHECK(cudaStreamSynchronize(stream));

	batch_nms(res, m_output_objects_host, m_cfg.batch_size, m_cfg.max_det * m_output_objects_width + 1, m_cfg.conf_threshold, m_cfg.iou_threshold, m_cfg.max_det);
	box2OriginalSize(res, m_dst2src, m_cfg.input_width, m_cfg.input_height);
}

void YOLO::warmUp(int epoch)
{
	cv::Mat img(m_cfg.input_height, m_cfg.input_width, CV_8UC3, cv::Scalar(128,128,128));
	std::vector<cv::Mat> img_batch;
	for (int i = 0; i < m_cfg.batch_size; i++)
	{
		img_batch.push_back(img);
	}

	for (int i = 0; i < epoch; i++)
	{
		preprocess(img_batch);
		infer();
	}
}

