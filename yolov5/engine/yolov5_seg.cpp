#include "yolov5_seg.h"
#include "utils/utils.h"
#include "process/preprocess.h"
#include "process/postprocess.h"
#include "logger.h"

YOLO_SEG::YOLO_SEG(Config cfg) :m_cfg(cfg)
{
	cuda_preprocess_init(m_cfg.batch_size * m_cfg.src_width * m_cfg.src_height);
	CUDA_CHECK(cudaStreamCreate(&m_stream));

	m_dst2src.resize(m_cfg.batch_size);

	m_output_objects_device = nullptr;
	m_output_objects_host = nullptr;
	m_output_seg_host = nullptr;

	m_img.resize(m_cfg.batch_size);

	m_output_objects_width = 39; // 7:left, top, right, bottom, scores, label, keepflag(nms过滤后，是否保留标志) + 32(mask) ;
}

YOLO_SEG::~YOLO_SEG()
{
	cuda_preprocess_destroy();
	cudaStreamDestroy(m_stream);
	cudaFree(m_buffers[0]);
	cudaFree(m_buffers[1]);
	cudaFree(m_buffers[2]);
}

bool YOLO_SEG::init()
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
	int protoIndex = m_engine->getBindingIndex(m_cfg.input_output_nodeNames[2].c_str()); //输出索引

	if (inputIndex < 0 || outputIndex < 0 || protoIndex < 0)
	{
		sample::gLogError << "input or output nodeName error!" << std::endl;
		return false;
	}

	// 获取输入维度信息
	nvinfer1::Dims input_dims = m_context->getBindingDimensions(inputIndex);
	if (m_cfg.batch_size != input_dims.d[0])
	{
		this->m_context->setBindingDimensions(inputIndex, nvinfer1::Dims4(m_cfg.batch_size, 3, m_cfg.input_height, m_cfg.input_width));
	}

	// 获取输出维度信息(detect)
	m_detect_dims = m_context->getBindingDimensions(outputIndex);
	if (m_detect_dims.nbDims == 2) { m_usePlugin = true; }
	m_output_area = 1;
	for (size_t i = 1; i < m_detect_dims.nbDims; i++)
	{
		if (m_detect_dims.d[i] != 0)
		{
			m_output_area *= m_detect_dims.d[i]; // 每个batch的输出维度信息
		}
	}
	// 获取输出维度信息(proto)
	m_proto_dims = m_context->getBindingDimensions(protoIndex);
	m_proto_area = 1;
	for (size_t i = 1; i < m_proto_dims.nbDims; i++)
	{
		if (m_proto_dims.d[i] != 0)
		{
			m_proto_area *= m_proto_dims.d[i];
		}
	}

	if (!m_usePlugin)
	{
		m_total_objects = m_detect_dims.d[1]; // 预测目标数量 80*80*3+40*40*3+20*20*3
		m_classes_nums = m_detect_dims.d[2] - 37; // 类别数量

		CUDA_CHECK(cudaMalloc((void**)&m_output_objects_device, m_cfg.batch_size * (m_cfg.max_det * m_output_objects_width + 1) * sizeof(float)));
		m_output_objects_host = new float[m_cfg.batch_size * (m_cfg.max_det * m_output_objects_width + 1)];
		m_output_seg_host = new float[m_proto_area * m_cfg.batch_size];
		m_topK = m_cfg.max_det;
	}
	else
	{
		m_output_objects_host = new float[m_output_area * m_cfg.batch_size];
		m_topK = (m_detect_dims.d[1] - 1) / m_output_objects_width;
	}



	CUDA_CHECK(cudaMalloc(&m_buffers[0], m_cfg.batch_size * m_cfg.input_height * m_cfg.input_width * 3 * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&m_buffers[1], m_cfg.batch_size * m_output_area * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&m_buffers[2], m_cfg.batch_size * m_proto_area * sizeof(float)));
	// 绑定输入输出数据地址
	m_context->setTensorAddress(m_cfg.input_output_nodeNames[0].c_str(), m_buffers[0]);
	m_context->setTensorAddress(m_cfg.input_output_nodeNames[1].c_str(), m_buffers[1]);
	m_context->setTensorAddress(m_cfg.input_output_nodeNames[2].c_str(), m_buffers[2]);

	return true;
}

void YOLO_SEG::Run(const cv::Mat& img, std::vector<Detection>& res, std::vector<cv::Mat>& masks)
{
	preprocess(img);
	infer();
	postprocess(res, masks);
}

void YOLO_SEG::Run(const std::vector<cv::Mat>& imgsBatch, std::vector<std::vector<Detection>>& res, std::vector<std::vector<cv::Mat>>& masks)
{
	preprocess(imgsBatch);
	infer();
	postprocess(res, masks);
}

void YOLO_SEG::warmUp(int epoch)
{
	cv::Mat img(m_cfg.input_height, m_cfg.input_width, CV_8UC3, cv::Scalar(128, 128, 128));
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

void YOLO_SEG::preprocess(const cv::Mat& img)
{
	cuda_preprocess(img, (float*)m_buffers[0], m_cfg, m_dst2src[0], m_stream);
	m_img[0] = img;
}

void YOLO_SEG::preprocess(const std::vector<cv::Mat>& imgsBatch)
{
	m_dst2src.clear();
	m_img.clear();
	cuda_batch_preprocess(imgsBatch, (float*)m_buffers[0], m_cfg, m_dst2src, m_stream);
	m_img = imgsBatch;
}

bool YOLO_SEG::infer()
{
	return m_context->enqueueV3(m_stream);
}

void YOLO_SEG::postprocess(std::vector<Detection>& res, std::vector<cv::Mat>& masks)
{
	CUDA_CHECK(cudaMemsetAsync(m_output_objects_device, 0, sizeof(float) * (m_cfg.max_det * m_output_objects_width + 1), m_stream));
	cuda_decodeSeg((float*)m_buffers[1], m_output_objects_device, 1, m_total_objects, m_detect_dims.d[2], m_classes_nums, m_cfg.max_det, m_cfg.conf_threshold, m_stream);

	cuda_nms(m_output_objects_device, m_cfg.iou_threshold, m_cfg.max_det, m_output_objects_width, m_stream);

	CUDA_CHECK(cudaMemcpyAsync(m_output_objects_host, m_output_objects_device, m_cfg.batch_size * (m_cfg.max_det * 38 + 1) * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
	CUDA_CHECK(cudaMemcpyAsync(m_output_seg_host, m_buffers[2], m_cfg.batch_size * m_proto_area * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
	CUDA_CHECK(cudaStreamSynchronize(m_stream));

	process(res, m_output_objects_host, m_cfg.input_width, m_cfg.input_height, m_topK, m_output_objects_width, m_dst2src[0]);

	masks = process_masks(m_output_seg_host, m_proto_dims.d[2], m_proto_dims.d[3], res);

	mask2OriginalSize(masks, m_img[0], m_cfg.input_width, m_cfg.input_height);
	box2OriginalSize(res, m_dst2src[0]);
}

void YOLO_SEG::postprocess(std::vector<std::vector<Detection>>& res, std::vector<std::vector<cv::Mat>>& masks)
{
	CUDA_CHECK(cudaMemsetAsync(m_output_objects_device, 0, sizeof(float) * (m_cfg.max_det * m_output_objects_width + 1), m_stream));
	cuda_decodeSeg((float*)m_buffers[1], m_output_objects_device, m_cfg.batch_size, m_total_objects, m_detect_dims.d[2], m_classes_nums, m_cfg.max_det, m_cfg.conf_threshold, m_stream);

	cuda_nms_batch(m_output_objects_device, m_cfg.batch_size, m_cfg.iou_threshold, m_cfg.max_det, m_output_objects_width, m_stream);

	CUDA_CHECK(cudaMemcpyAsync(m_output_objects_host, m_output_objects_device, m_cfg.batch_size * (m_cfg.max_det * 38 + 1) * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
	CUDA_CHECK(cudaMemcpyAsync(m_output_seg_host, m_buffers[2], m_cfg.batch_size * m_proto_area * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
	CUDA_CHECK(cudaStreamSynchronize(m_stream));

	batch_process(res, m_output_objects_host, m_cfg.batch_size, m_cfg.input_width, m_cfg.input_height, m_topK, m_output_objects_width, m_dst2src);

	masks.resize(m_cfg.batch_size);
	for (int i = 0; i < m_cfg.batch_size; i++)
	{
		masks[i] = process_masks(m_output_seg_host, m_proto_dims.d[2], m_proto_dims.d[3], res[0]);
	}
	mask2OriginalSize(masks, m_img, m_cfg.input_width, m_cfg.input_height);
	box2OriginalSize(res, m_dst2src);
}