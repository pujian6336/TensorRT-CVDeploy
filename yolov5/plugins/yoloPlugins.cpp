#include "yoloPlugins.h"
#include "yoloPlugins.h"
#include "utils/utils.h"
#include "yoloForward_nc.h"
#include "process/postprocess.h"

#include <cassert>
#include <NvInferPlugin.h>

#define MaxNumAnchors 3
#define MaxNumFeatures 4

namespace {
	template<typename T>
	void write(char*& buffer, const T& val) {
		*reinterpret_cast<T*>(buffer) = val;
		buffer += sizeof(T);
	}

	template<typename T>
	void read(const char*& buffer, T& val) {
		val = *reinterpret_cast<const T*>(buffer);
		buffer += sizeof(T);
	}
}

//cudaError_t cudaYoloLayer_nc(const void* input, void* output,
//	const uint32_t& batchSize, const uint32_t& topK, const uint32_t& numClasses, const float scoreThreshold,
//	const uint32_t netWidth, const uint32_t netHeight, const void* anchors,
//	const uint32_t gridSizeX, const uint32_t gridSizeY, const uint32_t& numAnchors, int threadCount, cudaStream_t stream);

YoloLayer::YoloLayer(const int& max_stride, const int& num_anchors, const int& num_features,const int& topK,
	const float &conf_thres,const float&iou_thres ,const std::vector<float>& anchors) : m_maxStride(max_stride), m_numAnchors(num_anchors),
	m_numFeatures(num_features),m_topK(topK), m_confThres(conf_thres), m_iouThres(iou_thres), m_anchors(anchors) {};

YoloLayer::YoloLayer(const void* data, size_t length)
{
	const char* d = static_cast<const char*>(data);

	read(d, m_netWidth);
	read(d, m_netHeight);
	read(d, m_maxStride);
	read(d, m_numClasses);
	read(d, m_confThres);
	read(d, m_numFeatures);
	read(d, m_numAnchors);
	read(d, m_topK);
	read(d, m_iouThres);

	m_anchors.resize(m_numFeatures * m_numAnchors * 2);
	for (int i = 0; i < m_anchors.size(); ++i)
	{
		read(d, m_anchors[i]);
	}

	for (int i = 0; i < m_numFeatures; ++i)
	{
		int height;
		int width;
		read(d, height);
		read(d, width);
		m_featureSpatialSize.push_back(nvinfer1::DimsHW(height, width));
	}
}

nvinfer1::IPluginV2DynamicExt* YoloLayer::clone() const noexcept
{
	return new YoloLayer(m_maxStride, m_numAnchors, m_numFeatures, m_topK, m_confThres, m_iouThres, m_anchors);
}

size_t YoloLayer::getSerializationSize() const noexcept
{
	size_t totalSize = 0;

	totalSize += sizeof(m_netWidth);
	totalSize += sizeof(m_netHeight);
	totalSize += sizeof(m_maxStride);
	totalSize += sizeof(m_numClasses);

	totalSize += sizeof(m_confThres);
	totalSize += sizeof(m_numFeatures);
	totalSize += sizeof(m_numAnchors);
	totalSize += sizeof(m_topK);
	totalSize += sizeof(m_iouThres);

	totalSize += m_anchors.size() * sizeof(m_anchors[0]);

	totalSize += m_featureSpatialSize.size() * 2 * sizeof(m_featureSpatialSize[0].h());

	return totalSize;
}

void YoloLayer::serialize(void* buffer) const noexcept
{
	char* d = static_cast<char*>(buffer);

	write(d, m_netWidth);
	write(d, m_netHeight);
	write(d, m_maxStride);
	write(d, m_numClasses);
	write(d, m_confThres);
	write(d, m_numFeatures);
	write(d, m_numAnchors);
	write(d, m_topK);
	write(d, m_iouThres);
	
	// write anchors
	for (int i = 0; i < m_anchors.size(); ++i)
	{
		write(d, m_anchors[i]);
	}

	for (int i = 0; i < m_featureSpatialSize.size(); ++i)
	{
		write(d, m_featureSpatialSize[i].h());
		write(d, m_featureSpatialSize[i].w());
	}
}

// 输出数据维度（注意：）
nvinfer1::DimsExprs YoloLayer::getOutputDimensions(int32_t index, const nvinfer1::DimsExprs* inputs, int32_t nbInputDims, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
	return nvinfer1::DimsExprs{ 2,{inputs->d[0],exprBuilder.constant(static_cast<int>(m_topK * 7 + 1))} };
}

// 指定该插件支持的输入，输出数据类型和格式
bool YoloLayer::supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
	// 该插件仅支持{N C H W}数据格式
	// 该插件仅支持FP32数据格式，并且输入输出数据格式必须相同 inOut[pos].type == nvinfer1::DataType::kFLOAT || 
	if (pos == nbInputs+ nbOutputs-1)
	{
		return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && inOut[pos].type == nvinfer1::DataType::kFLOAT;
	}
	else {
		return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && (inOut[pos].type == nvinfer1::DataType::kINT8 || inOut[pos].type == nvinfer1::DataType::kFLOAT);
	}
}

// 输出数据只能是fp32
nvinfer1::DataType YoloLayer::getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept
{
	return nvinfer1::DataType::kFLOAT;
}

void YoloLayer::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInput, const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutput) noexcept
{
	assert(m_numFeatures == nbInput);
	m_featureSpatialSize.clear();

	for (int i = 0; i < m_numFeatures; i++)
	{
		m_featureSpatialSize.push_back(nvinfer1::DimsHW(in[i].desc.dims.d[2], in[i].desc.dims.d[3]));
	}

	m_numClasses = in[m_numFeatures - 1].desc.dims.d[1] / m_numAnchors - 5;
	m_netHeight = in[m_numFeatures - 1].desc.dims.d[2] * m_maxStride;
	m_netWidth = in[m_numFeatures - 1].desc.dims.d[3] * m_maxStride;
}

int32_t YoloLayer::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
	uint32_t batchSize = inputDesc[0].dims.d[0];

	uint32_t outputElem = 1 + 7 * m_topK;
	for (uint32_t idx = 0; idx < batchSize; ++idx) {
		cudaMemsetAsync((float *)outputs[0] + idx * outputElem, 0, sizeof(float), stream);
	}
	
	for (int i = 0; i < m_numFeatures; ++i)
	{
		uint32_t gridSizeX = m_featureSpatialSize[i].w();
		uint32_t gridSizeY = m_featureSpatialSize[i].h();
		uint32_t inputSize = gridSizeX * gridSizeY * m_numAnchors * (5 + m_numClasses);

		std::vector<float> anchors(m_anchors.begin() + i * m_numAnchors * 2, m_anchors.begin() + (i + 1) * m_numAnchors * 2);

		void* anchors_cuda = NULL;
		if (anchors.size() > 0)
		{
			CUDA_CHECK(cudaMalloc(&anchors_cuda, sizeof(float) * anchors.size()));
			CUDA_CHECK(cudaMemcpyAsync(anchors_cuda, anchors.data(), sizeof(float) * anchors.size(), cudaMemcpyHostToDevice, stream));
		}
		if (inputDesc[i].type == nvinfer1::DataType::kINT8)
		{
			float scale = inputDesc[i].scale;
			cudaYoloLayer_int8(inputs[i], outputs[0], batchSize, m_topK, m_numClasses,
				m_confThres, m_netWidth, m_netHeight, anchors_cuda,
				gridSizeX, gridSizeY, m_numAnchors, m_threadCount, scale ,stream);
		}
		else {
			cudaYoloLayer_nc(inputs[i], outputs[0], batchSize, m_topK, m_numClasses,
				m_confThres, m_netWidth, m_netHeight, anchors_cuda,
				gridSizeX, gridSizeY, m_numAnchors, m_threadCount, stream);
		}

		if (anchors.size() > 0)
		{
			CUDA_CHECK(cudaFree(anchors_cuda));
		}
	}
	cuda_nms_batch(reinterpret_cast<float*>(outputs[0]), batchSize, m_iouThres, m_topK, 7 ,stream);
	return 0;
}


nvinfer1::PluginFieldCollection YoloLayerPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> YoloLayerPluginCreator::mPluginAttributes;


YoloLayerPluginCreator::YoloLayerPluginCreator() noexcept
{
	mPluginAttributes.emplace_back(nvinfer1::PluginField("max_stride", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(nvinfer1::PluginField("num_anchors", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(nvinfer1::PluginField("num_features", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(nvinfer1::PluginField("anchors", nullptr, nvinfer1::PluginFieldType::kFLOAT32, MaxNumAnchors * MaxNumFeatures * 2));
	mPluginAttributes.emplace_back(nvinfer1::PluginField("conf_thres", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
	mPluginAttributes.emplace_back(nvinfer1::PluginField("topK", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(nvinfer1::PluginField("iou_thres", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));

	mFC.nbFields = mPluginAttributes.size();
	mFC.fields = mPluginAttributes.data();
}

nvinfer1::IPluginV2DynamicExt* YoloLayerPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept
{
	const nvinfer1::PluginField* fields = fc->fields;

	int max_stride = 32;
	int num_anchors = 3;
	int num_features = 3;
	std::vector<float> anchors;
	float conf_thres = 0.0;
	int topK = 0;
	float iou_thres = 0.0;

	for (int i = 0; i < fc->nbFields; ++i)
	{
		const char* attrName = fields[i].name;
		if (!strcmp(attrName, "max_stride"))
		{
			assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
			max_stride = *(static_cast<const int*>(fields[i].data));
		}
		if (!strcmp(attrName, "num_anchors"))
		{
			assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
			num_anchors = *(static_cast<const int*>(fields[i].data));
		}
		if (!strcmp(attrName, "num_features"))
		{
			assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
			num_features = *(static_cast<const int*>(fields[i].data));
		}
		if (!strcmp(attrName, "conf_thres"))
		{
			assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
			conf_thres = *(static_cast<const float*>(fields[i].data));
		}
		if (!strcmp(attrName, "topK"))
		{
			assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
			topK = *(static_cast<const int*>(fields[i].data));
		}
		if (!strcmp(attrName, "iou_thres"))
		{
			assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
			iou_thres = *(static_cast<const float*>(fields[i].data));
		}

	}
	for (int i = 0; i < fc->nbFields; ++i) {
		const char* attrName = fields[i].name;
		if (!strcmp(attrName, "anchors"))
		{
			assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
			const auto anchors_ptr = static_cast<const float*>(fields[i].data);
			anchors.assign(anchors_ptr, anchors_ptr + num_anchors * num_features * 2);
		}
	}
	return new YoloLayer(max_stride, num_anchors, num_features, topK, conf_thres, iou_thres, anchors);
}

REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);