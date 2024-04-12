#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <logger.h>

#include <NvInferPlugin.h>

// 定义插件版本和插件名
namespace {
	const char* YOLOLAYER_PLUGIN_VERSION{ "1" };
	const char* YOLOLAYER_PLUGIN_NAME{ "YoloLayer_TRT" };
}

class YoloLayer : public nvinfer1::IPluginV2DynamicExt
{
public:
	YoloLayer(const int& max_stride, const int& num_anchors, const int& num_features, const int& topK,
		const float& conf_thres, const float& iou_thres, const std::vector<float>& anchors);
	YoloLayer(const void* data, size_t length);

	nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

	// engine创建时被调用，用于初始化Plugin层
	int initialize() noexcept override { return 0; }
	// 与intialize为一对，engine销毁时被调用，用于释放initialize函数申请的资源
	void terminate() noexcept override {}

	void destroy() noexcept override { delete this; }

	size_t getSerializationSize() const noexcept override;

	void serialize(void* buffer) const noexcept override;

	int getNbOutputs() const noexcept override { return 1; }

	nvinfer1::DimsExprs getOutputDimensions(int32_t index, const nvinfer1::DimsExprs* inputs, int32_t nbInputDims,
		nvinfer1::IExprBuilder& exprBuilder) noexcept override;

	size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
		const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override {
		return 0;
	}

	bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
		override;

	const char* getPluginType() const noexcept override { return YOLOLAYER_PLUGIN_NAME; }

	const char* getPluginVersion() const noexcept override { return YOLOLAYER_PLUGIN_VERSION; }

	void setPluginNamespace(const char* pluginNamespace) noexcept override { m_namespace = pluginNamespace; }

	const char* getPluginNamespace() const noexcept override { return m_namespace.c_str(); }

	nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept
		override; 

	void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator)
		noexcept override {}

	void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInput,
		const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutput) noexcept override;

	void detachFromContext() noexcept override {}

	int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
		void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
	std::string m_namespace{ "" };

	int m_numAnchors{ 0 };
	int m_numFeatures{ 0 };
	int m_maxStride{ 0 };

	int m_topK{ 1000 };

	int m_netWidth{ 0 };
	int m_netHeight{ 0 };
	int m_numClasses{ 0 };
	std::vector<float> m_anchors;
	std::vector<nvinfer1::DimsHW> m_featureSpatialSize;

	float m_confThres{ 0.0f };
	float m_iouThres{ 0.0f };

	int m_threadCount =256;
};

class YoloLayerPluginCreator :public nvinfer1::IPluginCreator {
public:
	YoloLayerPluginCreator() noexcept;

	~YoloLayerPluginCreator() noexcept {};

	const char* getPluginName() const noexcept override { return YOLOLAYER_PLUGIN_NAME; }

	const char* getPluginVersion() const noexcept override { return YOLOLAYER_PLUGIN_VERSION; }

	const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }

	nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

	nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override {
		sample::gLogInfo << "Deserialize yoloLayer plugin: " << name << std::endl;
		return new YoloLayer(serialData, serialLength);
	}

	void setPluginNamespace(const char* libNamespace) noexcept override {
		mNamespace = libNamespace;
	}

	const char* getPluginNamespace() const noexcept override {
		return mNamespace.c_str();
	}
private:
	std::string mNamespace;
	static nvinfer1::PluginFieldCollection mFC;
	static std::vector<nvinfer1::PluginField> mPluginAttributes;
};