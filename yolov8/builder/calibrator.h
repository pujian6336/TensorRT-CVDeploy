#pragma once
#include <NvInfer.h>
#include "utils/config.h"

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
	Int8EntropyCalibrator2(buildConf cfg);
	int getBatchSize() const noexcept override;
	bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
	const void* readCalibrationCache(std::size_t& length) noexcept override;
	void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
	std::vector<std::string> m_fileNames;

	float* m_deviceBatchData{ nullptr };
	std::vector<char> m_calibrationCache;

	int m_curBatch{ 0 };
	int m_batchCount;
	buildConf m_cfg;
	int m_imgSize;

};


