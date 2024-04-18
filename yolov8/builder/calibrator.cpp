#include "calibrator.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include "logger.h"
#include "process/preprocess.h"

std::vector<std::string> get_image_file_names(const std::string& folder_path) {
	std::vector<std::string> extensions = { "*.jpg", "*.jpeg", "*.png", "*.bmp"}; // 可根据需要添加更多格式

	std::vector<std::string> image_files;
	for (const auto& ext : extensions) {
		std::vector<std::string> found_files;
		cv::glob(folder_path + "/" + ext, found_files);
		image_files.insert(image_files.end(), found_files.begin(), found_files.end());
	}

	return image_files;
}


Int8EntropyCalibrator2::Int8EntropyCalibrator2(buildConf cfg)
{
	m_cfg = cfg;

	m_fileNames = get_image_file_names(m_cfg.dataDir);

	m_batchCount = m_fileNames.size() / m_cfg.batch_size;
	sample::gLogInfo << "CalibrationDataReader: " << m_fileNames.size() << " images, " << m_batchCount << " batches." << std::endl;

	m_imgSize = m_cfg.input_height * m_cfg.input_width;
	cuda_preprocess_init(m_imgSize);

	m_imgSize = (m_cfg.input_width * m_cfg.input_height * 3);
	cudaMalloc((void**)&m_deviceBatchData, m_cfg.batch_size * m_cfg.input_height * m_cfg.input_width * 3 * sizeof(float));

}

int Int8EntropyCalibrator2::getBatchSize() const noexcept
{
	return m_cfg.batch_size;
}

bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
{
	if (m_curBatch + 1 > m_batchCount)
	{
		return false;
	}
	int offset = m_cfg.input_height * m_cfg.input_width * 3 * sizeof(float);
	for (int i = 0; i < m_cfg.batch_size; i++)
	{
		int idx = m_curBatch * m_cfg.batch_size + i;
		cv::Mat img = cv::imread(m_fileNames[idx]);
		int new_img_size = img.cols * img.rows;
		if (new_img_size > m_imgSize)
		{
			m_imgSize = new_img_size;
			cuda_preprocess_destroy();
			cuda_preprocess_init(m_imgSize);
		}
		cuda_preprocess(img, m_deviceBatchData + i * offset, m_cfg);
	}
	for (int i = 0; i < nbBindings; i++)
	{
		if (!strcmp(names[i], m_cfg.input_nodeNames.c_str()))
		{
			bindings[i] = m_deviceBatchData + i * offset;
		}
	}
	m_curBatch++;
	return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(std::size_t& length) noexcept
{
	sample::gLogInfo << "reading calib cache : " << m_cfg.cacheFileName << std::endl;
 	m_calibrationCache.clear();

	std::ifstream input(m_cfg.cacheFileName, std::ios::binary);
	input >> std::noskipws;

	if (input.good()) {
		std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(m_calibrationCache));
	}
	length = m_calibrationCache.size();
	return length ? m_calibrationCache.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length) noexcept
{
	sample::gLogInfo << "writing calib cache: " << m_cfg.cacheFileName << "size: " << length << std::endl;
	std::ofstream output(m_cfg.cacheFileName, std::ios::binary);
	output.write(reinterpret_cast<const char*>(cache), length);
}
