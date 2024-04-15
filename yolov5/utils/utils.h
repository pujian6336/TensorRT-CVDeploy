#pragma once

#include <cuda_runtime_api.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "types.h"


#ifndef CUDA_CHECK
#define CUDA_CHECK(call) cuda_check(call,#call,__LINE__,__FILE__)
#endif // !CUDA_CHECK

bool cuda_check(cudaError_t err, const char* call, int iLine, const char* szFile);

namespace utils {
	namespace dataSets
	{
		const std::vector<std::string> coco80 = {
			"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
			"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
			"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
			"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
			"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
			"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
			"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
			"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
			"hair drier", "toothbrush"
		};
	}
	namespace Colors
	{
		const std::vector<cv::Scalar> color80{
			cv::Scalar(128, 77, 207),cv::Scalar(65, 32, 208),cv::Scalar(0, 224, 45),cv::Scalar(3, 141, 219),cv::Scalar(80, 239, 253),cv::Scalar(239, 184, 12),
			cv::Scalar(7, 144, 145),cv::Scalar(161, 88, 57),cv::Scalar(0, 166, 46),cv::Scalar(218, 113, 53),cv::Scalar(193, 33, 128),cv::Scalar(190, 94, 113),
			cv::Scalar(113, 123, 232),cv::Scalar(69, 205, 80),cv::Scalar(18, 170, 49),cv::Scalar(89, 51, 241),cv::Scalar(153, 191, 154),cv::Scalar(27, 26, 69),
			cv::Scalar(20, 186, 194),cv::Scalar(210, 202, 167),cv::Scalar(196, 113, 204),cv::Scalar(9, 81, 88),cv::Scalar(191, 162, 67),cv::Scalar(227, 73, 120),
			cv::Scalar(177, 31, 19),cv::Scalar(133, 102, 137),cv::Scalar(146, 72, 97),cv::Scalar(145, 243, 208),cv::Scalar(2, 184, 176),cv::Scalar(219, 220, 93),
			cv::Scalar(238, 153, 134),cv::Scalar(197, 169, 160),cv::Scalar(204, 201, 106),cv::Scalar(13, 24, 129),cv::Scalar(40, 38, 4),cv::Scalar(5, 41, 34),
			cv::Scalar(46, 94, 129),cv::Scalar(102, 65, 107),cv::Scalar(27, 11, 208),cv::Scalar(191, 240, 183),cv::Scalar(225, 76, 38),cv::Scalar(193, 89, 124),
			cv::Scalar(30, 14, 175),cv::Scalar(144, 96, 90),cv::Scalar(181, 186, 86),cv::Scalar(102, 136, 34),cv::Scalar(158, 71, 15),cv::Scalar(183, 81, 247),
			cv::Scalar(73, 69, 89),cv::Scalar(123, 73, 232),cv::Scalar(4, 175, 57),cv::Scalar(87, 108, 23),cv::Scalar(105, 204, 142),cv::Scalar(63, 115, 53),
			cv::Scalar(105, 153, 126),cv::Scalar(247, 224, 137),cv::Scalar(136, 21, 188),cv::Scalar(122, 129, 78),cv::Scalar(145, 80, 81),cv::Scalar(51, 167, 149),
			cv::Scalar(162, 173, 20),cv::Scalar(252, 202, 17),cv::Scalar(10, 40, 3),cv::Scalar(150, 90, 254),cv::Scalar(169, 21, 68),cv::Scalar(157, 148, 180),
			cv::Scalar(131, 254, 90),cv::Scalar(7, 221, 102),cv::Scalar(19, 191, 184),cv::Scalar(98, 126, 199),cv::Scalar(210, 61, 56),cv::Scalar(252, 86, 59),
			cv::Scalar(102, 195, 55),cv::Scalar(160, 26, 91),cv::Scalar(60, 94, 66),cv::Scalar(204, 169, 193),cv::Scalar(126, 4, 181),cv::Scalar(229, 209, 196),
			cv::Scalar(195, 170, 186),cv::Scalar(155, 207, 148)
		};
	}

	// 在图像上绘制检测框
	void show(const std::vector<Detection>& objects, const std::vector<std::string>& classNames,
		const int& cvDelayTime, cv::Mat& img);

	void drow_mask_bbox(cv::Mat& img, std::vector<Detection>& dets, const std::vector<std::string>& classNames, std::vector<cv::Mat>& masks, const int& cvDelayTime);


	// 统计主机程序运行时间
	class HostTime {
	public:
		HostTime();
		float getUsedTime();
		~HostTime();

	private:
		std::chrono::steady_clock::time_point t1;
		std::chrono::steady_clock::time_point t2;
	};

	// 统计CUDA程序运行时间
	class CUDATimer {
	public:
		CUDATimer();
		float getUsedTime();
		// overload 重载
		CUDATimer(cudaStream_t ctream);
		float getUsedTime(cudaStream_t ctream);

		~CUDATimer();

	private:
		cudaEvent_t start, end;
	};

	void save_txt(const std::vector<std::vector<Detection>>& objects, const std::vector<std::string>& savePath, std::vector<cv::Mat>& imgsBatch);

	void replace_root_extension(std::vector<std::string>& filePath,
		const std::string& rootPath, const std::string& newPath, const std::string& extension);

	std::vector<unsigned char> load_engine_model(const std::string& file_name);
}