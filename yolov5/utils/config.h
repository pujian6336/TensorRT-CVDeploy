#pragma once
#include <string>
#include <vector>
#include "utils/types.h"

struct buildConf
{
	std::string onnx_model_path; 
	std::string trt_model_save_path = "./weights_row/yolov5s_fp32.cfm";

	int batch_size = 1;
	int input_width = 640;
	int input_height = 640;

	Mode mode = Mode::FP16;
	// 1ul << 30 = 1GB, 对于嵌入式设备，maxWorkspaceSize设置小一点，比如1ul << 27 = 128MB,
	size_t maxWorkspaceSize = 1ul << 30;

	// int8量化所需参数
	std::string input_nodeNames = "images";
	std::string cacheFileName = "./calibration.cache";
	// 当前仅支持jpg, png, bmp三种格式图像。
	std::string dataDir="E:/coco_calib";
};


struct Config {
  // model information
  std::string model_path;

  float iou_threshold = 0.45f;
  float conf_threshold = 0.25f;
  int max_det{ 1000 };

  int batch_size = 1;
  int input_width = 640;
  int input_height = 640;

  // 用于初始化时申请cuda内存，申请时请尽可能只申请原图大小，避免资源浪费
  int src_height = 3000;
  int src_width = 3000;

  // ------------------------------------------------------------------
  // 以下参数只在特定情况下进行修改， 大部分情况下使用默认即可。
  // num_class正常可直接从网络信息中推理得到，当使用分割网络且mask维度自定义大小（不为32）时需要设置
  int num_class{ 80 };

  // 输入输出节点名称
  std::vector<std::string> input_output_nodeNames{ "images", "output0", "output1" };

  // 图像归一化参数 与训练时一致，否则预测结果不准确
  float scale{255.0f};
  float means[3] = {0.0f, 0.0f, 0.0f};
  float std[3] = {1.0f, 1.0f, 1.0f};
};

