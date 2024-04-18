#pragma once

#include <NvInfer.h>

#include <opencv2/opencv.hpp>

#include "utils/config.h"
#include "utils/types.h"

void cuda_preprocess_init(int max_image_size);

void cuda_preprocess_destroy();

// 使用cuda流，使得计算与传输可以同时进行，加快处理速度
void cuda_preprocess(const cv::Mat &srcImg, float *dst, Config cfg,
                     AffineMatrix &d2s, cudaStream_t stream);

// 重载用于int8量化时读取数据
void cuda_preprocess(const cv::Mat& srcImg, float* dst, buildConf cfg);

void cuda_batch_preprocess(const std::vector<cv::Mat> &img_batch, float *dst,
                           Config cfg, std::vector<AffineMatrix> &d2s,
                           cudaStream_t stream);