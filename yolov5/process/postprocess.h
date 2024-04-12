#pragma once

#include "utils/types.h"
#include "NvInfer.h"
#include <opencv2/opencv.hpp>

//void nms(std::vector<Detection>& res,float*output, float conf_thresh,float nms_thresh, int length_pre_object=7);

void batch_nms(std::vector<std::vector<Detection>>& batch_res, float* output, int batch_size, int output_size, float conf_thres, float nms_thresh, int max_det);

void batch_process(std::vector<std::vector<Detection>>& res_batch, const float* decode_ptr_host, int batch_size,int topK, int bbox_element, std::vector<AffineMatrix> d2s);

void batch_process(std::vector<std::vector<Detection>>& res_batch, std::vector<AffineMatrix> d2s, int width, int height);

void process(std::vector<Detection>& res, const float* decode_ptr_host, int topK, int bbox_element, AffineMatrix d2s);

void cuda_decode(float* src, float* dst, int batchSize, uint32_t lengthPreBatch, const uint32_t lengthPreCell, int topK, float confThres, cudaStream_t stream);

void cuda_decodeSeg(float* src, float* dst, int batchSize, uint32_t lengthPreBatch, uint32_t lengthPreBox, uint32_t numClasses, int topK, float confThres, cudaStream_t stream);


void cuda_nms(float* parray, float nms_threshold, int max_objects, int lengthPreObject, cudaStream_t stream);

void cuda_nms_batch(float* parray, int batchSize, float nms_threshold, int max_objects, int lengthPreObject, cudaStream_t stream);


