#pragma once
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include"utils/types.h"

void process(std::vector<Detection>& res, const float* decode_ptr_host, int width, int height, int topK, int bbox_element, AffineMatrix d2s);

void batch_process(std::vector<std::vector<Detection>>& res_batch, const float* decode_ptr_host, int batch_size, int width, int height, int topK, int bbox_element, std::vector<AffineMatrix> d2s);

std::vector<cv::Mat> process_masks(const float* proto, int proto_height, int proto_width, std::vector<Detection>& dets, int mask_ratio = 4);

void box2OriginalSize(std::vector<std::vector<Detection>>& res_batch, std::vector<AffineMatrix> d2s, int width = 0, int height = 0);

void box2OriginalSize(std::vector<Detection>& res_batch, AffineMatrix d2s, int width = 0, int height = 0);

void mask2OriginalSize(std::vector<std::vector<cv::Mat>>& masks, std::vector<cv::Mat>& imgs, int width, int height);

void mask2OriginalSize(std::vector<cv::Mat>& masks, cv::Mat& imgs, int width, int height);


void cuda_decode(float* src, float* dst, int batchSize, uint32_t lengthPreBatch, const uint32_t lengthPreCell, int topK, float confThres, cudaStream_t stream);

void cuda_decodeSeg(float* src, float* dst, int batchSize, uint32_t lengthPreBatch, uint32_t numClasses, int topK, float confThres, cudaStream_t stream);

void cuda_decodePose(float* src, float* dst, int batchSize, uint32_t lengthPreBatch, uint32_t lengthPreBox, uint32_t numKeyPoints, int topK, float confThres, cudaStream_t stream);

void cuda_nms(float* parray, float nms_threshold, int max_objects, int lengthPreObject, cudaStream_t stream);

void cuda_nms_batch(float* parray, int batchSize, float nms_threshold, int max_objects, int lengthPreObject, cudaStream_t stream);

namespace yolov8obb
{
	void cuda_decodeObb(float* src, float* dst, int batchSize, uint32_t lengthPreBatch, const uint32_t numClasses, int topK, float confThres, cudaStream_t stream);

	void cuda_nms(float* parray, float nms_threshold, int max_objects, int lengthPreObject, cudaStream_t stream);

	void cuda_nms(float* parray, int batchSize, float nms_threshold, int max_objects, int lengthPreObject, cudaStream_t stream);

	void process(std::vector<OBBResult> &res, const float* decode_ptr_host, int topK, int bbox_element, AffineMatrix d2s);

	void process(std::vector<std::vector<OBBResult>>& res, const float* decode_ptr_host, int batch_size, int topK, int bbox_element, AffineMatrix d2s);
}

namespace  yolov8pose {

	void process(std::vector<KeyPointResult>& res, const float* decode_ptr_host, int width, int height, int topK, int bbox_element, AffineMatrix d2s);
	void process(std::vector<std::vector<KeyPointResult>>& res, const float* decode_ptr_host, int batch_size, int width, int height, int topK, int bbox_element, AffineMatrix d2s);

}