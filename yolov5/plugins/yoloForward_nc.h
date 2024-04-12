#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

cudaError_t cudaYoloLayer_int8(const void* input, void* output, const uint32_t& batchSize, const uint32_t& topK, const uint32_t& numClasses,
	const float scoreThreshold, const uint32_t netWidth, const uint32_t netHeight, const void* anchors,
	const uint32_t gridSizeX, const uint32_t gridSizeY, const uint32_t& numAnchors, int threadCount,float scale, cudaStream_t stream);

cudaError_t cudaYoloLayer_nc(const void* input, void* output, const uint32_t& batchSize, const uint32_t& topK, const uint32_t& numClasses,
	const float scoreThreshold, const uint32_t netWidth, const uint32_t netHeight, const void* anchors,
	const uint32_t gridSizeX, const uint32_t gridSizeY, const uint32_t& numAnchors, int threadCount, cudaStream_t stream);
