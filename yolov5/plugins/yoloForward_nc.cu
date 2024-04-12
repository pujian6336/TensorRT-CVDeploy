#include <cuda_runtime_api.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include "yoloForward_nc.h"

__global__ void gpuYoloLayer_nc(const float* input, float* output, const uint32_t noElements,
	const uint32_t numAnchors, const float confThres,
	const uint32_t netWidth,const uint32_t netHeight, const uint32_t gridSizeX, const uint32_t gridSizeY, 
	const uint32_t numClasses, const uint32_t topK,
	const float* anchors
) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= noElements) return;

	uint32_t numGridCells = gridSizeX * gridSizeY;
	uint32_t cellSize = 5 + numClasses;

	uint32_t batchIdx = idx / (numGridCells * numAnchors);
	uint32_t anchorIdx = (idx % (numGridCells * numAnchors)) / numGridCells;
	uint32_t gridIdx = (idx % (numGridCells * numAnchors)) % numGridCells;

	uint32_t curIndex = batchIdx * numGridCells * cellSize * numAnchors + anchorIdx * cellSize * numGridCells + gridIdx;

	float objectness = input[curIndex + 4 * numGridCells];
	if (objectness < confThres) return;

	int classId = 0;
	float maxProb = 0.0;
	for (uint32_t i = 0; i < numClasses; ++i)
	{
		float prob = input[curIndex + (5 + i) * numGridCells];
		if (prob > maxProb)
		{
			maxProb = prob;
			classId = i;
		}
	}

	float score = objectness * maxProb;
	if (score < confThres) return;

	int count = (int)atomicAdd(output + batchIdx * (7 * topK + 1), 1);
	if (count > topK) { return; }

	uint32_t y = gridIdx / gridSizeX;
	uint32_t x = gridIdx % gridSizeX;
	float xc = (input[curIndex + 0 * numGridCells] * 2.0f - 0.5f + x) * netWidth / gridSizeX;
	float yc = (input[curIndex + 1 * numGridCells] * 2.0f- 0.5f + y) * netHeight / gridSizeY;
	float w = __powf(input[curIndex + 2 * numGridCells] * 2.0f, 2) * anchors[anchorIdx * 2];
	float h = __powf(input[curIndex + 3 * numGridCells] * 2.0f, 2) * anchors[anchorIdx * 2 + 1];

	float left = xc - w * 0.5f;
	float top = yc - h * 0.5f;
	float right = xc + w * 0.5f;
	float bottom = yc + h * 0.5f;

	float* pout_item = output + count * 7 + 1 + batchIdx * (topK * 7 + 1);
	*pout_item++ = left;
	*pout_item++ = top;
	*pout_item++ = right;
	*pout_item++ = bottom;
	*pout_item++ = score;
	*pout_item++ = classId;
	*pout_item++ = 1;
}

//cudaError_t cudaYoloLayer_nc(const void* input, void* output, const uint32_t& batchSize, const uint32_t& topK, const uint32_t& numClasses,
//	const float scoreThreshold, const uint32_t netWidth, const uint32_t netHeight, const void* anchors,
//	const uint32_t gridSizeX, const uint32_t gridSizeY, const uint32_t& numAnchors, int threadCount, cudaStream_t stream);

cudaError_t cudaYoloLayer_nc(const void* input, void* output, const uint32_t& batchSize, const uint32_t& topK, const uint32_t& numClasses,
	const float scoreThreshold, const uint32_t netWidth, const uint32_t netHeight, const void* anchors,
	const uint32_t gridSizeX, const uint32_t gridSizeY, const uint32_t& numAnchors, int threadCount, cudaStream_t stream)
{
	int numElem = gridSizeX * gridSizeY * numAnchors * batchSize;
	if (numElem < threadCount) threadCount = numElem;

	gpuYoloLayer_nc << <(numElem + threadCount - 1) / threadCount, threadCount, 0, stream >> > (
		reinterpret_cast<const float*>(input), reinterpret_cast<float*>(output),
		numElem, numAnchors, scoreThreshold, netWidth, netHeight, gridSizeX, gridSizeY,
		numClasses, topK, reinterpret_cast<const float*>(anchors));

	return cudaGetLastError();
}

__device__ float int8ToFloat(const int8_t value, const float scale)
{
	return static_cast<float>(value) * scale;
}

__device__ int8_t floatToInt8(const float value, const float scale)
{
	float scaleValue = value / scale;
	return static_cast<int8_t> (rintf(scaleValue));
	//return static_cast<int8_t>(__fadd_rn(scaleValue, 0.5f));
}


__global__ void gpuYoloLayer_nc(const int8_t* input, float* output, const uint32_t noElements,
	const uint32_t numAnchors, const float confThres,
	const uint32_t netWidth, const uint32_t netHeight, const uint32_t gridSizeX, const uint32_t gridSizeY,
	const uint32_t numClasses, const uint32_t topK,
	const float* anchors,const float scale
) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= noElements) return;

	uint32_t numGridCells = gridSizeX * gridSizeY;
	uint32_t cellSize = 5 + numClasses;

	uint32_t batchIdx = idx / (numGridCells * numAnchors);
	uint32_t anchorIdx = (idx % (numGridCells * numAnchors)) / numGridCells;
	uint32_t gridIdx = (idx % (numGridCells * numAnchors)) % numGridCells;

	uint32_t curIndex = batchIdx * numGridCells * cellSize * numAnchors + anchorIdx * cellSize * numGridCells + gridIdx;

	int8_t objectness = input[curIndex + 4 * numGridCells];
	int8_t confThresInt8 = floatToInt8(confThres, scale);
	if (objectness < confThresInt8) return;

	int classId = 0;
	int8_t maxProb = 0;
	for (uint32_t i = 0; i < numClasses; ++i)
	{
		int8_t prob = input[curIndex + (5 + i) * numGridCells];
		if (prob > maxProb)
		{
			maxProb = prob;
			classId = i;
		}
	}

	float score = int8ToFloat(objectness, scale) * int8ToFloat(maxProb, scale);
	if (score < confThres) return;

	int count = (int)atomicAdd(output + batchIdx * (7 * topK + 1), 1);
	if (count > topK) { return; }

	uint32_t y = gridIdx / gridSizeX;
	uint32_t x = gridIdx % gridSizeX;

	int8_t xcInt8 = input[curIndex + 0 * numGridCells];
	float xc = int8ToFloat(xcInt8, scale);
	xc = ( xc * 2.0f - 0.5f + x) * netWidth / gridSizeX;

	float yc = int8ToFloat(input[curIndex + 1 * numGridCells], scale);
	yc = (yc * 2.0f - 0.5f + y) * netHeight / gridSizeY;

	float w = int8ToFloat(input[curIndex + 2 * numGridCells], scale);
	w = __powf(w * 2.0f, 2) * anchors[anchorIdx * 2];

	float h = int8ToFloat(input[curIndex + 3 * numGridCells], scale);
	h = __powf(h * 2.0f, 2) * anchors[anchorIdx * 2 + 1];

	float left = xc - w * 0.5f;
	float top = yc - h * 0.5f;
	float right = xc + w * 0.5f;
	float bottom = yc + h * 0.5f;

	float* pout_item = output + count * 7 + 1 + batchIdx * (topK * 7 + 1);
	*pout_item++ = left;
	*pout_item++ = top;
	*pout_item++ = right;
	*pout_item++ = bottom;
	*pout_item++ = score;
	*pout_item++ = classId;
	*pout_item++ = 1;
}


cudaError_t cudaYoloLayer_int8(const void* input, void* output, const uint32_t& batchSize, const uint32_t& topK, const uint32_t& numClasses,
	const float scoreThreshold, const uint32_t netWidth, const uint32_t netHeight, const void* anchors,
	const uint32_t gridSizeX, const uint32_t gridSizeY, const uint32_t& numAnchors, int threadCount,float scale, cudaStream_t stream)
{
	int numElem = gridSizeX * gridSizeY * numAnchors * batchSize;
	if (numElem < threadCount) threadCount = numElem;

	gpuYoloLayer_nc << <(numElem + threadCount - 1) / threadCount, threadCount, 0, stream >> > (
		reinterpret_cast<const int8_t*>(input), reinterpret_cast<float*>(output),
		numElem, numAnchors, scoreThreshold, netWidth, netHeight, gridSizeX, gridSizeY,
		numClasses, topK, reinterpret_cast<const float*>(anchors),scale);

	return cudaGetLastError();
}