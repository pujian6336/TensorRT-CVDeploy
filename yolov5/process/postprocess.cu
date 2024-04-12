#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#include "postprocess.h"

static __global__ void 
decode_kernel(float* src, float* dst, const uint32_t noElements, const uint32_t lengthPreBatch, const uint32_t numClasses, const int topK,
    float confThres)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= noElements) return;

    uint32_t batchIdx = idx / lengthPreBatch;
    uint32_t curIdx = idx % lengthPreBatch;
    float* pitem = src + batchIdx * lengthPreBatch * (numClasses + 5) + curIdx * (numClasses + 5);

    float conf = pitem[4];
    if (conf < confThres) return;

    int classId = 0;
    float maxProb = 0.0;
    for (uint32_t i = 0; i < numClasses; ++i)
    {
        float prob = pitem[i + 5];
        if (prob > maxProb)
        {
            maxProb = prob;
            classId = i;
        }
    }

    float score = conf * maxProb;
    if (score < confThres) return;

    int count = atomicAdd(dst + batchIdx * (topK * 7 + 1), 1);
    if (count >= topK) { return; }

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    float* pout_item = dst + batchIdx * (topK * 7 + 1) + 1 + count * 7;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = score;
    *pout_item++ = classId;
    *pout_item++ = 1;
}

static __global__ void
decode_seg_kernel(float* src, float* dst, const uint32_t noElements, const uint32_t lengthPreBatch, const uint32_t lengthPreBox, const uint32_t numClasses, const int topK,
    float confThres)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= noElements) return;

    uint32_t batchIdx = idx / lengthPreBatch;
    uint32_t curIdx = idx % lengthPreBatch;
    float* pitem = src + batchIdx * lengthPreBatch * lengthPreBox + curIdx * lengthPreBox;

    float conf = pitem[4];
    if (conf < confThres) return;

    int classId = 0;
    float maxProb = 0.0;
    for (uint32_t i = 0; i < numClasses; ++i)
    {
        float prob = pitem[i + 5];
        if (prob > maxProb)
        {
            maxProb = prob;
            classId = i;
        }
    }

    float score = conf * maxProb;
    if (score < confThres) return;

    int count = atomicAdd(dst + batchIdx * (topK * 39 + 1), 1);
    if (count >= topK) { return; }

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    float* pout_item = dst + batchIdx * (topK * 39 + 1) + 1 + count * 39;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = score;
    *pout_item++ = classId;
    *pout_item++ = 1;
    memcpy(pout_item, pitem + numClasses + 1, 32 * sizeof(float));
}


static __device__ float
box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop, float bright, float bbottom) {
    float cleft = fmaxf(aleft, bleft);
    float ctop = fmaxf(atop, btop);
    float cright = fminf(aright, bright);
    float cbottom = fminf(abottom, bbottom);
    float c_area = fmaxf(cright - cleft, 0.0f) * fmaxf(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) return 0.0f;

    float a_area = fmaxf(0.0f, aright - aleft) * fmaxf(0.0f, abottom - atop);
    float b_area = fmaxf(0.0f, bright - bleft) * fmaxf(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void nms_kernel(float* bboxes, int max_objects, int lengthPreObject, float threshold) {
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = bboxes[0];
    count = min(count, max_objects);

    if (position >= count) return;

    float* pcurrent = bboxes + 1 + position * lengthPreObject;
    for (int i = 0; i < count; ++i) {
        float* pitem = bboxes + 1 + i * lengthPreObject;
        if (i == position || pcurrent[5] != pitem[5]) continue;
        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position) continue;
            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0], pitem[1], pitem[2], pitem[3]
            );
            if (iou > threshold) {
                pcurrent[6] = 0;
                return;
            }
        }
    }
}

void cuda_decode(float *src, float *dst, int batchSize, uint32_t lengthPreBatch, const uint32_t numClasses, int topK, float confThres, cudaStream_t stream)
{
    int threadPreBlock = 256;
    int numElem = batchSize * lengthPreBatch;
    int blockPreGrid = ceil(numElem / float(threadPreBlock));
    decode_kernel << < blockPreGrid, threadPreBlock, 0, stream >> > (src, dst, numElem, lengthPreBatch, numClasses, topK, confThres);
}

void cuda_decodeSeg(float* src, float* dst, int batchSize, uint32_t lengthPreBatch, uint32_t lengthPreBox, uint32_t numClasses, int topK, float confThres, cudaStream_t stream)
{
    int threadPreBlock = 256;
    int numElem = batchSize * lengthPreBatch;
    int blockPreGrid = ceil(numElem / float(threadPreBlock));
    decode_seg_kernel << < blockPreGrid, threadPreBlock, 0, stream >> > (src, dst, numElem, lengthPreBatch, lengthPreBox, numClasses, topK, confThres);
}

void cuda_nms(float* src, float nms_threshold, int max_objects, int lengthPreObject, cudaStream_t stream) {
    int block = max_objects < 256 ? max_objects : 256;
    int grid = ceil(max_objects / (float)block);
    nms_kernel << <grid, block, 0, stream >> > (src, max_objects, lengthPreObject, nms_threshold);
}

void cuda_nms_batch(float* src, int batchSize, float nms_threshold, int max_objects, int lengthPreObject, cudaStream_t stream)
{
    int block = max_objects < 256 ? max_objects : 256;
    int grid = ceil(max_objects / (float)block);
    for (int i = 0; i < batchSize; i++)
    {
        nms_kernel << <grid, block, 0, stream >> > (src+i*7001, max_objects, lengthPreObject, nms_threshold);
    }
}
