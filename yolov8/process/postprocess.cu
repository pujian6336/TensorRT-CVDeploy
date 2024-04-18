#include"postprocess.h"
#include <device_launch_parameters.h>

static __global__ void
decodev8_kernel(float* src, float* dst, const uint32_t noElements, const uint32_t lengthPreBatch, const uint32_t numClasses, const int topK,
    float confThres)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= noElements) return;

    uint32_t batchIdx = idx / lengthPreBatch;
    uint32_t curIdx = idx % lengthPreBatch;

    float* pitem = src + batchIdx * lengthPreBatch * (numClasses + 4);

    int classId = 0;
    float score = 0.0;
    for (uint32_t i = 0; i < numClasses; ++i)
    {
        float prob = pitem[(4 + i) * lengthPreBatch + curIdx];
        if (prob > score)
        {
            score = prob;
            classId = i;
        }
    }

    if (score < confThres) return;

    int count = atomicAdd(dst + batchIdx * (topK * 7 + 1), 1);
    if (count >= topK) { return; }

    float cx = pitem[curIdx];
    float cy = pitem[curIdx + lengthPreBatch];
    float width = pitem[curIdx + lengthPreBatch * 2];
    float height = pitem[curIdx + lengthPreBatch * 3];
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
decodev8_seg_kernel(float* src, float* dst, const uint32_t noElements, const uint32_t lengthPreBatch, const uint32_t numClasses, const int topK,
    float confThres)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= noElements) return;

    uint32_t batchIdx = idx / lengthPreBatch;
    uint32_t curIdx = idx % lengthPreBatch;

    float* pitem = src + batchIdx * lengthPreBatch * (numClasses + 4);

    int classId = 0;
    float score = 0.0;
    for (uint32_t i = 0; i < numClasses; ++i)
    {
        float prob = pitem[(4 + i) * lengthPreBatch + curIdx];
        if (prob > score)
        {
            score = prob;
            classId = i;
        }
    }

    if (score < confThres) return;

    int count = atomicAdd(dst + batchIdx * (topK * 39 + 1), 1);
    if (count >= topK) { return; }

    float cx = pitem[curIdx];
    float cy = pitem[curIdx + lengthPreBatch];
    float width = pitem[curIdx + lengthPreBatch * 2];
    float height = pitem[curIdx + lengthPreBatch * 3];
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
    for (int maskindx = 84; maskindx < 116; ++maskindx)
    {
        *pout_item++ = pitem[curIdx + lengthPreBatch * maskindx];
    }
}

static __global__ void
decodev8_pose_kernel(float* src, float* dst, const uint32_t noElements, const uint32_t lengthPreBatch, const uint32_t lengthPreBox, const uint32_t numKeyPoints, const int topK,
    float confThres)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= noElements) return;

    uint32_t batchIdx = idx / lengthPreBatch;
    uint32_t curIdx = idx % lengthPreBatch;

    float* pitem = src + batchIdx * lengthPreBatch * lengthPreBox;
    float confidence = pitem[4 * lengthPreBatch + curIdx];
    if (confidence < confThres) return;

    int count = atomicAdd(dst + batchIdx * (topK * (7 + numKeyPoints * 3) + 1), 1);
    if (count >= topK) { return; }

    float cx = pitem[curIdx];
    float cy = pitem[curIdx + lengthPreBatch];
    float width = pitem[curIdx + lengthPreBatch * 2];
    float height = pitem[curIdx + lengthPreBatch * 3];
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    float* pout_item = dst + batchIdx * (topK * (7 + numKeyPoints * 3) + 1) + 1 + count * (7 + numKeyPoints * 3);
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = 0;
    *pout_item++ = 1;
    for (int maskindx = 5; maskindx < lengthPreBox; ++maskindx)
    {
        *pout_item++ = pitem[curIdx + lengthPreBatch * maskindx];
    }
}

static __global__ void decodev8_obb_kernel(float* src, float* dst,const uint32_t noElements, const uint32_t lengthPreBatch, const uint32_t numClasses, const int topK,
    float confThres)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= noElements) return;

    uint32_t batchIdx = idx / lengthPreBatch;
    uint32_t curIdx = idx % lengthPreBatch;

    float* pitem = src + batchIdx * lengthPreBatch * (numClasses + 5);

    int classId = 0;
    float score = 0.0;
    for (uint32_t i = 0; i < numClasses; ++i)
    {
        float prob = pitem[(4 + i) * lengthPreBatch + curIdx];
        if (prob > score)
        {
            score = prob;
            classId = i;
        }
    }

    if (score < confThres) return;

    int count = atomicAdd(dst + batchIdx * (topK * 8 + 1), 1);
    if (count >= topK) { return; }

    float cx = pitem[curIdx];
    float cy = pitem[curIdx + lengthPreBatch];
    float width = pitem[curIdx + lengthPreBatch * 2];
    float height = pitem[curIdx + lengthPreBatch * 3];
    float angle = pitem[curIdx + lengthPreBatch * 19];
    float* pout_item = dst + batchIdx * (topK * 8 + 1) + 1 + count * 8;
    *pout_item++ = cx;
    *pout_item++ = cy;
    *pout_item++ = width;
    *pout_item++ = height;
    *pout_item++ = angle;
    *pout_item++ = score;
    *pout_item++ = classId;
    *pout_item++ = 1; // 1=keep, 0=ignore;
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

static __device__ void convariance_matrix(float w, float h, float r, float& a, float& b, float& c)
{
    float a_val = w * w / 12.0f;
    float b_val = h * h / 12.0f;
    float cos_r = cosf(r);
    float sin_r = sinf(r);

    a = a_val * cos_r * cos_r + b_val * sin_r * sin_r;
    b = a_val * sin_r * sin_r + b_val * cos_r * cos_r;
    c = (a_val - b_val) * sin_r * cos_r;
}

static __device__ float box_probiou(
    float cx1, float cy1, float w1, float h1, float r1,
    float cx2, float cy2, float w2, float h2, float r2,
    float eps = 1e-7
    )
{
    float a1, b1, c1, a2, b2, c2;
    convariance_matrix(w1, h1, r1, a1, b1, c1);
    convariance_matrix(w2, h2, r2, a2, b2, c2);

    float t1 = ((a1 + a2) * powf(cy1 - cy2, 2) + (b1 + b2) * powf(cx1 - cx2, 2)) / ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
    float t2 = ((c1 + c2) * (cx2 - cx1) * (cy1 - cy2)) / ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
    float t3 = logf(((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2)) / (4 * sqrtf(fmaxf(a1 * b1 - c1 * c1, 0.0f)) * sqrtf(fmaxf(a2 * b2 - c2 * c2, 0.0f)) + eps) + eps);
    float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
    bd = fmaxf(fminf(bd, 100.0f), eps);
    float hd = sqrtf(1.0f - expf(-bd) + eps);
    return 1 - hd;
}

static __global__ void obb_nms_kernel(float* bboxes, int max_objects, int lengthPreObject, float threshold)
{
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, max_objects);

    if (position >= count) return;

    float* pcurrent = bboxes + 1 + position * lengthPreObject;
    for (int i = 0; i < count; ++i) {
        float* pitem = bboxes + 1 + i * lengthPreObject;
        if (i == position || pcurrent[6] != pitem[6]) continue;
        if (pitem[5] >= pcurrent[5]) {
            if (pitem[5] == pcurrent[5] && i < position) continue;
            float iou = box_probiou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pcurrent[4],
                pitem[0], pitem[1], pitem[2], pitem[3], pitem[4]
            );
            if (iou > threshold) {
                pcurrent[7] = 0;
                return;
            }
        }
    }
}

void cuda_decode(float* src, float* dst, int batchSize, uint32_t lengthPreBatch, const uint32_t numClasses, int topK, float confThres, cudaStream_t stream)
{
    int threadPreBlock = 256;
    int numElem = batchSize * lengthPreBatch;
    int blockPreGrid = ceil(numElem / float(threadPreBlock));
    decodev8_kernel << < blockPreGrid, threadPreBlock, 0, stream >> > (src, dst, numElem, lengthPreBatch, numClasses, topK, confThres);
}

void cuda_decodeSeg(float* src, float* dst, int batchSize, uint32_t lengthPreBatch, uint32_t numClasses, int topK, float confThres, cudaStream_t stream)
{
    int threadPreBlock = 256;
    int numElem = batchSize * lengthPreBatch;
    int blockPreGrid = ceil(numElem / float(threadPreBlock));
    decodev8_seg_kernel << < blockPreGrid, threadPreBlock, 0, stream >> > (src, dst, numElem, lengthPreBatch, numClasses, topK, confThres);
}

void cuda_decodePose(float* src, float* dst, int batchSize, uint32_t lengthPreBatch, uint32_t lengthPreBox, uint32_t numKeyPoints, int topK, float confThres, cudaStream_t stream)
{
    int threadPreBlock = 256;
    int numElem = batchSize * lengthPreBatch;
    int blockPreGrid = ceil(numElem / float(threadPreBlock));
    decodev8_pose_kernel << <blockPreGrid, threadPreBlock, 0, stream >> > (src, dst, numElem, lengthPreBatch, lengthPreBox, numKeyPoints, topK, confThres);
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
        nms_kernel << <grid, block, 0, stream >> > (src + i * (max_objects * lengthPreObject + 1), max_objects, lengthPreObject, nms_threshold);
    }
}

void yolov8obb::cuda_decodeObb(float* src, float* dst, int batchSize, uint32_t lengthPreBatch, const uint32_t numClasses, int topK, float confThres, cudaStream_t stream)
{
    int threadPreBlock = 256;
    int numElem = batchSize * lengthPreBatch;
    int blockPreGrid = ceil(numElem / float(threadPreBlock));
    decodev8_obb_kernel << < blockPreGrid, threadPreBlock, 0, stream >> > (src, dst, numElem, lengthPreBatch, numClasses, topK, confThres);
}

void yolov8obb::cuda_nms(float* parray, float nms_threshold, int max_objects, int lengthPreObject, cudaStream_t stream)
{
    int block = max_objects < 256 ? max_objects : 256;
    int grid = ceil(max_objects / (float)block);
    obb_nms_kernel << <grid, block, 0, stream >> > (parray, max_objects, lengthPreObject, nms_threshold);
}

void yolov8obb::cuda_nms(float* parray, int batchSize, float nms_threshold, int max_objects, int lengthPreObject, cudaStream_t stream)
{
    int block = max_objects < 256 ? max_objects : 256;
    int grid = ceil(max_objects / (float)block);
    for (int i = 0; i < batchSize; i++)
    {
        obb_nms_kernel << <grid, block, 0, stream >> > (parray + i * (max_objects * lengthPreObject + 1), max_objects, lengthPreObject, nms_threshold);
    }
}

