#include "postprocess.h"

static bool cmp(const Detection& a, const Detection& b) {
    return a.conf > b.conf;
}

static float iou(Bbox lbox, Bbox& rbox) {
  Bbox interBox;
  interBox.left = (std::max)(lbox.left, rbox.left);
  interBox.top = (std::max)(lbox.top, rbox.top);
  interBox.right = (std::min)(lbox.right, rbox.right);
  interBox.bottom = (std::min)(lbox.bottom, rbox.bottom);

  if (interBox.left > interBox.right || interBox.top > interBox.bottom)
    return 0.0f;

  float interBoxArea =
      (interBox.right - interBox.left) * (interBox.bottom - interBox.top);

  float unionBoxArea = (lbox.right- lbox.left) * (lbox.bottom - lbox.top) +
                       (rbox.right- rbox.left) * (rbox.bottom - rbox.top) -
                       interBoxArea;

  return interBoxArea / unionBoxArea;
}

static float iou_actual(Bbox lbox, Bbox& rbox) {
  Bbox interBox;
  interBox.left = (std::max)(lbox.left, rbox.left);
  interBox.top = (std::max)(lbox.top, rbox.top);
  interBox.right = (std::min)(lbox.right, rbox.right);
  interBox.bottom = (std::min)(lbox.bottom, rbox.bottom);

  if (interBox.left > interBox.right || interBox.top > interBox.bottom)
    return 0.0f;

  // (2,10)实际长度是9，而不是10-2=8
  float interBoxArea = (interBox.right - interBox.left + 1.0f) *
                       (interBox.bottom - interBox.top + 1.0f);

  float unionBoxArea =
      (lbox.left - lbox.right + 1.0f) * (lbox.bottom - lbox.top + 1.0f) +
      (rbox.left - rbox.right + 1.0f) * (rbox.bottom - rbox.top + 1.0f) -
      interBoxArea;

  return interBoxArea / unionBoxArea;
}

void nms(std::vector<Detection>& res, float* output, float conf_thresh,
    float nms_thresh, int max_det, int length_pre_object = 7) {
    std::map<float, std::vector<Detection>> m;

    int count = std::min((int)output[0], max_det);

    for (int i = 0; i < count; i++)
    {
        if (output[1 + length_pre_object * i + 4] < conf_thresh) continue;
        Detection det;
        memcpy(&det, &output[1 + length_pre_object * i], 5 * sizeof(float));
        det.class_id = (int)output[6 + length_pre_object * i];
        if (m.count(det.class_id) == 0)m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }

    for (auto it = m.begin(); it != m.end(); it++)
    {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m)
        {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n)
            {
                if (iou(item.bbox, dets[n].bbox) >= nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

void batch_nms(std::vector<std::vector<Detection>>& batch_res, float* output, int batch_size, int output_size, float conf_thres, float nms_thresh,int max_det)
{
    batch_res.resize(batch_size);
    for (int i = 0; i < batch_res.size(); i++)
    {
        nms(batch_res[i], &output[i * output_size], conf_thres, nms_thresh, max_det);
    }
}

// 处理cuda_nms的结果
void process_decode_ptr_host(std::vector<Detection>& res, const float* decode_ptr_host, int bbox_element,int count, AffineMatrix d2s) {
    for (int i = 0; i < count; i++) {
        const float* ptr = decode_ptr_host + i * bbox_element + 1;
        int keep_flag = ptr[6];
        if (keep_flag) {
            float left = d2s.value[0] * ptr[0] + d2s.value[1] * ptr[1] + d2s.value[2];
            float top = d2s.value[3] * ptr[0] + d2s.value[4] * ptr[1] + d2s.value[5];
            float right = d2s.value[0] * ptr[2] + d2s.value[1] * ptr[3] + d2s.value[2];
            float bottom = d2s.value[3] * ptr[2] + d2s.value[4] * ptr[3] + d2s.value[5];

            if (bbox_element == 7)
            {
                res.emplace_back(left, top, right, bottom, ptr[4], (int)ptr[5]);
            }
            else
            {
                float mask[32];
                memcpy(&mask, &ptr[7], 32 * sizeof(float));
                res.emplace_back(left, top, right, bottom, ptr[4], (int)ptr[5], mask);
            }
        }
    }
}

void batch_process(std::vector<std::vector<Detection>>& res_batch, const float* decode_ptr_host, int batch_size, int topK, int bbox_element, std::vector<AffineMatrix> d2s)
{
    res_batch.resize(batch_size);
    for (int i = 0; i < batch_size; ++i)
    {
        int count = static_cast<int>(*(decode_ptr_host+i * (topK * bbox_element + 1)));
        count = std::min(count, topK);
        process_decode_ptr_host(res_batch[i], &decode_ptr_host[i * (topK * bbox_element + 1)], bbox_element, count, d2s[i]);
    }
}

void batch_process(std::vector<std::vector<Detection>>& res_batch, std::vector<AffineMatrix> d2s, int width, int height)
{
    for (int i = 0; i < res_batch.size(); ++i)
    {
        for (int j = 0; j < res_batch[i].size(); j++)
        {
            res_batch[i][j].bbox.left = std::max(0.0f, res_batch[i][j].bbox.left);
            res_batch[i][j].bbox.top = std::max(0.0f, res_batch[i][j].bbox.top);
            res_batch[i][j].bbox.right = std::min((float)width, res_batch[i][j].bbox.right);
            res_batch[i][j].bbox.bottom = std::min((float)height, res_batch[i][j].bbox.bottom);

            res_batch[i][j].bbox.left = d2s[i].value[0] * res_batch[i][j].bbox.left + d2s[i].value[1] * res_batch[i][j].bbox.top + d2s[i].value[2];
            res_batch[i][j].bbox.top = d2s[i].value[3] * res_batch[i][j].bbox.left + d2s[i].value[4] * res_batch[i][j].bbox.top + d2s[i].value[5];
            res_batch[i][j].bbox.right = d2s[i].value[0] * res_batch[i][j].bbox.right + d2s[i].value[1] * res_batch[i][j].bbox.bottom + d2s[i].value[2];
            res_batch[i][j].bbox.bottom = d2s[i].value[3] * res_batch[i][j].bbox.right + d2s[i].value[4] * res_batch[i][j].bbox.bottom + d2s[i].value[5];
        }
    }
}

void process(std::vector<Detection>& res, const float* decode_ptr_host, int topK, int bbox_element, AffineMatrix d2s)
{
    int count = static_cast<int>(*decode_ptr_host);
    count = std::min(count, topK);
    process_decode_ptr_host(res, decode_ptr_host, bbox_element, count, d2s);
    
}
