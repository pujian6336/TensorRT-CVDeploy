#include "postprocess.h"


void process_decode_ptr_host(std::vector<Detection>& res, const float* decode_ptr_host, int bbox_element, int count, int width, int height, AffineMatrix d2s) {
    for (int i = 0; i < count; i++) {
        const float* ptr = decode_ptr_host + i * bbox_element + 1;
        int keep_flag = ptr[6];
        if (keep_flag) {
            if (bbox_element == 7)
            {
                float left = std::max(0.0f, ptr[0]);
                float top = std::max(0.0f, ptr[1]);
                float right = std::min((float)width - 1, ptr[2]);
                float bottom = std::min((float)height - 1, ptr[3]);

                left = d2s.value[0] * left + d2s.value[1] * top + d2s.value[2];
                top = d2s.value[3] * left + d2s.value[4] * top + d2s.value[5];
                right = d2s.value[0] * right + d2s.value[1] * bottom + d2s.value[2];
                bottom = d2s.value[3] * right + d2s.value[4] * bottom + d2s.value[5];
                res.emplace_back(left, top, right, bottom, ptr[4], (int)ptr[5]);
            }
            else
            {
                float left = std::max(0.0f, ptr[0]);
                float top = std::max(0.0f, ptr[1]);
                float right = std::min((float)width - 1, ptr[2]);
                float bottom = std::min((float)height - 1, ptr[3]);
                float mask[32];
                memcpy(&mask, &ptr[7], 32 * sizeof(float));
                res.emplace_back(left, top, right, bottom, ptr[4], (int)ptr[5], mask);
            }
        }
    }
}

void process(std::vector<Detection>& res, const float* decode_ptr_host, int width, int height, int topK, int bbox_element, AffineMatrix d2s)
{
    int count = static_cast<int>(*decode_ptr_host);
    count = std::min(count, topK);
    process_decode_ptr_host(res, decode_ptr_host, bbox_element, count, width, height, d2s);
}

void batch_process(std::vector<std::vector<Detection>>& res_batch, const float* decode_ptr_host, int batch_size, int width, int height, int topK, int bbox_element, std::vector<AffineMatrix> d2s)
{
    res_batch.resize(batch_size);
    for (int i = 0; i < batch_size; ++i)
    {
        int count = static_cast<int>(*(decode_ptr_host + i * (topK * bbox_element + 1)));
        count = std::min(count, topK);
        process_decode_ptr_host(res_batch[i], &decode_ptr_host[i * (topK * bbox_element + 1)], bbox_element, count, width, height, d2s[i]);
    }
}

static cv::Rect get_downscale_rect(Bbox bbox, float scale) {
    float left = bbox.left / scale;
    float top = bbox.top / scale;
    float right = bbox.right / scale;
    float bottom = bbox.bottom / scale;
    return cv::Rect(round(left), round(top), round(right - left), round(bottom - top));
}

std::vector<cv::Mat> process_masks(const float* proto, int proto_height, int proto_width, std::vector<Detection>& dets, int mask_ratio)
{
    std::vector<cv::Mat> masks;
    for (size_t i = 0; i < dets.size(); i++)
    {
        cv::Mat mask = cv::Mat::zeros(proto_height, proto_width, CV_32FC1);
        auto r = get_downscale_rect(dets[i].bbox, mask_ratio);
        for (int x = r.x; x < r.x + r.width; x++)
        {
            for (int y = r.y; y < r.y + r.height; y++)
            {
                float e = 0.0f;
                for (int j = 0; j < 32; j++)
                {
                    e += dets[i].mask[j] * proto[j * proto_width * proto_height + y * mask.cols + x];
                }
                e = 1.0f / (1.0f + expf(-e));
                mask.at<float>(y, x) = e;
            }
        }
        cv::resize(mask, mask, cv::Size(proto_width * mask_ratio, mask_ratio * proto_height));
        masks.push_back(mask > 0.5f);
    }
    return masks;
}

void box2OriginalSize(std::vector<std::vector<Detection>>& res_batch, std::vector<AffineMatrix> d2s, int width, int height)
{
    for (int i = 0; i < res_batch.size(); ++i)
    {
        for (int j = 0; j < res_batch[i].size(); j++)
        {
            if (width != 0 && height != 0)
            {
                res_batch[i][j].bbox.left = std::max(0.0f, res_batch[i][j].bbox.left);
                res_batch[i][j].bbox.top = std::max(0.0f, res_batch[i][j].bbox.top);
                res_batch[i][j].bbox.right = std::min((float)width - 1, res_batch[i][j].bbox.right);
                res_batch[i][j].bbox.bottom = std::min((float)height - 1, res_batch[i][j].bbox.bottom);
            }

            res_batch[i][j].bbox.left = d2s[i].value[0] * res_batch[i][j].bbox.left + d2s[i].value[1] * res_batch[i][j].bbox.top + d2s[i].value[2];
            res_batch[i][j].bbox.top = d2s[i].value[3] * res_batch[i][j].bbox.left + d2s[i].value[4] * res_batch[i][j].bbox.top + d2s[i].value[5];
            res_batch[i][j].bbox.right = d2s[i].value[0] * res_batch[i][j].bbox.right + d2s[i].value[1] * res_batch[i][j].bbox.bottom + d2s[i].value[2];
            res_batch[i][j].bbox.bottom = d2s[i].value[3] * res_batch[i][j].bbox.right + d2s[i].value[4] * res_batch[i][j].bbox.bottom + d2s[i].value[5];
        }
    }
}

void box2OriginalSize(std::vector<Detection>& res_batch, AffineMatrix d2s, int width, int height)
{
    for (int j = 0; j < res_batch.size(); j++)
    {
        if (width != 0 && height != 0)
        {
            res_batch[j].bbox.left = std::max(0.0f, res_batch[j].bbox.left);
            res_batch[j].bbox.top = std::max(0.0f, res_batch[j].bbox.top);
            res_batch[j].bbox.right = std::min((float)width - 1, res_batch[j].bbox.right);
            res_batch[j].bbox.bottom = std::min((float)height - 1, res_batch[j].bbox.bottom);
        }

        res_batch[j].bbox.left = d2s.value[0] * res_batch[j].bbox.left + d2s.value[1] * res_batch[j].bbox.top + d2s.value[2];
        res_batch[j].bbox.top = d2s.value[3] * res_batch[j].bbox.left + d2s.value[4] * res_batch[j].bbox.top + d2s.value[5];
        res_batch[j].bbox.right = d2s.value[0] * res_batch[j].bbox.right + d2s.value[1] * res_batch[j].bbox.bottom + d2s.value[2];
        res_batch[j].bbox.bottom = d2s.value[3] * res_batch[j].bbox.right + d2s.value[4] * res_batch[j].bbox.bottom + d2s.value[5];
    }
}

void mask2OriginalSize(std::vector<std::vector<cv::Mat>>& masks, std::vector<cv::Mat>& imgs, int width, int height)
{
    for (int i = 0; i < masks.size(); i++)
    {
        int x, y, w, h;
        float r_w = width / (float)imgs[i].cols;
        float r_h = height / (float)imgs[i].rows;
        if (r_h > r_w) {
            w = width;
            h = r_w * imgs[i].rows;
            x = 0;
            y = (height - h) / 2;
        }
        else {
            w = r_h * imgs[i].cols;
            h = height;
            x = (width - w) / 2;
            y = 0;
        }

        cv::Rect r(x, y, w, h);
        cv::Mat res;
        for (int j = 0; j < masks[i].size(); j++)
        {
            cv::resize(masks[i][j](r), masks[i][j], imgs[i].size());
        }
    }
}

void mask2OriginalSize(std::vector<cv::Mat>& masks, cv::Mat& imgs, int width, int height)
{
    int x, y, w, h;
    float r_w = width / (float)imgs.cols;
    float r_h = height / (float)imgs.rows;
    if (r_h > r_w) {
        w = width;
        h = r_w * imgs.rows;
        x = 0;
        y = (height - h) / 2;
    }
    else {
        w = r_h * imgs.cols;
        h = height;
        x = (width - w) / 2;
        y = 0;
    }

    cv::Rect r(x, y, w, h);
    cv::Mat res;
    for (int j = 0; j < masks.size(); j++)
    {
        cv::resize(masks[j](r), masks[j], imgs.size());
    }
}

void yolov8pose::process(std::vector<KeyPointResult>& res, const float* decode_ptr_host, int width, int height, int topK, int bbox_element, AffineMatrix d2s)
{
    int count = static_cast<int>(*decode_ptr_host);
    count = std::min(count, topK);
    int numKeyPoint = (bbox_element - 7) / 3;

    for (int i = 0; i < count; i++) {
        KeyPointResult tempres;
        const float* ptr = decode_ptr_host + i * bbox_element + 1;
        int keep_flag = ptr[6];
        if (keep_flag) {
            float left = std::max(0.0f, ptr[0]);
            float top = std::max(0.0f, ptr[1]);
            float right = std::min((float)width - 1, ptr[2]);
            float bottom = std::min((float)height - 1, ptr[3]);

            tempres.bbox.left = d2s.value[0] * left + d2s.value[1] * top + d2s.value[2];
            tempres.bbox.top = d2s.value[3] * left + d2s.value[4] * top + d2s.value[5];
            tempres.bbox.right = d2s.value[0] * right + d2s.value[1] * bottom + d2s.value[2];
            tempres.bbox.bottom = d2s.value[3] * right + d2s.value[4] * bottom + d2s.value[5];
            tempres.conf = ptr[4];
            
            for (int kc = 0; kc < numKeyPoint; kc++)
            {
                KeyPoint kp;
                kp.id = kc;
                kp.x = d2s.value[0] * ptr[7 + kc * 3] + d2s.value[1] * ptr[8 + kc * 3] + d2s.value[2];
                kp.y = d2s.value[3] * ptr[7 + kc * 3] + d2s.value[4] * ptr[8 + kc * 3] + d2s.value[5];
                kp.score = ptr[9 + kc * 3];
                tempres.keyPoints.push_back(kp);
            }
            res.push_back(tempres);
        }
    }
}

void yolov8pose::process(std::vector<std::vector<KeyPointResult>>& res, const float* decode_ptr_host, int batch_size, int width, int height, int topK, int bbox_element, AffineMatrix d2s)
{
    res.resize(batch_size);
    for (int bi = 0; bi < batch_size; bi++)
    {
        int count = static_cast<int>(*decode_ptr_host);
        count = std::min(count, topK);
        int numKeyPoint = (bbox_element - 7) / 3;

        for (int i = 0; i < count; i++) {
            KeyPointResult tempres;
            const float* ptr = decode_ptr_host + i * bbox_element + 1;
            int keep_flag = ptr[6];
            if (keep_flag) {
                float left = std::max(0.0f, ptr[0]);
                float top = std::max(0.0f, ptr[1]);
                float right = std::min((float)width - 1, ptr[2]);
                float bottom = std::min((float)height - 1, ptr[3]);

                tempres.bbox.left = d2s.value[0] * left + d2s.value[1] * top + d2s.value[2];
                tempres.bbox.top = d2s.value[3] * left + d2s.value[4] * top + d2s.value[5];
                tempres.bbox.right = d2s.value[0] * right + d2s.value[1] * bottom + d2s.value[2];
                tempres.bbox.bottom = d2s.value[3] * right + d2s.value[4] * bottom + d2s.value[5];
                tempres.conf = ptr[4];

                for (int kc = 0; kc < numKeyPoint; kc++)
                {
                    KeyPoint kp;
                    kp.id = kc;
                    kp.x = d2s.value[0] * ptr[7 + kc * 3] + d2s.value[1] * ptr[8 + kc * 3] + d2s.value[2];
                    kp.y = d2s.value[3] * ptr[7 + kc * 3] + d2s.value[4] * ptr[8 + kc * 3] + d2s.value[5];
                    kp.score = ptr[9 + kc * 3];
                    tempres.keyPoints.push_back(kp);
                }
                res[bi].push_back(tempres);
            }
        }
    }
}

void yolov8obb::process(std::vector<OBBResult> &res, const float* decode_ptr_host, int topK, int bbox_element, AffineMatrix d2s)
{
    int count = static_cast<int>(*decode_ptr_host);
    count = std::min(count, topK);

    for (int i = 0; i < count; i++) {
        const float* ptr = decode_ptr_host + i * bbox_element + 1;
        int keep_flag = ptr[7];
        if (keep_flag) {
            float cx = d2s.value[0] * ptr[0] + d2s.value[1] * ptr[1] + d2s.value[2];
            float cy = d2s.value[3] * ptr[0] + d2s.value[4] * ptr[1] + d2s.value[5];
            float w = d2s.value[0] * ptr[2];
            float h = d2s.value[4] * ptr[3];
            res.emplace_back(cx, cy, w, h, ptr[4], ptr[5], (int)ptr[6]);
        }
    }
}

void yolov8obb::process(std::vector<std::vector<OBBResult>>& res, const float* decode_ptr_host, int batch_size, int topK, int bbox_element, AffineMatrix d2s)
{
    res.resize(batch_size);
    for (size_t bi = 0; bi < batch_size; bi++)
    {
        int count = static_cast<int>(*decode_ptr_host + bi * (topK * bbox_element + 1));
        count = std::min(count, topK);

        for (int i = 0; i < count; i++) {
            const float* ptr = decode_ptr_host + i * bbox_element + 1;
            int keep_flag = ptr[7];
            if (keep_flag) {
                float cx = d2s.value[0] * ptr[0] + d2s.value[1] * ptr[1] + d2s.value[2];
                float cy = d2s.value[3] * ptr[0] + d2s.value[4] * ptr[1] + d2s.value[5];
                float w = d2s.value[0] * ptr[2];
                float h = d2s.value[4] * ptr[3];
                res[bi].emplace_back(cx, cy, w, h, ptr[4], ptr[5], (int)ptr[6]);
            }
        }
    }
}
