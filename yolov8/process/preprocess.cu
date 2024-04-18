#include <device_launch_parameters.h>

#include "preprocess.h"
#include "utils/utils.h"

static uint8_t* img_buffer_host = nullptr;
static uint8_t* img_buffer_device = nullptr;

// 一个线程处理一个像素点
__global__ void preprocess_kernel(uint8_t* src, int src_width, int src_height,
                                  float* dst, int dst_width, int dst_height,
                                  uint8_t padding_value, AffineMatrix d2s,
                                  float scale, float mean0, float mean1,
                                  float mean2, float std0, float std1,
                                  float std2, int edge) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= edge) return;

  // 从d2s中读取变换矩阵
  float m_x1 = d2s.value[0];
  float m_y1 = d2s.value[1];
  float m_z1 = d2s.value[2];
  float m_x2 = d2s.value[3];
  float m_y2 = d2s.value[4];
  float m_z2 = d2s.value[5];

  int dx = position % dst_width;  // 计算当前线程对应的目标图像x坐标
  int dy = position / dst_width;  // 计算当前线程对应的目标图像y指标
  float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
  float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
  float c0, c1, c2;

  if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
    c0 = padding_value;
    c1 = padding_value;
    c2 = padding_value;
  } else {
    // 双线性插值，实现图像放大缩小
    // floorf返回不大于参数的最大整数值(float类型),获取最近的四个点坐标
    int x_low = floorf(src_x);
    int y_low = floorf(src_y);
    int x_high = x_low + 1;
    int y_high = y_low + 1;

    uint8_t const_value[] = {padding_value, padding_value, padding_value};
    float lx = src_x - x_low;
    float ly = src_y - y_low;
    float hx = 1 - lx;
    float hy = 1 - ly;
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    uint8_t* v1 = const_value;
    uint8_t* v2 = const_value;
    uint8_t* v3 = const_value;
    uint8_t* v4 = const_value;

    if (y_low >= 0) {
      if (x_low >= 0) {
        v1 = src + y_low * src_width * 3 + x_low * 3;
      }

      if (x_high < src_width) {
        v2 = src + y_low * src_width * 3 + x_high * 3;
      }
    }

    if (y_high < src_height) {
      if (x_low >= 0) {
        v3 = src + y_high * src_width * 3 + x_low * 3;
      }

      if (x_high < src_width) {
        v4 = src + y_high * src_width * 3 + x_high * 3;
      }
    }

    c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
    c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
    c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
  }

  // bgr to rgb
  float t = c2;
  c2 = c0;
  c0 = t;

  // normalization
  c0 = (c0 / scale - mean0) / std0;
  c1 = (c1 / scale - mean1) / std1;
  c2 = (c2 / scale - mean2) / std2;

  int area = dst_width * dst_height;
  float* pdst_c0 = dst + dy * dst_width + dx;
  float* pdst_c1 = pdst_c0 + area;
  float* pdst_c2 = pdst_c1 + area;
  *pdst_c0 = c0;
  *pdst_c1 = c1;
  *pdst_c2 = c2;
}

void cuda_preprocess(const cv::Mat& srcImg, float* input_device_memory,
                     Config cfg, AffineMatrix& d2s,
                     cudaStream_t stream) {
  int src_height = srcImg.rows;
  int src_width = srcImg.cols;
  int img_size = src_height * src_width * 3;

  // copy data to pinned memory
  memcpy(img_buffer_host, srcImg.ptr(), img_size);
  // copy data to device memory
  CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, img_size,
                             cudaMemcpyHostToDevice, stream));

  // 计算伪射变换矩阵
  AffineMatrix s2d;
  float scale = std::min(cfg.input_height / (float)src_height,
                         cfg.input_width / (float)src_width);

  s2d.value[0] = scale;
  s2d.value[1] = 0;
  s2d.value[2] = -scale * src_width * 0.5 + cfg.input_width * 0.5;
  s2d.value[3] = 0;
  s2d.value[4] = scale;
  s2d.value[5] = -scale * src_height * 0.5 + cfg.input_height * 0.5;

  cv::Mat src2dst_mat(2, 3, CV_32F, s2d.value);
  cv::Mat dst2src_mat(2, 3, CV_32F, d2s.value);
  cv::invertAffineTransform(src2dst_mat, dst2src_mat);

  memcpy(d2s.value, dst2src_mat.ptr<float>(0), sizeof(d2s.value));

  // 一个线程处理一个像素点，一共需要dst_height*dst_width个线程
  int jobs = cfg.input_height * cfg.input_width;
  int threads = 256;
  int blocks = ceil(jobs / (float)threads);

  preprocess_kernel << <blocks, threads, 0, stream >> > (
      img_buffer_device, src_width, src_height, input_device_memory,
      cfg.input_width, cfg.input_height, 127, d2s, cfg.scale, cfg.means[0],
      cfg.means[1], cfg.means[2], cfg.std[0], cfg.std[1], cfg.std[2], jobs);
}

void cuda_preprocess(const cv::Mat& srcImg, float* dst, buildConf cfg)
{
    int src_height = srcImg.rows;
    int src_width = srcImg.cols;
    int img_size = src_height * src_width * 3;

    // copy data to device memory
    CUDA_CHECK(cudaMemcpy(img_buffer_device, srcImg.ptr(), img_size, cudaMemcpyHostToDevice));

    // 计算伪射变换矩阵
    AffineMatrix s2d, d2s;
    float scale = std::min(cfg.input_height / (float)src_height,
        cfg.input_width / (float)src_width);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width * 0.5 + cfg.input_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + cfg.input_height * 0.5;

    cv::Mat src2dst_mat(2, 3, CV_32F, s2d.value);
    cv::Mat dst2src_mat(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(src2dst_mat, dst2src_mat);

    memcpy(d2s.value, dst2src_mat.ptr<float>(0), sizeof(d2s.value));

    // 一个线程处理一个像素点，一共需要dst_height*dst_width个线程
    int jobs = cfg.input_height * cfg.input_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);

    preprocess_kernel << <blocks, threads, 0, NULL >> > (
        img_buffer_device, src_width, src_height, dst,
        cfg.input_width, cfg.input_height, 127, d2s, 255.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 1.0f, 1.0f, jobs);
}

void cuda_batch_preprocess(const std::vector<cv::Mat>& img_batch, float* dst,
                           Config cfg, std::vector<AffineMatrix>& vd2s,
                           cudaStream_t stream) {
  int dst_size = cfg.input_height * cfg.input_width * 3;
  int src_size = cfg.src_height * cfg.src_width * 3;
  for (size_t i = 0; i < img_batch.size(); i++) {
    AffineMatrix d2s{0};
    cuda_preprocess(img_batch[i], &dst[dst_size * i], cfg, d2s, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    vd2s.emplace_back(d2s);
  }
}

void cuda_preprocess_init(int max_image_size) {
  // prepare input data in pinned memory
  CUDA_CHECK(cudaMallocHost((void**)&img_buffer_host, max_image_size * 3));
  // prepare input data in device memory
  CUDA_CHECK(cudaMalloc((void**)&img_buffer_device, max_image_size * 3));
}

void cuda_preprocess_destroy() {
  CUDA_CHECK(cudaFree(img_buffer_device));
  CUDA_CHECK(cudaFreeHost(img_buffer_host));
}
