#pragma once
#include <algorithm>

struct Bbox {
  float left, top, right, bottom;
  Bbox() = default;
  Bbox(float left, float top, float right, float bottom) : left(left), top(top), right(right), bottom(bottom) {};
};

struct Detection {
  Bbox bbox;
  float conf;
  int class_id;
  float mask[32]{ 0 };

  Detection() = default;
  Detection(float left, float top, float right, float bottom, float conf, int label) : bbox(left,top,right,bottom),
	  conf(conf), class_id(label) {}

  Detection(float left, float top, float right, float bottom, float conf, int label, const float init_mask[32])
      : bbox(left, top, right, bottom),
      conf(conf),
      class_id(label) {
      std::copy_n(init_mask, 32, mask);
  }
};

struct AffineMatrix {
  float value[6];
};

enum class Mode : int
{
    FP32,
    FP16,
    INT8
};
