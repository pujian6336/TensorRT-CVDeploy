# 用法

+ 序列化模型文件

```c++
#include <builder/trt_builder.h>

int main()
{
	buildConf cfg;

	cfg.onnx_model_path = "./weights/yolov5n.onnx";
	cfg.trt_model_save_path = "./weights/yolov5n.trt";

	cfg.mode = Mode::FP32; // FP32模式
    cfg.mode = Mode::FP16; // FP16模式

	buildEngine(cfg);
}
```

+ 目标检测

```c++
#include "engine/yolov5.h"
#include "utils/utils.h"

void main()
{
	Config cfg;
	cfg.model_path = "./weights/yolov5s_fp32.cfm";
	YOLO yolov5(cfg);

	yolov5.init();

	cv::Mat img = cv::imread("./assets/bus.jpg");

	std::vector<Detection> res;
	yolov5.Run(img, res);

	utils::DrawDetection(img, res, utils::dataSets::coco80);

	cv::imshow("img", img);
	cv::waitKey(0);
}
```

+ 实例分割
```c++
#include "engine/yolov5.h"
#include "utils/utils.h"

void main()
{
	#include "engine/yolov5_seg.h"
#include "utils/utils.h"

void main()
{
	Config cfg;
	cfg.model_path = "./weights_seg/yolov5n-seg_fp32.cfm";
	YOLO_SEG yolov5(cfg);

	yolov5.init();

	cv::Mat img = cv::imread("./assets/bus.jpg");

	std::vector<Detection> res;
	std::vector<cv::Mat> masks;
	yolov5.Run(img, res, masks);

	utils::DrawSegmentation(img, res, masks, utils::dataSets::coco80);

	cv::imshow("img", img);
	cv::waitKey(0);
}
}
```


### TensorRT部署精度与速度

| Model  |size<br><sup>(pixes) |  Mode | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | speed<br>(usePlugin)<br>(FPS) | speed<br>(FPS) |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| YOLOv5n | 640 | PyTorch<br>FP32<br>FP16<br>INT8 | 23.2<br>22.9<br>22.9<br>- | 36.0<br>35.9<br>35.9<br>- |-<br>302<br>395<br>- | -<br>317<br>340<br>- |
| YOLOv5s | 640 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>216<br>311<br>- | -<br>216<br>326<br>-|
| YOLOv5m | 640 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>95<br>266<br>- | -<br>101<br>261<br>- |
| YOLOv5l | 640 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>51<br>163<br>- | -<br>51<br>163<br>-
| YOLOv5x | 640 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>26<br>98<br>- | -<br>27<br>99<br>-
| | | | | |
| YOLOv5n6 | 1280 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>-<br>-<br>- |
| YOLOv5s6 | 1280 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>-<br>-<br>- |
| YOLOv5m6 | 1280 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>-<br>-<br>- |
| YOLOv5l6 | 1280 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>-<br>-<br>- |
| YOLOv5x6 | 1280 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>-<br>-<br>- |
