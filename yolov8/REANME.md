用法：

+ 序列化模型文件

```c++
#include <builder/trt_builder.h>

int main()
{
	buildConf cfg;

	cfg.onnx_model_path = "./weights/yolov8n.onnx";
	cfg.trt_model_save_path = "./weights/yolov8n.trt";

	cfg.mode = Mode::FP32; // FP32模式
    cfg.mode = Mode::FP16; // FP16模式

	buildEngine(cfg);
}


```

+ 目标检测
```c++
#include "engine/yolov8.h"
#include "utils/utils.h"

void main()
{
	Config cfg;
	cfg.model_path = "./weights/yolov8n_fp32.cfm";
	YOLO yolov8(cfg);

	yolov8.init();

	cv::Mat img = cv::imread("./assets/zidane.jpg");
	std::vector<Detection> res;
	yolov8.Run(img, res);

	utils::DrawDetection(img, res, utils::dataSets::coco80);

	cv::imshow("img", img);
	cv::waitKey(0);
}
```

+ 实例分割
```c++
#include "engine/yolov8_seg.h"
#include "utils/utils.h"

void main()
{
	Config cfg;
	cfg.model_path = "./weights_seg/yolov8n-seg_fp32.cfm";
	YOLO_SEG yolov8(cfg);

	yolov8.init();

	cv::Mat img = cv::imread("./assets/zidane.jpg");
	std::vector<Detection> res;
	std::vector<cv::Mat> masks;
	yolov8.Run(img, res, masks);

	utils::DrawSegmentation(img, res, masks, utils::dataSets::coco80);

	cv::imshow("img", img);
	cv::waitKey(0);
}
```

+ 关键点检测
```c++
#include "engine/yolov8_pose.h"
#include "utils/utils.h"

void main()
{
	Config cfg;
	cfg.model_path = "./weights_pose/yolov8n-pose_fp32.cfm";
	YOLO_POSE yolov8(cfg);

	yolov8.init();

	cv::Mat img = cv::imread("./assets/bus.jpg");
	std::vector<KeyPointResult> res;
	yolov8.Run(img, res);

	utils::DrawKeyPoints(img, res, "person");

	cv::imshow("img", img);
	cv::waitKey(0);
}
```
+ 旋转目标检测
```c++
#include "engine/yolov8_obb.h"
#include "utils/utils.h"

void main()
{
	Config cfg;
	cfg.input_width = 1024;
	cfg.input_height = 1024;
	cfg.model_path = "./weights_obb/yolov8n-obb_fp32.cfm";
	YOLO_OBB yolov8(cfg);

	yolov8.init();

	cv::Mat img = cv::imread("./assets/P0168.png");
		
	std::vector<OBBResult> res;
	yolov8.Run(img, res);

	utils::DrawOBB(img, res, utils::dataSets::dotav1);

	cv::imshow("img", img);
	cv::waitKey(0);
}
```