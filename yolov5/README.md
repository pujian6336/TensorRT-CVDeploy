### TensorRT部署精度与速度

| Model  |size<br><sup>(pixes) |  Mode | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | speed<br>(usePlugin)<br>(FPS) |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| YOLOv5n | 640 | PyTorch<br>FP32<br>FP16<br>INT8 | 23.2<br>22.9<br>22.9<br>- | 36.0<br>35.9<br>35.9<br>- |-<br>297<br>397<br>- |
| YOLOv5s | 640 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>216<br>311<br>- |
| YOLOv5m | 640 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>95<br>266<br>- |
| YOLOv5l | 640 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>51<br>163<br>- |
| YOLOv5x | 640 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>-<br>98<br>- |
| | | | | |
| YOLOv5n6 | 1280 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>-<br>-<br>- |
| YOLOv5s6 | 1280 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>-<br>-<br>- |
| YOLOv5m6 | 1280 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>-<br>-<br>- |
| YOLOv5l6 | 1280 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>-<br>-<br>- |
| YOLOv5x6 | 1280 | PyTorch<br>FP32<br>FP16<br>INT8 | -<br>-<br>-<br>- | -<br>-<br>-<br>- | -<br>-<br>-<br>- |
