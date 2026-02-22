# Facial Mask Segmentation

基于 RetinaFace 人脸检测与 FaRL 人脸解析的分割项目，支持 PyTorch 与 ONNX/TensorFlow 推理，并提供一键导出 ONNX 的脚本。

## 功能概览

- PyTorch 推理：RetinaFace 检测 + FaRL 解析
- ONNX/TensorFlow 推理：ONNX Runtime 或 TF SavedModel
- 模型转换：JIT/权重导出为 ONNX

## 目录结构

- tools/convert：模型转换脚本
- data/input：输入图片目录
- data/results：输出结果目录
- data/torch/hub/checkpoints：PyTorch 模型目录
- data/onnx：ONNX 模型目录

## 依赖安装

```bash
pip install -r requirements.txt
```

## PyTorch 推理

```bash
python test_torch.py
```

模型会在首次运行时自动下载到 data/torch/hub/checkpoints。

## ONNX/TensorFlow 推理

```bash
python test_TensorFlow.py
```

默认会下载并使用 FaRL CelebM ONNX 模型。

## 模型下载与来源

- FaRL CelebM JIT  
  https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.celebm.main_ema_181500_jit.pt
- FaRL LaPa JIT  
  https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.lapa.main_ema_136500_jit191.pt
- RetinaFace MobileNet 权重  
  由 retinaface_detector.py 中的 pretrained_urls 提供

## 导出 ONNX

```bash
python tools/convert/convert_torch_onnx.py
```

支持交互式选择模型，也可通过参数跳过交互：

```bash
python tools/convert/convert_torch_onnx.py --with_retinaface --skip_farl_lapa
```

## FaRL 输出说明

FaRL ONNX 输出为 512×512，推理端需先 resize 到 448×448 再进行反变换。




## 许可证

MIT License
