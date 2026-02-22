# 项目结构与代码职能

## 总览

本项目包含两条推理路径与一套模型转换链路：

- PyTorch 推理：使用 RetinaFace 检测 + FaRL 解析
- TensorFlow/ONNX 推理：使用 ONNXRuntime 或 TensorFlow SavedModel
- 模型转换：PyTorch JIT/权重导出为 ONNX，必要时再转成 TensorFlow SavedModel

## 入口脚本

- test_torch.py：PyTorch 推理入口，加载 RetinaFace 检测与 FaRL 解析，生成结果图
- test_TensorFlow.py：TensorFlow/ONNX 推理入口，支持 ONNXRuntime 与 TF SavedModel

## 模型转换脚本

- convert_torch_onnx.py：将 RetinaFace 与 FaRL JIT 模型导出为 ONNX
- convert_onnx_tf.py：将 ONNX 转换为 TensorFlow SavedModel
- convert_farl_script.py：将 FaRL 模型重建为可脚本化结构并导出 ONNX
- convert_farl_rebuild.py：重建 FaRL 头部结构并与 JIT 权重对齐后导出 ONNX

## 关键模块

- retinaface_detector.py：RetinaFace 模型结构、后处理与检测封装
- farl_face_parser.py：FaRL 解析模型加载、对齐、warp/grid 处理与输出封装
- farl_backbone.py：FaRL 视觉骨干实现与权重映射

## 数据与模型目录约定

- data/input：输入图像
- data/results：输出结果
- data/onnx：ONNX 模型
- data/torch/hub/checkpoints：PyTorch JIT 模型
