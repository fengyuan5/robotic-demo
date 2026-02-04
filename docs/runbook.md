# 运行指南

## 训练流程
1) 准备数据集（仿真或真实）
2) 运行训练：
```
python3 -m src.train --data DATA_ROOT
```
3) 查看 checkpoints 输出

## 推理部署（离线）
1) 导出 ONNX：
```
python3 -m src.export_onnx --ckpt checkpoints/epoch_020.pt --out policy.onnx
```
2) 使用 TensorRT/ONNX Runtime 加载

## 常见问题
- `torch`/`numpy` 缺失：先 `pip install -r requirements.txt`
- 测试跳过：没有安装相关依赖
