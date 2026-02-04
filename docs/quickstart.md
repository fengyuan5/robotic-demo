# 快速开始

## 1) 安装依赖
```
pip install -r requirements.txt
```

## 2) 生成样例数据
```
python3 scripts/make_sample_data.py --out examples/sample_data
```

## 3) 训练
```
python3 -m src.train --data examples/sample_data
```

## 4) 导出 ONNX
```
python3 -m src.export_onnx --ckpt checkpoints/epoch_020.pt --out policy.onnx
```

## 5) 跑测试
```
python3 -m unittest discover -s tests -p "test_*.py"
```

## 6) 完整教程
- `docs/tutorial_zh.md`
