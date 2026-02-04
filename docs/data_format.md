# 数据格式说明

## 目录结构
```
DATA_ROOT/
  samples.jsonl
  cam0_000001.npy
  cam1_000001.npy
  cam2_000001.npy
  cam3_000001.npy
  ...
```

## samples.jsonl
每行一个样本：
```
{"paths": ["cam0_000001.npy", "cam1_000001.npy", "cam2_000001.npy", "cam3_000001.npy"], "action": 5}
```

- `paths`: 对应四路相机的 numpy 文件
- `action`: 离散动作 ID（见 `actions.json`/`actions.yaml`）

## .npy 文件格式
- dtype: float32
- shape: `(frames, 4, H, W)`
  - `frames`: 时序帧数（默认 3）
  - `4`: RGBD 通道（R,G,B,Depth）
  - `H, W`: 高宽（默认 224×224）

## 注意
- 多相机必须同步（或使用上一帧补齐缺帧）
- Depth 建议归一化到 0–1 或以米为单位
