# 完整教程（中文）

本教程覆盖：数据准备（仿真/真实）→ 训练 → 评估 → 导出 → 部署对接要点。

## 0. 前置依赖
```
pip install -r requirements.txt
```

## 1. 数据准备
### 1.1 数据结构
参见 `docs/data_format.md`。核心是：
- `samples.jsonl`（每行一个样本）
- 4 路相机 `.npy`（shape: frames×4×H×W）

### 1.2 快速生成样例
```
python3 scripts/make_sample_data.py --out examples/sample_data
```

### 1.3 仿真采集接入
使用 `scripts/sim_collect_stub.py` 作为模板，对接你的仿真环境：
- `reset_env()`
- `step_env(action_id)`
- `get_multicam_rgbd()`

建议在仿真中做强随机化：
- 桌面高度（低/中/高 3 档）
- 光照/材质/纹理/遮挡
- 多品类尺寸与形状

### 1.4 真实示范数据
- 用遥操作/示范采集少量轨迹
- 覆盖难例（透明/反光/小物体）
- 量级建议：500–1000 轨迹

## 2. 训练
```
python3 -m src.train --data DATA_ROOT
```
- 默认配置来自 `config.yaml`
- 模型与动作空间定义在 `src/model.py` 和 `src/actions.py`
- 训练输出：`checkpoints/epoch_XXX.pt`

## 3. 评估（离线）
当前为最小骨架，可以按以下方式补充：
- 记录训练 loss 变化趋势
- 加一个验证集，输出 Top-1 准确率
- 统计动作分布（避免单一动作塌缩）

建议在后续增加：
- `scripts/eval.py`（读取模型+数据集）
- 输出成功率/平均步数等

## 4. 导出 ONNX
```
python3 -m src.export_onnx --ckpt checkpoints/epoch_020.pt --out policy.onnx
```

## 5. 部署要点（概念级）
### 5.1 推理频率
- 控制周期：100ms
- 建议推理 < 30ms

### 5.2 多相机同步
- 每个控制周期同步 4 路 RGB-D
- 缺帧用上一帧补齐

### 5.3 动作执行
- 动作 ID → 控制指令映射
- 位移/角度步长由 `config.yaml` 控制

## 6. 常见问题
- `numpy/torch` 未安装：运行 `pip install -r requirements.txt`
- 测试跳过：依赖未安装
- 训练输出为空：检查数据路径与 `samples.jsonl`

## 7. 下一步建议
- 增加验证集与评估指标
- 增加日志系统（loss/acc/动作分布）
- 补充仿真引擎适配层
