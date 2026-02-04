# Training Skeleton

## Expected Dataset Layout
```
DATA_ROOT/
  samples.jsonl
  cam0_000001.npy
  cam1_000001.npy
  cam2_000001.npy
  cam3_000001.npy
  ...
```

Each line in `samples.jsonl`:
```
{"paths": ["cam0_000001.npy", "cam1_000001.npy", "cam2_000001.npy", "cam3_000001.npy"], "action": 5}
```

Each `.npy` contains float32 array shape `(frames, 4, H, W)`
where 4 = RGBD channels.

## Train
```
python3 -m src.train --data DATA_ROOT --epochs 20 --batch 16
```

## Export ONNX
```
python3 -m src.export_onnx --ckpt checkpoints/epoch_020.pt --out policy.onnx
```

## Action Mapping
See `src/actions.py` for action IDs and step sizes.

## Tests
```
python3 -m unittest discover -s tests -p "test_*.py"
```

## Config
Defaults are in `config.yaml`. If PyYAML is installed, training/export will read it.
