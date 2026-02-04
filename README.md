# Project 0204

Embodied grasping policy skeleton with multi-RGBD input and discrete actions.

## Docs
- `docs/quickstart.md`
- `docs/data_format.md`
- `docs/runbook.md`
- `docs/tutorial_zh.md`

## Quick Start
```
python3 scripts/make_sample_data.py --out examples/sample_data
python3 -m src.train --data examples/sample_data
```

## Tests
```
python3 -m unittest discover -s tests -p "test_*.py"
```
