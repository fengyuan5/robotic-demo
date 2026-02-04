"""
Simulation collection stub.

Expected outputs:
- samples.jsonl with {"paths": [cam0.npy, cam1.npy, cam2.npy, cam3.npy], "action": int}
- per-camera .npy arrays: float32 shape (frames, 4, H, W)

Integrate with your simulator by filling in:
- reset_env()
- step_env(action_id)
- get_multicam_rgbd()
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def reset_env() -> None:
    raise NotImplementedError


def step_env(action_id: int) -> Tuple[bool, dict]:
    """Returns done flag and info dict."""
    raise NotImplementedError


def get_multicam_rgbd() -> List[np.ndarray]:
    """
    Returns list of 4 RGB-D arrays, each shape (frames, 4, H, W).
    """
    raise NotImplementedError


def main(out_dir: str = "sim_data", episodes: int = 1000) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "samples.jsonl"

    with index_path.open("w", encoding="utf-8") as f:
        for ep in range(episodes):
            reset_env()
            done = False
            step_idx = 0
            while not done:
                cams = get_multicam_rgbd()
                # TODO: replace with expert or scripted policy
                action_id = 0

                paths = []
                for i, arr in enumerate(cams):
                    name = f"cam{i}_ep{ep:06d}_step{step_idx:04d}.npy"
                    np.save(out / name, arr.astype(np.float32))
                    paths.append(name)

                f.write(json.dumps({"paths": paths, "action": int(action_id)}) + "\n")
                done, _info = step_env(action_id)
                step_idx += 1


if __name__ == "__main__":
    main()
