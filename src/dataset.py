import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    Dataset = object  # type: ignore


class MultiCamDataset(Dataset):
    """
    Expects a dataset directory with:
      - samples.jsonl: each line contains {"paths": [cam0.npy, cam1.npy, cam2.npy, cam3.npy], "action": int}
      - npy files store float32 arrays of shape (frames, 4, H, W) per camera
    """

    def __init__(self, root: str):
        if np is None or torch is None:
            raise RuntimeError("numpy/torch not installed; cannot use MultiCamDataset")
        self.root = Path(root)
        self.index = self._load_index(self.root / "samples.jsonl")

    @staticmethod
    def _load_index(path: Path) -> List[Dict]:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        if np is None or torch is None:
            raise RuntimeError("numpy/torch not installed; cannot use MultiCamDataset")
        item = self.index[idx]
        cams = []
        for rel in item["paths"]:
            arr = np.load(self.root / rel).astype(np.float32)
            # arr: (frames, 4, H, W) -> (4*frames, H, W)
            f, c, h, w = arr.shape
            cams.append(arr.reshape(f * c, h, w))
        x = np.stack(cams, axis=0)
        y = np.int64(item["action"])
        return torch.from_numpy(x), torch.tensor(y)
