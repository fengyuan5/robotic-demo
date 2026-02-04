import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency in tests
    np = None

from src.dataset import MultiCamDataset


@unittest.skipIf(np is None, "numpy not installed")
class TestDataset(unittest.TestCase):
    def test_dataset_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames, h, w = 3, 8, 8
            paths = []
            for cam in range(4):
                arr = np.zeros((frames, 4, h, w), dtype=np.float32)
                name = f"cam{cam}_000000.npy"
                np.save(root / name, arr)
                paths.append(name)

            with (root / "samples.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({"paths": paths, "action": 3}) + "\n")

            ds = MultiCamDataset(str(root))
            x, y = ds[0]
            self.assertEqual(x.shape, (4, frames * 4, h, w))
            self.assertEqual(int(y), 3)


if __name__ == "__main__":
    unittest.main()
