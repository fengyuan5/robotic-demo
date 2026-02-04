import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="examples/sample_data")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--h", type=int, default=64)
    p.add_argument("--w", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = out / "samples.jsonl"
    with index.open("w", encoding="utf-8") as f:
        for i in range(args.samples):
            paths = []
            for cam in range(4):
                arr = np.random.rand(args.frames, 4, args.h, args.w).astype(np.float32)
                name = f"cam{cam}_{i:06d}.npy"
                np.save(out / name, arr)
                paths.append(name)
            action = int(i % 22)
            f.write(json.dumps({"paths": paths, "action": action}) + "\n")


if __name__ == "__main__":
    main()
