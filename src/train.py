import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import MultiCamDataset
from src.model import MultiCamPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Dataset root")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--actions", type=int, default=None)
    p.add_argument("--cams", type=int, default=None)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actions = args.actions or int(cfg.get("model", {}).get("actions", 22))
    cams = args.cams or int(cfg.get("cameras", {}).get("count", 4))
    frames = args.frames or int(cfg.get("cameras", {}).get("frames", 3))
    batch = args.batch or int(cfg.get("training", {}).get("batch", 16))
    epochs = args.epochs or int(cfg.get("training", {}).get("epochs", 20))

    ds = MultiCamDataset(args.data)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)

    model = MultiCamPolicy(cams=cams, frames=frames, actions=actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)

        avg = total / len(ds)
        ckpt = out_dir / f"epoch_{epoch:03d}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
        print(f"epoch={epoch} loss={avg:.4f} ckpt={ckpt}")


if __name__ == "__main__":
    main()
