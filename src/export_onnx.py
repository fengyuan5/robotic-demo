import argparse
import torch

from src.config import load_config
from src.model import MultiCamPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="policy.onnx")
    p.add_argument("--cams", type=int, default=None)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--actions", type=int, default=None)
    p.add_argument("--h", type=int, default=None)
    p.add_argument("--w", type=int, default=None)
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    cams = args.cams or int(cfg.get("cameras", {}).get("count", 4))
    frames = args.frames or int(cfg.get("cameras", {}).get("frames", 3))
    actions = args.actions or int(cfg.get("model", {}).get("actions", 22))
    h = args.h or int(cfg.get("cameras", {}).get("height", 224))
    w = args.w or int(cfg.get("cameras", {}).get("width", 224))

    model = MultiCamPolicy(cams=cams, frames=frames, actions=actions)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    dummy = torch.randn(1, cams, 4 * frames, h, w)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["rgbd"],
        output_names=["action_logits"],
        dynamic_axes={"rgbd": {0: "batch"}, "action_logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()
