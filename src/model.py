try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


if nn is None:  # pragma: no cover - optional dependency
    class Backbone:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("torch not installed; cannot use Backbone")

        def forward(self, x):
            raise RuntimeError("torch not installed; cannot use Backbone")
else:
    class Backbone(nn.Module):
        def __init__(self, in_ch: int = 12, out_ch: int = 256):
            super().__init__()
            # Lightweight CNN; replace with torchvision resnet18 if desired.
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, out_ch, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)
    def __init__(self, in_ch: int = 12, out_ch: int = 256):
        super().__init__()
        # Lightweight CNN; replace with torchvision resnet18 if desired.
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_ch, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if nn is None:  # pragma: no cover - optional dependency
    class MultiCamPolicy:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("torch not installed; cannot use MultiCamPolicy")

        def forward(self, x):
            raise RuntimeError("torch not installed; cannot use MultiCamPolicy")
else:
    class MultiCamPolicy(nn.Module):
        def __init__(self, cams: int = 4, frames: int = 3, actions: int = 22):
            super().__init__()
            in_ch = 4 * frames
            self.cams = cams
            self.backbones = nn.ModuleList([Backbone(in_ch=in_ch) for _ in range(cams)])
            self.fuse = nn.Sequential(
                nn.Conv2d(256 * cams, 256, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, actions),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, cams, 4*frames, H, W)
            feats = []
            for i in range(self.cams):
                feats.append(self.backbones[i](x[:, i]))
            fused = torch.cat(feats, dim=1)
            fused = self.fuse(fused)
            pooled = self.pool(fused)
            return self.head(pooled)
