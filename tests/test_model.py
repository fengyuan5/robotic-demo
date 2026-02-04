import unittest

try:
    import torch
except Exception:  # pragma: no cover - optional dependency in tests
    torch = None

from src.model import MultiCamPolicy


@unittest.skipIf(torch is None, "torch not installed")
class TestModel(unittest.TestCase):
    def test_forward_shape(self) -> None:
        model = MultiCamPolicy(cams=4, frames=3, actions=22)
        x = torch.randn(2, 4, 12, 64, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 22))


if __name__ == "__main__":
    unittest.main()
