import unittest

from src.config import load_config


class TestConfig(unittest.TestCase):
    def test_missing_config(self) -> None:
        cfg = load_config("/tmp/does_not_exist.yaml")
        self.assertEqual(cfg, {})


if __name__ == "__main__":
    unittest.main()
