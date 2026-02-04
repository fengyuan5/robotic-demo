from pathlib import Path
from typing import Any, Dict


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data
