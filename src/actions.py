from dataclasses import dataclass
from typing import Dict, List

from src.config import load_config


@dataclass(frozen=True)
class ActionSpec:
    name: str
    group: str
    delta: float = 0.0
    unit: str = ""


def _build_action_specs(step_pos_m: float, step_rot_deg: float) -> List[ActionSpec]:
    return [
        ActionSpec("FWD", "base"),
        ActionSpec("LEFT", "base"),
        ActionSpec("RIGHT", "base"),
        ActionSpec("STOP", "base"),
        ActionSpec("ALIGN_LEFT", "align"),
        ActionSpec("ALIGN_RIGHT", "align"),
        ActionSpec("ALIGN_FORWARD", "align"),
        ActionSpec("X+", "arm_pos", step_pos_m, "m"),
        ActionSpec("X-", "arm_pos", -step_pos_m, "m"),
        ActionSpec("Y+", "arm_pos", step_pos_m, "m"),
        ActionSpec("Y-", "arm_pos", -step_pos_m, "m"),
        ActionSpec("Z+", "arm_pos", step_pos_m, "m"),
        ActionSpec("Z-", "arm_pos", -step_pos_m, "m"),
        ActionSpec("ROLL+", "arm_rot", step_rot_deg, "deg"),
        ActionSpec("ROLL-", "arm_rot", -step_rot_deg, "deg"),
        ActionSpec("PITCH+", "arm_rot", step_rot_deg, "deg"),
        ActionSpec("PITCH-", "arm_rot", -step_rot_deg, "deg"),
        ActionSpec("YAW+", "arm_rot", step_rot_deg, "deg"),
        ActionSpec("YAW-", "arm_rot", -step_rot_deg, "deg"),
        ActionSpec("GRIP_OPEN", "gripper"),
        ActionSpec("GRIP_CLOSE", "gripper"),
        ActionSpec("BACKOFF", "recovery"),
    ]


_cfg = load_config()
_step_pos = float(_cfg.get("actions", {}).get("step_pos_m", 0.015))
_step_rot = float(_cfg.get("actions", {}).get("step_rot_deg", 3.0))

ACTION_SPECS: List[ActionSpec] = _build_action_specs(_step_pos, _step_rot)
ACTION_TO_ID: Dict[str, int] = {spec.name: i for i, spec in enumerate(ACTION_SPECS)}
ID_TO_ACTION: Dict[int, str] = {i: spec.name for i, spec in enumerate(ACTION_SPECS)}


def num_actions() -> int:
    return len(ACTION_SPECS)
