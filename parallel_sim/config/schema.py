from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ConfigValidationResult:
    ok: bool
    errors: List[str]


def validate_config(cfg: Dict[str, Any]) -> ConfigValidationResult:
    errors: List[str] = []
    for key in ["model", "hardware", "strategy", "schedule"]:
        if key not in cfg:
            errors.append(f"missing required section: {key}")

    if "hardware" in cfg:
        hw = cfg["hardware"]
        if "nodes" not in hw or not isinstance(hw["nodes"], list) or len(hw["nodes"]) == 0:
            errors.append("hardware.nodes must be a non-empty list")

    if "strategy" in cfg:
        st = cfg["strategy"]
        for k in ["tp", "pp", "dp"]:
            if k not in st:
                errors.append(f"strategy.{k} is required")
            elif int(st[k]) < 1:
                errors.append(f"strategy.{k} must be >= 1")

    if "schedule" in cfg:
        sc = cfg["schedule"]
        if int(sc.get("micro_batch_size", 1)) < 1:
            errors.append("schedule.micro_batch_size must be >= 1")

    return ConfigValidationResult(ok=len(errors) == 0, errors=errors)
