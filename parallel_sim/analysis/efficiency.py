from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class EfficiencyPoint:
    op_type: str
    dtype: str
    min_mnk: int
    max_mnk: int
    efficiency: float


class EfficiencyTable:
    def __init__(self, points: list[EfficiencyPoint] | None = None):
        self.points = points or [
            EfficiencyPoint("MatMul", "fp16", 0, 512**3, 0.35),
            EfficiencyPoint("MatMul", "fp16", 512**3, 4096**3, 0.52),
            EfficiencyPoint("MatMul", "bf16", 0, 512**3, 0.33),
            EfficiencyPoint("MatMul", "bf16", 512**3, 4096**3, 0.50),
            EfficiencyPoint("Conv", "fp16", 0, 10**18, 0.45),
        ]

    def query(self, op_type: str, dtype: str, size_hint: int) -> float:
        for p in self.points:
            if p.op_type == op_type and p.dtype == dtype and p.min_mnk <= size_hint < p.max_mnk:
                return p.efficiency
        return 0.4


def extract_size_hint(attrs: Dict[str, object]) -> int:
    m, k, n = int(attrs.get("m", 1)), int(attrs.get("k", 1)), int(attrs.get("n", 1))
    return max(m * k * n, 1)
