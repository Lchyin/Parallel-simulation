from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ParallelStrategy:
    name: str
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    sequence_parallel: bool = False

    def validate(self, total_devices: int) -> bool:
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel <= total_devices

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "tp": self.tensor_parallel,
            "pp": self.pipeline_parallel,
            "dp": self.data_parallel,
            "sp": self.sequence_parallel,
        }


def default_strategy_space(max_devices: int) -> list[ParallelStrategy]:
    space = []
    for tp in [1, 2, 4, 8]:
        for pp in [1, 2, 4]:
            for dp in [1, 2, 4]:
                s = ParallelStrategy(name=f"tp{tp}_pp{pp}_dp{dp}", tensor_parallel=tp, pipeline_parallel=pp, data_parallel=dp)
                if s.validate(max_devices):
                    space.append(s)
    return space
