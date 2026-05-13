from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SchedulePolicy:
    name: str
    overlap_communication: bool
    micro_batch_size: int


def scheduler_space() -> List[SchedulePolicy]:
    return [
        SchedulePolicy("greedy", overlap_communication=False, micro_batch_size=1),
        SchedulePolicy("greedy_overlap", overlap_communication=True, micro_batch_size=1),
        SchedulePolicy("wavefront", overlap_communication=True, micro_batch_size=4),
    ]
