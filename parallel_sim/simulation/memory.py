from __future__ import annotations

from dataclasses import dataclass
from typing import List

from parallel_sim.analysis.metrics import ModelMetrics
from parallel_sim.models.graph import ModelGraph


@dataclass
class MemorySnapshot:
    stage: int
    micro_batch: int
    weights_bytes: int
    grads_bytes: int
    optimizer_bytes: int
    activations_bytes: int
    total_bytes: int


def estimate_memory_timeline(
    model: ModelGraph,
    metrics: ModelMetrics,
    pipeline_stages: int = 1,
    micro_batches: int = 1,
    recompute: bool = False,
    optimizer_multiplier: float = 2.0,
) -> List[MemorySnapshot]:
    weights = model.total_output_memory_bytes
    grads = weights
    optimizer = int(weights * optimizer_multiplier)
    act_total = metrics.activation_bytes
    if recompute:
        act_total = int(act_total * 0.6)

    timeline: List[MemorySnapshot] = []
    stage_act = max(act_total // max(pipeline_stages, 1), 1)
    mb_act = max(stage_act // max(micro_batches, 1), 1)
    for s in range(pipeline_stages):
        acc = 0
        for mb in range(micro_batches):
            acc = min(stage_act, acc + mb_act)
            total = weights + grads + optimizer + acc
            timeline.append(MemorySnapshot(s, mb, weights, grads, optimizer, acc, total))
    return timeline
