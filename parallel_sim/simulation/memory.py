from __future__ import annotations

from dataclasses import dataclass
from typing import List

from parallel_sim.analysis.metrics import ModelMetrics
from parallel_sim.models.graph import ModelGraph


@dataclass
class MemorySnapshot:
    step: int
    weights_bytes: int
    grads_bytes: int
    optimizer_bytes: int
    activations_bytes: int
    total_bytes: int


def estimate_memory_timeline(model: ModelGraph, metrics: ModelMetrics, optimizer_multiplier: float = 2.0) -> List[MemorySnapshot]:
    # 粗略假设：weights≈outputs，grads≈weights，optimizer≈2x weights
    weights = model.total_output_memory_bytes
    grads = weights
    optimizer = int(weights * optimizer_multiplier)

    timeline: List[MemorySnapshot] = []
    act = 0
    steps = max(len(model.ops), 1)
    per_step = max(metrics.activation_bytes // steps, 1)
    for i in range(steps):
        act = min(metrics.activation_bytes, act + per_step)
        total = weights + grads + optimizer + act
        timeline.append(MemorySnapshot(i, weights, grads, optimizer, act, total))
    return timeline
