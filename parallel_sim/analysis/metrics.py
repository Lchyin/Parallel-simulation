from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from parallel_sim.models.graph import ModelGraph


@dataclass
class ModelMetrics:
    flops: float
    output_memory_bytes: int
    communication_bytes: int
    memory_pattern: str


def estimate_model_metrics(model: ModelGraph, tensor_parallel: int = 1) -> ModelMetrics:
    flops = model.total_flops
    out_bytes = model.total_output_memory_bytes
    communication_bytes = int(out_bytes * max(tensor_parallel - 1, 0) / max(tensor_parallel, 1))
    memory_pattern = "streaming" if len(model.ops) > 32 else "layerwise_reuse"
    return ModelMetrics(
        flops=flops,
        output_memory_bytes=out_bytes,
        communication_bytes=communication_bytes,
        memory_pattern=memory_pattern,
    )


def bottleneck_analysis(runtime_s: float, compute_s: float, comm_s: float, memory_s: float) -> Dict[str, str]:
    items = {"compute": compute_s, "communication": comm_s, "memory": memory_s}
    major = max(items, key=items.get)
    advice = {
        "compute": "提高算力利用率：开启算子融合、增大batch、降低pipeline bubble。",
        "communication": "降低通信瓶颈：提高TP局部性、开启通信计算重叠、优化拓扑映射。",
        "memory": "优化访存：重计算、激活检查点、算子重排以提升缓存命中。",
    }[major]
    return {"runtime_s": f"{runtime_s:.4f}", "major_bottleneck": major, "recommendation": advice}
