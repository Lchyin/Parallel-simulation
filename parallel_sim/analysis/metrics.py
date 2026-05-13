from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from parallel_sim.models.graph import ModelGraph


@dataclass
class ModelMetrics:
    flops: float
    output_memory_bytes: int
    activation_bytes: int
    communication_bytes: int
    memory_pattern: str


def estimate_model_metrics(model: ModelGraph, tensor_parallel: int = 1, data_parallel: int = 1) -> ModelMetrics:
    flops = model.total_flops
    out_bytes = model.total_output_memory_bytes
    activation_bytes = int(out_bytes * 1.5)

    # 粗粒度通信需求分解：
    # - TP: 近似 all-reduce/all-gather，规模与 (tp-1)/tp 成正比
    # - DP: 梯度 all-reduce，规模与参数/激活量近似相关（此处以 out_bytes 代理）
    tp_bytes = int(out_bytes * max(tensor_parallel - 1, 0) / max(tensor_parallel, 1))
    dp_bytes = int(out_bytes * max(data_parallel - 1, 0) / max(data_parallel, 1))
    communication_bytes = tp_bytes + dp_bytes

    memory_pattern = "streaming" if len(model.ops) > 32 else "layerwise_reuse"
    return ModelMetrics(
        flops=flops,
        output_memory_bytes=out_bytes,
        activation_bytes=activation_bytes,
        communication_bytes=communication_bytes,
        memory_pattern=memory_pattern,
    )


def bottleneck_analysis(runtime_s: float, compute_s: float, comm_s: float, memory_s: float) -> Dict[str, str]:
    items = {"compute": compute_s, "communication": comm_s, "memory": memory_s}
    major = max(items, key=items.get)
    advice = {
        "compute": "提高算力利用率：开启算子融合、增大batch、降低pipeline bubble。",
        "communication": "降低通信瓶颈：提高TP/DP局部性、通信计算重叠、优化拓扑映射和collective算法。",
        "memory": "优化访存：重计算、激活检查点、算子重排以提升缓存命中。",
    }[major]
    return {"runtime_s": f"{runtime_s:.4f}", "major_bottleneck": major, "recommendation": advice}
