from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from parallel_sim.analysis.efficiency import EfficiencyTable, extract_size_hint
from parallel_sim.models.graph import ModelGraph


@dataclass
class ModelMetrics:
    flops: float
    effective_flops: float
    output_memory_bytes: int
    activation_bytes: int
    tp_ar_bytes: int
    tp_ag_bytes: int
    dp_rs_bytes: int
    dp_ar_bytes: int
    memory_pattern: str


def estimate_model_metrics(model: ModelGraph, tensor_parallel: int = 1, data_parallel: int = 1, dtype: str = "fp16") -> ModelMetrics:
    table = EfficiencyTable()
    flops = model.total_flops
    eff_weighted = 0.0
    for op in model.ops:
        hint = extract_size_hint(op.attrs)
        eff = table.query(op.op_type, dtype, hint)
        eff_weighted += op.estimate_flops() * eff

    out_bytes = model.total_output_memory_bytes
    activation_bytes = int(out_bytes * 1.5)

    tp_factor = max(tensor_parallel - 1, 0) / max(tensor_parallel, 1)
    dp_factor = max(data_parallel - 1, 0) / max(data_parallel, 1)

    tp_ar_bytes = int(out_bytes * tp_factor)
    tp_ag_bytes = int(out_bytes * tp_factor)
    dp_rs_bytes = int(out_bytes * dp_factor)
    dp_ar_bytes = int(out_bytes * dp_factor)

    memory_pattern = "streaming" if len(model.ops) > 32 else "layerwise_reuse"
    return ModelMetrics(flops, eff_weighted, out_bytes, activation_bytes, tp_ar_bytes, tp_ag_bytes, dp_rs_bytes, dp_ar_bytes, memory_pattern)


def bottleneck_analysis(runtime_s: float, compute_s: float, comm_s: float, memory_s: float) -> Dict[str, str]:
    items = {"compute": compute_s, "communication": comm_s, "memory": memory_s}
    major = max(items, key=items.get)
    advice = {
        "compute": "提高算力利用率：开启算子融合、增大batch、降低pipeline bubble。",
        "communication": "降低通信瓶颈：提高TP/DP局部性、通信计算重叠、优化拓扑映射和collective算法。",
        "memory": "优化访存：重计算、激活检查点、算子重排以提升缓存命中。",
    }[major]
    return {"runtime_s": f"{runtime_s:.4f}", "major_bottleneck": major, "recommendation": advice}
