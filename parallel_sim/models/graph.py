from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TensorShape:
    dims: List[int]

    @property
    def numel(self) -> int:
        n = 1
        for d in self.dims:
            n *= max(d, 1)
        return n


@dataclass
class ModelOp:
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    output_shape: TensorShape
    attrs: Dict[str, object] = field(default_factory=dict)

    def estimate_flops(self) -> float:
        # baseline estimator; extensible for more op types
        if self.op_type in {"MatMul", "Gemm"}:
            m, k, n = self.attrs.get("m", 1), self.attrs.get("k", 1), self.attrs.get("n", 1)
            return 2.0 * float(m) * float(k) * float(n)
        if self.op_type in {"Conv", "ConvTranspose"}:
            oc = int(self.attrs.get("out_channels", 1))
            ic = int(self.attrs.get("in_channels", 1))
            kh = int(self.attrs.get("kernel_h", 1))
            kw = int(self.attrs.get("kernel_w", 1))
            oh = int(self.attrs.get("out_h", 1))
            ow = int(self.attrs.get("out_w", 1))
            return 2.0 * oc * ic * kh * kw * oh * ow
        return float(self.output_shape.numel)

    def estimate_memory_bytes(self, dtype_bytes: int = 4) -> int:
        return self.output_shape.numel * dtype_bytes


@dataclass
class ModelGraph:
    name: str
    ops: List[ModelOp]
    metadata: Dict[str, object] = field(default_factory=dict)

    def op_by_name(self, name: str) -> Optional[ModelOp]:
        return next((o for o in self.ops if o.name == name), None)

    @property
    def total_flops(self) -> float:
        return sum(op.estimate_flops() for op in self.ops)

    @property
    def total_output_memory_bytes(self) -> int:
        return sum(op.estimate_memory_bytes() for op in self.ops)
