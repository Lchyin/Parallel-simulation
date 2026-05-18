from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from parallel_sim.models.graph import ModelGraph, ModelOp, TensorShape


class ModelLoader(Protocol):
    def load(self, model_path: str | Path) -> ModelGraph:
        ...


@dataclass
class ONNXModelLoader:
    """ONNX loader with pluggable fallback for environments without onnx package."""

    def load(self, model_path: str | Path) -> ModelGraph:
        path = Path(model_path)
        try:
            import onnx  # type: ignore

            model = onnx.load(str(path))
            ops = []
            for i, node in enumerate(model.graph.node):
                # lightweight shape fallback
                shape = TensorShape([1])
                attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                ops.append(
                    ModelOp(
                        name=node.name or f"node_{i}",
                        op_type=node.op_type,
                        inputs=list(node.input),
                        outputs=list(node.output),
                        output_shape=shape,
                        attrs=attrs,
                    )
                )
            return ModelGraph(name=path.stem, ops=ops, metadata={"format": "onnx"})
        except Exception:
            # structured placeholder path for tests and extension
            return ModelGraph(
                name=path.stem,
                ops=[
                    ModelOp(
                        name="placeholder_matmul",
                        op_type="MatMul",
                        inputs=["x", "w"],
                        outputs=["y"],
                        output_shape=TensorShape([1024, 1024]),
                        attrs={"m": 1024, "k": 1024, "n": 1024},
                    )
                ],
                metadata={"format": "onnx", "fallback": True},
            )


@dataclass
class LoaderRegistry:
    onnx_loader: ModelLoader = field(default_factory=ONNXModelLoader)

    def load(self, model_path: str | Path, model_format: str = "onnx") -> ModelGraph:
        if model_format == "onnx":
            return self.onnx_loader.load(model_path)
        raise ValueError(f"Unsupported model format: {model_format}. Register extra loaders for extension.")
