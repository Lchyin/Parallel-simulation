from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DeviceSpec:
    name: str
    peak_tflops: float
    memory_bandwidth_gbps: float
    memory_capacity_gb: float
    interconnect_bandwidth_gbps: float


@dataclass
class NodeSpec:
    name: str
    devices: List[DeviceSpec]
    topology: str = "fully_connected"


@dataclass
class ClusterSpec:
    nodes: List[NodeSpec]
    cross_node_bandwidth_gbps: float
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def total_devices(self) -> int:
        return sum(len(n.devices) for n in self.nodes)
