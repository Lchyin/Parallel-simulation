from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from parallel_sim.config.schema import validate_config
from parallel_sim.hardware.spec import ClusterSpec, DeviceSpec, NodeSpec
from parallel_sim.parallel.strategies import ParallelStrategy
from parallel_sim.scheduler.strategies import SchedulePolicy


@dataclass
class SimulationConfig:
    raw: Dict[str, Any]
    cluster: ClusterSpec
    strategy: ParallelStrategy
    schedule: SchedulePolicy


class ConfigLoader:
    def load(self, path: str | Path) -> SimulationConfig:
        data = json.loads(Path(path).read_text())
        result = validate_config(data)
        if not result.ok:
            raise ValueError("Invalid config: " + "; ".join(result.errors))

        nodes = []
        for n in data["hardware"]["nodes"]:
            devices = [DeviceSpec(**d) for d in n["devices"]]
            nodes.append(NodeSpec(name=n["name"], devices=devices, topology=n.get("topology", "fully_connected")))
        cluster = ClusterSpec(
            nodes=nodes,
            cross_node_bandwidth_gbps=float(data["hardware"].get("cross_node_bandwidth_gbps", 400)),
            cross_node_latency_us=float(data["hardware"].get("cross_node_latency_us", 5.0)),
        )

        st = data["strategy"]
        strategy = ParallelStrategy(
            name=st.get("name", "from_config"),
            tensor_parallel=int(st["tp"]),
            pipeline_parallel=int(st["pp"]),
            data_parallel=int(st["dp"]),
            sequence_parallel=bool(st.get("sp", False)),
        )

        sc = data["schedule"]
        schedule = SchedulePolicy(
            name=sc.get("name", "from_config"),
            overlap_communication=bool(sc.get("overlap_communication", True)),
            micro_batch_size=int(sc.get("micro_batch_size", 1)),
        )

        return SimulationConfig(raw=data, cluster=cluster, strategy=strategy, schedule=schedule)
