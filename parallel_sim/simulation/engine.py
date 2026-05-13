from __future__ import annotations

from dataclasses import dataclass
from typing import List

from parallel_sim.analysis.metrics import estimate_model_metrics
from parallel_sim.hardware.spec import ClusterSpec
from parallel_sim.models.graph import ModelGraph
from parallel_sim.parallel.strategies import ParallelStrategy
from parallel_sim.scheduler.strategies import SchedulePolicy


@dataclass
class TimelineEvent:
    resource: str
    start: float
    end: float
    label: str


@dataclass
class SimulationResult:
    total_time_s: float
    compute_time_s: float
    comm_time_s: float
    memory_time_s: float
    events: List[TimelineEvent]


class SimulationEngine:
    def run(self, model: ModelGraph, cluster: ClusterSpec, strategy: ParallelStrategy, schedule: SchedulePolicy) -> SimulationResult:
        metrics = estimate_model_metrics(model, strategy.tensor_parallel)
        total_peak_tflops = sum(d.peak_tflops for n in cluster.nodes for d in n.devices)
        total_mem_bw = sum(d.memory_bandwidth_gbps for n in cluster.nodes for d in n.devices)
        compute_time = metrics.flops / max(total_peak_tflops * 1e12, 1.0)
        memory_time = (metrics.output_memory_bytes * 8) / max(total_mem_bw * 1e9, 1.0)
        effective_bw = cluster.cross_node_bandwidth_gbps * (1.2 if schedule.overlap_communication else 1.0)
        comm_time = (metrics.communication_bytes * 8) / max(effective_bw * 1e9, 1.0)
        total = compute_time + memory_time + comm_time

        events = [
            TimelineEvent("compute", 0.0, compute_time, "compute phase"),
            TimelineEvent("memory", compute_time, compute_time + memory_time, "memory phase"),
            TimelineEvent("comm", compute_time + memory_time, total, "communication phase"),
        ]
        return SimulationResult(total, compute_time, comm_time, memory_time, events)
