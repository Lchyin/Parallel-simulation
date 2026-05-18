from __future__ import annotations

from dataclasses import dataclass
from typing import List

from parallel_sim.analysis.metrics import estimate_model_metrics
from parallel_sim.hardware.spec import ClusterSpec
from parallel_sim.models.graph import ModelGraph
from parallel_sim.parallel.strategies import ParallelStrategy
from parallel_sim.scheduler.strategies import SchedulePolicy
from parallel_sim.simulation.collectives import all_gather_time, all_reduce_time, reduce_scatter_time


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
        metrics = estimate_model_metrics(model, strategy.tensor_parallel, strategy.data_parallel)
        devices = [d for n in cluster.nodes for d in n.devices]

        total_peak_tflops = sum(d.peak_tflops * d.compute_efficiency for d in devices)
        total_mem_bw = sum(d.memory_bandwidth_gbps * d.memory_efficiency for d in devices)
        compute_time = metrics.effective_flops / max(total_peak_tflops * 1e12, 1.0)
        memory_time = (metrics.activation_bytes * 8) / max(total_mem_bw * 1e9, 1.0)

        intra_bw = sum(d.interconnect_bandwidth_gbps for d in devices) / max(len(devices), 1)
        intra_lat = 1.0
        inter_bw = cluster.comm_beta_gbps
        inter_lat = cluster.comm_alpha_us

        tp = max(strategy.tensor_parallel, 1)
        dp = max(strategy.data_parallel, 1)

        tp_time = all_reduce_time(metrics.tp_ar_bytes, tp, intra_bw, intra_lat) + all_gather_time(metrics.tp_ag_bytes, tp, intra_bw, intra_lat)
        dp_time = reduce_scatter_time(metrics.dp_rs_bytes, dp, inter_bw, inter_lat) + all_reduce_time(metrics.dp_ar_bytes, dp, inter_bw, inter_lat)

        overlap_factor = 0.75 if schedule.overlap_communication else 1.0
        comm_time = (tp_time + dp_time) * overlap_factor

        bubble_penalty = 1.0 + (strategy.pipeline_parallel - 1) / max(strategy.pipeline_parallel * schedule.micro_batch_size, 1)
        total = (compute_time + memory_time + comm_time) * bubble_penalty

        events = [
            TimelineEvent("compute", 0.0, compute_time, "compute phase"),
            TimelineEvent("memory", compute_time, compute_time + memory_time, "memory phase"),
            TimelineEvent("comm", compute_time + memory_time, compute_time + memory_time + comm_time, "communication phase"),
            TimelineEvent("pipeline_bubble", compute_time + memory_time + comm_time, total, "pipeline bubble penalty"),
        ]
        return SimulationResult(total, compute_time, comm_time, memory_time, events)
