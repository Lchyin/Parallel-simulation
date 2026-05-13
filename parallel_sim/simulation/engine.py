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
    def _allreduce_alpha_beta_time(self, message_bytes: int, participants: int, bandwidth_gbps: float, latency_us: float) -> float:
        # Ring all-reduce approx: 2*(p-1)*(alpha + n/(p*beta))
        if participants <= 1 or message_bytes <= 0:
            return 0.0
        alpha = latency_us * 1e-6
        beta_Bps = max(bandwidth_gbps, 1e-9) * 1e9 / 8.0
        return 2.0 * (participants - 1) * (alpha + message_bytes / (participants * beta_Bps))

    def run(self, model: ModelGraph, cluster: ClusterSpec, strategy: ParallelStrategy, schedule: SchedulePolicy) -> SimulationResult:
        metrics = estimate_model_metrics(model, strategy.tensor_parallel, strategy.data_parallel)

        devices = [d for n in cluster.nodes for d in n.devices]
        total_peak_tflops = sum(d.peak_tflops * d.compute_efficiency for d in devices)
        total_mem_bw = sum(d.memory_bandwidth_gbps * d.memory_efficiency for d in devices)

        compute_time = metrics.flops / max(total_peak_tflops * 1e12, 1.0)
        memory_time = (metrics.activation_bytes * 8) / max(total_mem_bw * 1e9, 1.0)

        # 通信分两层：节点内（高带宽）+ 节点间（低带宽高时延）
        intra_bw = sum(d.interconnect_bandwidth_gbps for d in devices) / max(len(devices), 1)
        intra_latency_us = 1.0
        inter_bw = cluster.cross_node_bandwidth_gbps
        inter_latency_us = cluster.cross_node_latency_us

        tp_group = max(strategy.tensor_parallel, 1)
        dp_group = max(strategy.data_parallel, 1)

        tp_bytes = int(metrics.communication_bytes * (tp_group / max(tp_group + dp_group, 1)))
        dp_bytes = metrics.communication_bytes - tp_bytes

        tp_comm = self._allreduce_alpha_beta_time(tp_bytes, tp_group, intra_bw, intra_latency_us)
        inter_participants = max(len(cluster.nodes), 1)
        dp_comm = self._allreduce_alpha_beta_time(dp_bytes, max(dp_group, inter_participants), inter_bw, inter_latency_us)

        overlap_factor = 0.75 if schedule.overlap_communication else 1.0
        comm_time = (tp_comm + dp_comm) * overlap_factor

        bubble_penalty = 1.0 + (strategy.pipeline_parallel - 1) / max(strategy.pipeline_parallel * schedule.micro_batch_size, 1)
        total = (compute_time + memory_time + comm_time) * bubble_penalty

        events = [
            TimelineEvent("compute", 0.0, compute_time, "compute phase"),
            TimelineEvent("memory", compute_time, compute_time + memory_time, "memory phase"),
            TimelineEvent("comm", compute_time + memory_time, compute_time + memory_time + comm_time, "communication phase"),
            TimelineEvent("pipeline_bubble", compute_time + memory_time + comm_time, total, "pipeline bubble penalty"),
        ]
        return SimulationResult(total, compute_time, comm_time, memory_time, events)
