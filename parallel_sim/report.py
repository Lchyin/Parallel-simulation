from __future__ import annotations

from typing import Iterable

from parallel_sim.analysis.metrics import bottleneck_analysis
from parallel_sim.simulation.engine import SimulationResult


def gantt_ascii(events: Iterable, width: int = 60) -> str:
    events = list(events)
    if not events:
        return ""
    t_end = max(e.end for e in events)
    lines = ["Gantt timeline"]
    for e in events:
        start = int((e.start / max(t_end, 1e-12)) * width)
        end = max(start + 1, int((e.end / max(t_end, 1e-12)) * width))
        bar = " " * start + "#" * (end - start)
        lines.append(f"{e.resource:>10}: {bar} {e.start:.6f}->{e.end:.6f}s")
    return "\n".join(lines)


def summarize(result: SimulationResult) -> str:
    analysis = bottleneck_analysis(result.total_time_s, result.compute_time_s, result.comm_time_s, result.memory_time_s)
    return (
        f"total={result.total_time_s:.6f}s, compute={result.compute_time_s:.6f}s, "
        f"memory={result.memory_time_s:.6f}s, comm={result.comm_time_s:.6f}s, "
        f"bottleneck={analysis['major_bottleneck']}, advice={analysis['recommendation']}"
    )
