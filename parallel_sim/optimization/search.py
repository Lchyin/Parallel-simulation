from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List

from parallel_sim.hardware.spec import ClusterSpec
from parallel_sim.models.graph import ModelGraph
from parallel_sim.parallel.strategies import ParallelStrategy
from parallel_sim.scheduler.strategies import SchedulePolicy
from parallel_sim.simulation.engine import SimulationEngine, SimulationResult


@dataclass
class SearchRecord:
    strategy: ParallelStrategy
    schedule: SchedulePolicy
    params: Dict[str, int]
    result: SimulationResult


class Explorer:
    def __init__(self, engine: SimulationEngine):
        self.engine = engine

    def explore(
        self,
        model: ModelGraph,
        cluster: ClusterSpec,
        strategies: List[ParallelStrategy],
        schedules: List[SchedulePolicy],
        param_grid: Dict[str, List[int]],
    ) -> List[SearchRecord]:
        keys = sorted(param_grid.keys())
        values = [param_grid[k] for k in keys]
        records: List[SearchRecord] = []
        for strategy, schedule, combo in product(strategies, schedules, product(*values)):
            if not strategy.validate(cluster.total_devices):
                continue
            params = dict(zip(keys, combo))
            result = self.engine.run(model, cluster, strategy, schedule)
            records.append(SearchRecord(strategy, schedule, params, result))
        records.sort(key=lambda r: r.result.total_time_s)
        return records
