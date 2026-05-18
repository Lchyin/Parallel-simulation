from parallel_sim.analysis.metrics import estimate_model_metrics
from parallel_sim.config.loader import ConfigLoader
from parallel_sim.hardware.spec import ClusterSpec, DeviceSpec, NodeSpec
from parallel_sim.io.model_loader import LoaderRegistry
from parallel_sim.optimization.search import Explorer
from parallel_sim.parallel.strategies import default_strategy_space
from parallel_sim.report import gantt_ascii, summarize
from parallel_sim.scheduler.strategies import scheduler_space
from parallel_sim.simulation.engine import SimulationEngine
from parallel_sim.simulation.memory import estimate_memory_timeline


def build_demo_cluster() -> ClusterSpec:
    node = NodeSpec(
        name="node0",
        devices=[
            DeviceSpec("gpu0", peak_tflops=120, memory_bandwidth_gbps=3000, memory_capacity_gb=80, interconnect_bandwidth_gbps=900),
            DeviceSpec("gpu1", peak_tflops=120, memory_bandwidth_gbps=3000, memory_capacity_gb=80, interconnect_bandwidth_gbps=900),
        ],
    )
    node2 = NodeSpec(
        name="node1",
        devices=[
            DeviceSpec("gpu2", peak_tflops=120, memory_bandwidth_gbps=3000, memory_capacity_gb=80, interconnect_bandwidth_gbps=900),
            DeviceSpec("gpu3", peak_tflops=120, memory_bandwidth_gbps=3000, memory_capacity_gb=80, interconnect_bandwidth_gbps=900),
        ],
    )
    return ClusterSpec(nodes=[node, node2], cross_node_bandwidth_gbps=400)


def main() -> None:
    loader = LoaderRegistry()
    model = loader.load("demo.onnx", "onnx")

    try:
        cfg = ConfigLoader().load("examples/simulation_config.json")
        cluster = cfg.cluster
        strategies = [cfg.strategy]
        schedules = [cfg.schedule]
    except Exception:
        cluster = build_demo_cluster()
        strategies = default_strategy_space(cluster.total_devices)
        schedules = scheduler_space()

    explorer = Explorer(SimulationEngine())
    records = explorer.explore(
        model,
        cluster,
        strategies,
        schedules,
        param_grid={"batch": [1, 2, 4], "seq": [1024, 2048]},
    )
    best = records[0]
    metrics = estimate_model_metrics(model, best.strategy.tensor_parallel, best.strategy.data_parallel)
    mem_timeline = estimate_memory_timeline(model, metrics, best.strategy.pipeline_parallel, best.schedule.micro_batch_size, recompute=True)

    print("Model metrics:", metrics)
    print("Best:", best.strategy.to_dict(), best.schedule)
    print(summarize(best.result))
    print("Peak memory(bytes):", max(s.total_bytes for s in mem_timeline))
    print(gantt_ascii(best.result.events))


if __name__ == "__main__":
    main()
