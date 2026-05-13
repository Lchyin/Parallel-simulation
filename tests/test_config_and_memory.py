from parallel_sim.analysis.metrics import estimate_model_metrics
from parallel_sim.config.loader import ConfigLoader
from parallel_sim.models.graph import ModelGraph, ModelOp, TensorShape
from parallel_sim.simulation.memory import estimate_memory_timeline


def test_config_loader():
    cfg = ConfigLoader().load("examples/simulation_config.json")
    assert cfg.strategy.tensor_parallel == 2
    assert cfg.schedule.micro_batch_size == 2
    assert cfg.cluster.total_devices == 2


def test_memory_timeline():
    model = ModelGraph(
        name="t",
        ops=[ModelOp("mm", "MatMul", ["a"], ["b"], TensorShape([8, 8]), {"m": 8, "k": 8, "n": 8})],
    )
    metrics = estimate_model_metrics(model, 1, 1)
    snapshots = estimate_memory_timeline(model, metrics)
    assert len(snapshots) == 1
    assert snapshots[0].total_bytes > 0
