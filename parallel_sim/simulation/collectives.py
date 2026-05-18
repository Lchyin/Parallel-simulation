from __future__ import annotations


def _alpha_beta(message_bytes: int, participants: int, bandwidth_gbps: float, latency_us: float) -> float:
    if participants <= 1 or message_bytes <= 0:
        return 0.0
    alpha = latency_us * 1e-6
    beta_Bps = max(bandwidth_gbps, 1e-9) * 1e9 / 8.0
    return alpha + message_bytes / (participants * beta_Bps)


def all_reduce_time(message_bytes: int, participants: int, bandwidth_gbps: float, latency_us: float) -> float:
    return 2.0 * (participants - 1) * _alpha_beta(message_bytes, participants, bandwidth_gbps, latency_us)


def all_gather_time(message_bytes: int, participants: int, bandwidth_gbps: float, latency_us: float) -> float:
    return (participants - 1) * _alpha_beta(message_bytes, participants, bandwidth_gbps, latency_us)


def reduce_scatter_time(message_bytes: int, participants: int, bandwidth_gbps: float, latency_us: float) -> float:
    return (participants - 1) * _alpha_beta(message_bytes, participants, bandwidth_gbps, latency_us)
