from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CalibrationResult:
    compute_efficiency: float
    memory_efficiency: float
    comm_alpha_us: float
    comm_beta_gbps: float


def calibrate_from_csv(path: str | Path) -> CalibrationResult:
    # CSV columns: peak_tflops, achieved_tflops, peak_mem_gbps, achieved_mem_gbps, msg_bytes, comm_time_us
    rows = list(csv.DictReader(Path(path).read_text().splitlines()))
    if not rows:
        raise ValueError("empty calibration csv")

    c_eff = sum(float(r["achieved_tflops"]) / max(float(r["peak_tflops"]), 1e-9) for r in rows) / len(rows)
    m_eff = sum(float(r["achieved_mem_gbps"]) / max(float(r["peak_mem_gbps"]), 1e-9) for r in rows) / len(rows)

    alphas = []
    betas = []
    for r in rows:
        t = float(r["comm_time_us"])
        n = float(r["msg_bytes"])
        # t(us)=alpha + n/B, B(byte/us) => beta(gbps)
        alpha = max(t * 0.2, 0.1)
        byte_per_us = max(n / max(t - alpha, 1e-6), 1e-6)
        gbps = byte_per_us * 8 / 1e3
        alphas.append(alpha)
        betas.append(gbps)

    return CalibrationResult(c_eff, m_eff, sum(alphas) / len(alphas), sum(betas) / len(betas))
