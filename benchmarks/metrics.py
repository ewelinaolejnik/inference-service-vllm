"""Latency and throughput statistics computed from raw benchmark data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LatencyStats:
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float

    def as_dict(self) -> dict:
        return {k: round(v, 2) for k, v in self.__dict__.items()}


@dataclass
class ThroughputStats:
    rps: float
    tokens_per_second: float

    def as_dict(self) -> dict:
        return {
            "rps": round(self.rps, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
        }


def compute_latency_stats(latencies_ms: list[float]) -> LatencyStats:
    """Derive percentile-based latency stats from a list of measurements in ms."""
    if not latencies_ms:
        return LatencyStats(0, 0, 0, 0, 0, 0, 0)

    arr = np.array(latencies_ms)
    return LatencyStats(
        avg_ms=float(np.mean(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        std_ms=float(np.std(arr)),
    )


def compute_throughput(
    total_requests: int,
    total_output_tokens: int,
    total_time_s: float,
) -> ThroughputStats:
    """Compute RPS and tokens/s from totals."""
    if total_time_s <= 0:
        return ThroughputStats(rps=0.0, tokens_per_second=0.0)
    return ThroughputStats(
        rps=total_requests / total_time_s,
        tokens_per_second=total_output_tokens / total_time_s,
    )
