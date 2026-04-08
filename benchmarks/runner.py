"""Benchmark runner for vLLM inference with full metric collection.

Collects: Avg Latency, P50, P95, RPS, Tokens/s, GPU Util %, Mem Used (MB),
across configurable batch sizes and dtypes with warmup support.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.config import AppConfig, Precision, load_config
from app.inference import InferenceEngine
from benchmarks.gpu_monitor import GPUMonitor, GPUStats
from benchmarks.metrics import (
    LatencyStats,
    ThroughputStats,
    compute_latency_stats,
    compute_throughput,
)

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to sort a list.",
    "What are the benefits of renewable energy?",
    "Describe the water cycle in detail.",
    "Summarize the plot of Romeo and Juliet.",
    "What is machine learning and how does it work?",
    "Explain quantum computing to a five-year-old.",
    "Write a short poem about the ocean.",
]


@dataclass
class ExperimentResult:
    model_name: str
    dtype: str
    batch_size: int
    num_iterations: int
    latency: LatencyStats
    throughput: ThroughputStats
    gpu_stats: GPUStats
    total_output_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "num_iterations": self.num_iterations,
            "total_output_tokens": self.total_output_tokens,
            **self.latency.as_dict(),
            **self.throughput.as_dict(),
            **self.gpu_stats.as_dict(),
        }


@dataclass
class BenchmarkConfig:
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dtypes: list[str] = field(default_factory=lambda: ["float16"])
    num_warmup: int = 2
    num_iterations: int = 5
    max_tokens: int = 64
    temperature: float = 0.0
    gpu_monitor_interval: float = 0.5
    gpu_memory_utilization: float = 0.85
    output_dir: str = "benchmarks/results"
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))


class BenchmarkRunner:

    def __init__(self, app_config: AppConfig, bench_config: BenchmarkConfig | None = None):
        self.app_config = app_config
        self.bench = bench_config or BenchmarkConfig()
        self.results: list[ExperimentResult] = []

    def _build_batches(self, batch_size: int) -> list[list[str]]:
        prompts = self.bench.prompts
        total = self.bench.num_warmup + self.bench.num_iterations
        batches: list[list[str]] = []
        for run_idx in range(total):
            start = run_idx * batch_size
            batch = [prompts[(start + j) % len(prompts)] for j in range(batch_size)]
            batches.append(batch)
        return batches

    def run_experiment(
        self,
        engine: InferenceEngine,
        dtype: str,
        batch_size: int,
    ) -> ExperimentResult:
        nw = self.bench.num_warmup
        ni = self.bench.num_iterations
        max_tok = self.bench.max_tokens
        temp = self.bench.temperature
        batches = self._build_batches(batch_size)

        logger.info(
            "  Experiment: dtype=%s  batch=%d  (%d warmup + %d iters)",
            dtype, batch_size, nw, ni,
        )

        for i in range(nw):
            engine.generate_batch(
                batches[i % len(batches)], max_tokens=max_tok, temperature=temp,
            )

        monitor = GPUMonitor(interval=self.bench.gpu_monitor_interval)
        monitor.start()

        latencies_ms: list[float] = []
        total_output_tokens = 0
        overall_start = time.perf_counter()

        for i in range(ni):
            batch = batches[(nw + i) % len(batches)]
            start = time.perf_counter()
            results = engine.generate_batch(
                batch, max_tokens=max_tok, temperature=temp,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)
            total_output_tokens += sum(r.output_tokens for r in results)
            logger.debug("    Iter %d/%d: %.1f ms", i + 1, ni, elapsed_ms)

        total_time_s = time.perf_counter() - overall_start
        gpu_stats = monitor.stop()
        total_requests = ni * batch_size

        experiment = ExperimentResult(
            model_name=self.app_config.model.model_name,
            dtype=dtype,
            batch_size=batch_size,
            num_iterations=ni,
            latency=compute_latency_stats(latencies_ms),
            throughput=compute_throughput(total_requests, total_output_tokens, total_time_s),
            gpu_stats=gpu_stats,
            total_output_tokens=total_output_tokens,
        )
        self.results.append(experiment)
        return experiment

    def run_all(self) -> list[ExperimentResult]:
        self.results.clear()

        for dtype_str in self.bench.dtypes:
            logger.info("=== Loading model (dtype=%s) ===", dtype_str)
            self.app_config.model.dtype = Precision(dtype_str)
            self.app_config.engine.gpu_memory_utilization = self.bench.gpu_memory_utilization

            engine = InferenceEngine(self.app_config)
            engine.load_model()

            try:
                for bs in self.bench.batch_sizes:
                    self.run_experiment(engine, dtype_str, bs)
            finally:
                logger.info("=== Engine shut down ===\n")

        return self.results


def run_standard_benchmarks(
    model_name: Optional[str] = None,
    max_tokens: int = 64,
    batch_sizes: Optional[list[int]] = None,
    dtypes: Optional[list[str]] = None,
    num_warmup: int = 2,
    num_iterations: int = 5,
    output_dir: str = "benchmarks/results",
) -> list[ExperimentResult]:
    """High-level entry point for running a full benchmark suite."""
    app_cfg = load_config()
    if model_name:
        app_cfg.model.model_name = model_name

    bench_cfg = BenchmarkConfig(
        batch_sizes=batch_sizes or [1, 2, 4, 8],
        dtypes=dtypes or [app_cfg.model.dtype.value],
        num_warmup=num_warmup,
        num_iterations=num_iterations,
        max_tokens=max_tokens,
        output_dir=output_dir,
    )

    runner = BenchmarkRunner(app_cfg, bench_cfg)
    results = runner.run_all()

    from benchmarks.report import ReportGenerator

    reporter = ReportGenerator(results, output_dir)
    reporter.print_report()
    reporter.save_text()
    reporter.save_json()

    return results
