"""Benchmark runner for measuring inference latency and throughput.

Supports comparing:
- Single vs batch requests
- Different model sizes
- FP32 vs FP16 precision
- Various engine configuration knobs
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from app.config import AppConfig, Precision, load_config
from app.inference import InferenceEngine

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent
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
class BenchmarkResult:
    label: str
    model_name: str
    dtype: str
    num_requests: int
    batch_size: int
    max_tokens: int
    gpu_memory_utilization: float
    max_num_batched_tokens: Optional[int]
    latencies_ms: list[float] = field(default_factory=list)
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    total_time_s: float = 0.0
    gpu_memory: dict = field(default_factory=dict)

    def compute_stats(self) -> None:
        if not self.latencies_ms:
            return
        self.mean_latency_ms = round(statistics.mean(self.latencies_ms), 2)
        self.median_latency_ms = round(statistics.median(self.latencies_ms), 2)
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.95)
        self.p95_latency_ms = round(sorted_lat[min(idx, len(sorted_lat) - 1)], 2)
        if self.total_time_s > 0:
            self.throughput_rps = round(self.num_requests / self.total_time_s, 2)


def _make_result(
    label: str, config: AppConfig, num_requests: int, batch_size: int, max_tokens: int
) -> BenchmarkResult:
    return BenchmarkResult(
        label=label,
        model_name=config.model.model_name,
        dtype=config.model.dtype.value,
        num_requests=num_requests,
        batch_size=batch_size,
        max_tokens=max_tokens,
        gpu_memory_utilization=config.engine.gpu_memory_utilization,
        max_num_batched_tokens=config.engine.max_num_batched_tokens,
    )


def benchmark_single_requests(
    engine: InferenceEngine,
    config: AppConfig,
    prompts: Optional[list[str]] = None,
    max_tokens: int = 64,
) -> BenchmarkResult:
    """Send prompts one at a time and record per-request latency."""
    prompts = prompts or DEFAULT_PROMPTS
    result = _make_result(
        "single_request", config, len(prompts), batch_size=1, max_tokens=max_tokens
    )

    overall_start = time.perf_counter()
    for prompt in prompts:
        out = engine.generate(prompt, max_tokens=max_tokens, temperature=0.0)
        result.latencies_ms.append(out.latency_ms)
    result.total_time_s = round(time.perf_counter() - overall_start, 3)

    result.gpu_memory = engine.get_gpu_memory_usage()
    result.compute_stats()
    return result


def benchmark_batch_requests(
    engine: InferenceEngine,
    config: AppConfig,
    prompts: Optional[list[str]] = None,
    max_tokens: int = 64,
    batch_size: int = 4,
) -> BenchmarkResult:
    """Send prompts in batches and record latency per batch."""
    prompts = prompts or DEFAULT_PROMPTS
    result = _make_result(
        "batch_request", config, len(prompts), batch_size=batch_size, max_tokens=max_tokens
    )

    overall_start = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        start = time.perf_counter()
        engine.generate_batch(batch, max_tokens=max_tokens, temperature=0.0)
        batch_ms = (time.perf_counter() - start) * 1000
        result.latencies_ms.append(round(batch_ms, 2))
    result.total_time_s = round(time.perf_counter() - overall_start, 3)

    result.gpu_memory = engine.get_gpu_memory_usage()
    result.compute_stats()
    return result


def benchmark_precision(
    model_name: str,
    precisions: Optional[list[Precision]] = None,
    prompts: Optional[list[str]] = None,
    max_tokens: int = 64,
) -> list[BenchmarkResult]:
    """Compare inference across different precision modes."""
    precisions = precisions or [Precision.FP32, Precision.FP16]
    prompts = prompts or DEFAULT_PROMPTS[:4]
    results = []

    for prec in precisions:
        logger.info("Benchmarking precision: %s", prec.value)
        cfg = load_config()
        cfg.model.model_name = model_name
        cfg.model.dtype = prec

        eng = InferenceEngine(cfg)
        eng.load_model()
        res = benchmark_single_requests(eng, cfg, prompts, max_tokens)
        res.label = f"precision_{prec.value}"
        results.append(res)

    return results


def benchmark_config_tuning(
    model_name: str,
    gpu_memory_values: Optional[list[float]] = None,
    batched_token_values: Optional[list[int]] = None,
    prompts: Optional[list[str]] = None,
    max_tokens: int = 64,
) -> list[BenchmarkResult]:
    """Experiment with different engine configurations."""
    gpu_memory_values = gpu_memory_values or [0.85, 0.90, 0.95]
    prompts = prompts or DEFAULT_PROMPTS[:4]
    results = []

    for gpu_mem in gpu_memory_values:
        logger.info("Benchmarking gpu_memory_utilization=%.2f", gpu_mem)
        cfg = load_config()
        cfg.model.model_name = model_name
        cfg.engine.gpu_memory_utilization = gpu_mem

        eng = InferenceEngine(cfg)
        eng.load_model()
        res = benchmark_single_requests(eng, cfg, prompts, max_tokens)
        res.label = f"gpu_mem_{gpu_mem}"
        results.append(res)

    if batched_token_values:
        for bt in batched_token_values:
            logger.info("Benchmarking max_num_batched_tokens=%d", bt)
            cfg = load_config()
            cfg.model.model_name = model_name
            cfg.engine.max_num_batched_tokens = bt

            eng = InferenceEngine(cfg)
            eng.load_model()
            res = benchmark_batch_requests(eng, cfg, prompts, max_tokens, batch_size=4)
            res.label = f"batched_tokens_{bt}"
            results.append(res)

    return results


def save_results(results: list[BenchmarkResult], filename: str = "results.json") -> Path:
    """Persist benchmark results to JSON."""
    path = RESULTS_DIR / filename
    data = [asdict(r) for r in results]
    path.write_text(json.dumps(data, indent=2))
    logger.info("Results saved to %s", path)
    return path


def run_standard_benchmarks(
    model_name: Optional[str] = None,
    max_tokens: int = 64,
) -> list[BenchmarkResult]:
    """Run single + batch benchmarks with the default configuration."""
    cfg = load_config()
    if model_name:
        cfg.model.model_name = model_name

    engine = InferenceEngine(cfg)
    engine.load_model()

    results = []
    logger.info("Running single-request benchmark...")
    results.append(benchmark_single_requests(engine, cfg, max_tokens=max_tokens))

    for batch_size in [2, 4, 8]:
        logger.info("Running batch benchmark (batch_size=%d)...", batch_size)
        results.append(
            benchmark_batch_requests(
                engine, cfg, max_tokens=max_tokens, batch_size=batch_size
            )
        )

    save_results(results)
    return results
