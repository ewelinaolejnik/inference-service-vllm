"""Tests for the benchmark subsystem (metrics, runner, report, GPU monitor)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.config import AppConfig, load_config
from app.inference import InferenceEngine, InferenceResult
from benchmarks.gpu_monitor import GPUMonitor, GPUSnapshot, GPUStats
from benchmarks.metrics import (
    LatencyStats,
    ThroughputStats,
    compute_latency_stats,
    compute_throughput,
)
from benchmarks.report import ReportGenerator
from benchmarks.runner import BenchmarkConfig, BenchmarkRunner, ExperimentResult


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_bench_engine(app_config):
    with patch("app.inference.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        engine = InferenceEngine(app_config)

    from tests.conftest import FakeLLM

    engine._engine = FakeLLM()
    return engine


@pytest.fixture
def sample_experiment() -> ExperimentResult:
    return ExperimentResult(
        model_name="test-model",
        dtype="float16",
        batch_size=4,
        num_iterations=5,
        latency=LatencyStats(
            avg_ms=120.0, p50_ms=115.0, p95_ms=180.0, p99_ms=200.0,
            min_ms=90.0, max_ms=210.0, std_ms=30.0,
        ),
        throughput=ThroughputStats(rps=8.5, tokens_per_second=340.0),
        gpu_stats=GPUStats(
            avg_gpu_utilization_pct=75.0,
            max_gpu_utilization_pct=92.0,
            avg_memory_used_mb=4200.0,
            max_memory_used_mb=4500.0,
            memory_total_mb=8192.0,
            num_samples=10,
        ),
        total_output_tokens=1700,
    )


# ── metrics ──────────────────────────────────────────────────────────────


class TestLatencyStats:
    def test_basic(self):
        stats = compute_latency_stats([10.0, 20.0, 30.0, 40.0])
        assert stats.avg_ms == pytest.approx(25.0)
        assert stats.p50_ms == pytest.approx(25.0)
        assert stats.min_ms == pytest.approx(10.0)
        assert stats.max_ms == pytest.approx(40.0)
        assert stats.p95_ms > stats.p50_ms

    def test_empty(self):
        stats = compute_latency_stats([])
        assert stats.avg_ms == 0.0

    def test_as_dict(self):
        stats = compute_latency_stats([100.0])
        d = stats.as_dict()
        assert "avg_ms" in d
        assert "p95_ms" in d


class TestThroughputStats:
    def test_basic(self):
        tp = compute_throughput(total_requests=10, total_output_tokens=500, total_time_s=2.0)
        assert tp.rps == pytest.approx(5.0)
        assert tp.tokens_per_second == pytest.approx(250.0)

    def test_zero_time(self):
        tp = compute_throughput(10, 500, 0.0)
        assert tp.rps == 0.0
        assert tp.tokens_per_second == 0.0


# ── GPU monitor ──────────────────────────────────────────────────────────


class TestGPUStats:
    def test_as_dict(self):
        stats = GPUStats(avg_gpu_utilization_pct=80.0, max_memory_used_mb=4000.0)
        d = stats.as_dict()
        assert d["avg_gpu_util_pct"] == 80.0
        assert d["max_mem_used_mb"] == 4000.0

    def test_empty_monitor(self):
        monitor = GPUMonitor.__new__(GPUMonitor)
        monitor._snapshots = []
        result = monitor._aggregate()
        assert result.num_samples == 0

    def test_aggregate(self):
        monitor = GPUMonitor.__new__(GPUMonitor)
        monitor._snapshots = [
            GPUSnapshot(0, 80.0, 4000.0, 8000.0),
            GPUSnapshot(1, 90.0, 4500.0, 8000.0),
        ]
        stats = monitor._aggregate()
        assert stats.avg_gpu_utilization_pct == pytest.approx(85.0)
        assert stats.max_gpu_utilization_pct == pytest.approx(90.0)
        assert stats.max_memory_used_mb == pytest.approx(4500.0)
        assert stats.num_samples == 2


# ── experiment result ────────────────────────────────────────────────────


class TestExperimentResult:
    def test_as_dict(self, sample_experiment):
        d = sample_experiment.as_dict()
        assert d["model_name"] == "test-model"
        assert d["dtype"] == "float16"
        assert d["batch_size"] == 4
        assert d["avg_ms"] == pytest.approx(120.0)
        assert d["rps"] == pytest.approx(8.5)
        assert d["tokens_per_second"] == pytest.approx(340.0)
        assert d["avg_gpu_util_pct"] == pytest.approx(75.0)
        assert d["max_mem_used_mb"] == pytest.approx(4500.0)


# ── runner ───────────────────────────────────────────────────────────────


class TestBenchmarkRunner:
    def test_run_experiment(self, mock_bench_engine, app_config):
        bench_cfg = BenchmarkConfig(
            batch_sizes=[2],
            dtypes=["auto"],
            num_warmup=1,
            num_iterations=2,
            max_tokens=16,
            prompts=["Hello", "World"],
        )
        runner = BenchmarkRunner(app_config, bench_cfg)

        fake_sp = MagicMock()
        with (
            patch("benchmarks.runner.GPUMonitor") as MockMonitor,
            patch("app.inference.SamplingParams", fake_sp, create=True),
            patch.dict("sys.modules", {"vllm": MagicMock(), "vllm.SamplingParams": fake_sp}),
        ):
            mock_mon = MagicMock()
            mock_mon.stop.return_value = GPUStats()
            MockMonitor.return_value = mock_mon

            result = runner.run_experiment(mock_bench_engine, "auto", batch_size=2)

        assert result.batch_size == 2
        assert result.num_iterations == 2
        assert result.latency.avg_ms > 0
        assert result.throughput.rps >= 0


# ── report ───────────────────────────────────────────────────────────────


class TestReport:
    def test_summary_table(self, sample_experiment):
        gen = ReportGenerator([sample_experiment], output_dir="dummy")
        table = gen.summary_table()
        assert "float16" in table
        assert "120" in table
        assert "8.5" in table

    def test_insights(self, sample_experiment):
        gen = ReportGenerator([sample_experiment], output_dir="dummy")
        insights = gen.generate_insights()
        assert len(insights) >= 1

    def test_text_report(self, sample_experiment):
        gen = ReportGenerator([sample_experiment], output_dir="dummy")
        text = gen.text_report()
        assert "BENCHMARK REPORT" in text
        assert "Avg Latency" in text

    def test_save_json(self, sample_experiment, tmp_path):
        gen = ReportGenerator([sample_experiment], output_dir=str(tmp_path))
        path = gen.save_json()
        assert path.exists()
        data = json.loads(path.read_text())
        assert "experiments" in data
        assert data["experiments"][0]["dtype"] == "float16"

    def test_find_best(self, sample_experiment):
        gen = ReportGenerator([sample_experiment], output_dir="dummy")
        best = gen.find_best_config()
        assert "best_throughput" in best
        assert "best_latency" in best
