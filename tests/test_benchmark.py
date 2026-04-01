"""Tests for the benchmark runner (mocked engine, no GPU needed)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.config import AppConfig, load_config
from app.inference import InferenceEngine, InferenceResult
from benchmarks.runner import (
    BenchmarkResult,
    benchmark_batch_requests,
    benchmark_single_requests,
    save_results,
)


@pytest.fixture
def mock_bench_engine(app_config):
    """InferenceEngine with mocked backend for benchmarking tests."""
    with patch("app.inference.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        engine = InferenceEngine(app_config)

    from tests.conftest import FakeLLM

    engine._engine = FakeLLM()
    return engine


class TestBenchmarkResult:
    def test_compute_stats(self):
        br = BenchmarkResult(
            label="test",
            model_name="test-model",
            dtype="float16",
            num_requests=4,
            batch_size=1,
            max_tokens=64,
            gpu_memory_utilization=0.9,
            max_num_batched_tokens=None,
            latencies_ms=[10.0, 20.0, 30.0, 40.0],
            total_time_s=1.0,
        )
        br.compute_stats()
        assert br.mean_latency_ms == 25.0
        assert br.median_latency_ms == 25.0
        assert br.throughput_rps == 4.0
        assert br.p95_latency_ms > 0

    def test_compute_stats_empty(self):
        br = BenchmarkResult(
            label="empty",
            model_name="m",
            dtype="auto",
            num_requests=0,
            batch_size=1,
            max_tokens=64,
            gpu_memory_utilization=0.9,
            max_num_batched_tokens=None,
        )
        br.compute_stats()
        assert br.mean_latency_ms == 0.0


class TestSingleBenchmark:
    def test_runs(self, mock_bench_engine, app_config):
        result = benchmark_single_requests(
            mock_bench_engine, app_config, prompts=["Hello", "World"], max_tokens=16
        )
        assert result.label == "single_request"
        assert result.num_requests == 2
        assert len(result.latencies_ms) == 2
        assert result.mean_latency_ms > 0
        assert result.total_time_s > 0


class TestBatchBenchmark:
    def test_runs(self, mock_bench_engine, app_config):
        result = benchmark_batch_requests(
            mock_bench_engine,
            app_config,
            prompts=["A", "B", "C", "D"],
            max_tokens=16,
            batch_size=2,
        )
        assert result.label == "batch_request"
        assert result.num_requests == 4
        assert result.batch_size == 2
        assert len(result.latencies_ms) == 2


class TestSaveResults:
    def test_save_and_load(self, tmp_path):
        br = BenchmarkResult(
            label="save_test",
            model_name="model",
            dtype="auto",
            num_requests=1,
            batch_size=1,
            max_tokens=64,
            gpu_memory_utilization=0.9,
            max_num_batched_tokens=None,
            latencies_ms=[10.0],
            total_time_s=0.5,
        )
        br.compute_stats()

        with patch("benchmarks.runner.RESULTS_DIR", tmp_path):
            path = save_results([br], "test_results.json")

        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["label"] == "save_test"
