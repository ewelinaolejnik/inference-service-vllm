"""Unit tests for the inference engine."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.config import AppConfig, Precision, load_config
from app.inference import InferenceEngine, InferenceResult


class TestInferenceResult:
    def test_fields(self, sample_result: InferenceResult):
        assert sample_result.output == "Hello world"
        assert sample_result.latency_ms == 42.0
        assert sample_result.prompt_tokens == 5
        assert sample_result.output_tokens == 10


class TestInferenceEngineInit:
    def test_engine_created(self, app_config: AppConfig):
        with patch("app.inference.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            engine = InferenceEngine(app_config)

        assert engine._engine is None
        assert engine.gpu_available is False

    def test_generate_without_load_raises(self, app_config: AppConfig):
        with patch("app.inference.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            engine = InferenceEngine(app_config)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.generate("hello")


class TestInferenceEngineGenerate:
    def test_single_generate(self, mock_engine: InferenceEngine):
        result = mock_engine.generate("What is AI?", max_tokens=32)
        assert isinstance(result, InferenceResult)
        assert result.output == "This is generated text."
        assert result.latency_ms > 0
        assert result.prompt_tokens == 5
        assert result.output_tokens == 10

    def test_batch_generate(self, mock_engine: InferenceEngine):
        prompts = ["Hello", "World", "Test"]
        results = mock_engine.generate_batch(prompts, max_tokens=32)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, InferenceResult)
            assert r.output == "This is generated text."

    def test_batch_generate_without_load_raises(self, app_config: AppConfig):
        with patch("app.inference.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            engine = InferenceEngine(app_config)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.generate_batch(["hello"])


class TestGPUMemory:
    def test_no_gpu_returns_error(self, mock_engine: InferenceEngine):
        info = mock_engine.get_gpu_memory_usage()
        assert "error" in info


class TestModelConfig:
    def test_default_config(self):
        config = load_config()
        assert config.model.model_name == "facebook/opt-1.3b"
        assert config.engine.gpu_memory_utilization == 0.90
        assert config.inference.max_tokens == 128

    def test_precision_enum(self):
        assert Precision.FP16.value == "float16"
        assert Precision.FP32.value == "float32"
        assert Precision.BF16.value == "bfloat16"
