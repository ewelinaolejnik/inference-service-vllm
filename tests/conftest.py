"""Shared fixtures for the test suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from app.config import AppConfig, load_config
from app.inference import InferenceEngine, InferenceResult


# ── Fake vLLM objects for unit testing without a GPU ─────────────────────


@dataclass
class FakeTokenOutput:
    text: str = "This is generated text."
    token_ids: list = None

    def __post_init__(self):
        if self.token_ids is None:
            self.token_ids = list(range(10))


@dataclass
class FakeRequestOutput:
    prompt_token_ids: list = None
    outputs: list = None

    def __post_init__(self):
        if self.prompt_token_ids is None:
            self.prompt_token_ids = list(range(5))
        if self.outputs is None:
            self.outputs = [FakeTokenOutput()]


class FakeLLM:
    """Mimics vllm.LLM for testing without GPU."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, prompts: list[str], sampling_params) -> list:
        return [FakeRequestOutput() for _ in prompts]


@pytest.fixture
def app_config() -> AppConfig:
    return load_config()


@pytest.fixture
def mock_engine(app_config: AppConfig) -> InferenceEngine:
    """Return an InferenceEngine with a mocked vLLM backend."""
    with patch("app.inference.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        engine = InferenceEngine(app_config)

    engine._engine = FakeLLM()
    return engine


@pytest.fixture
def sample_result() -> InferenceResult:
    return InferenceResult(
        output="Hello world",
        latency_ms=42.0,
        prompt_tokens=5,
        output_tokens=10,
    )
