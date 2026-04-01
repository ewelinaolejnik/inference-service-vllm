"""Integration tests for the FastAPI endpoints.

These tests mock the inference engine to avoid needing a GPU,
while still validating the full request/response cycle.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.inference import InferenceResult


@pytest.fixture
def client():
    """Create a test client with a mocked inference engine."""
    with patch("app.main.engine") as mock_engine:
        mock_engine.gpu_available = False
        mock_engine.get_gpu_memory_usage.return_value = {"error": "No GPU available"}
        mock_engine.generate.return_value = InferenceResult(
            output="Generated text response",
            latency_ms=50.0,
            prompt_tokens=5,
            output_tokens=8,
        )
        mock_engine.generate_batch.return_value = [
            InferenceResult(
                output=f"Response {i}",
                latency_ms=30.0,
                prompt_tokens=5,
                output_tokens=6,
            )
            for i in range(3)
        ]

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as tc:
            yield tc


class TestGenerateEndpoint:
    def test_generate_success(self, client: TestClient):
        response = client.post(
            "/generate",
            json={"prompt": "Hello, world!", "max_tokens": 50},
        )
        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        assert "latency_ms" in data
        assert data["output"] == "Generated text response"
        assert data["latency_ms"] == 50.0
        assert data["prompt_tokens"] == 5
        assert data["output_tokens"] == 8

    def test_generate_empty_prompt_rejected(self, client: TestClient):
        response = client.post("/generate", json={"prompt": ""})
        assert response.status_code == 422

    def test_generate_missing_prompt_rejected(self, client: TestClient):
        response = client.post("/generate", json={})
        assert response.status_code == 422

    def test_generate_with_optional_params(self, client: TestClient):
        response = client.post(
            "/generate",
            json={
                "prompt": "Test prompt",
                "max_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.9,
            },
        )
        assert response.status_code == 200

    def test_generate_invalid_temperature(self, client: TestClient):
        response = client.post(
            "/generate",
            json={"prompt": "Test", "temperature": 5.0},
        )
        assert response.status_code == 422


class TestBatchEndpoint:
    def test_batch_generate_success(self, client: TestClient):
        response = client.post(
            "/generate/batch",
            json={"prompts": ["Hello", "World", "Test"], "max_tokens": 32},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 3
        assert "total_latency_ms" in data

    def test_batch_generate_empty_list_rejected(self, client: TestClient):
        response = client.post(
            "/generate/batch",
            json={"prompts": []},
        )
        assert response.status_code == 422


class TestHealthEndpoint:
    def test_health(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model" in data
        assert "gpu_available" in data


class TestMetricsEndpoint:
    def test_metrics(self, client: TestClient):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "gpu_memory" in data
        assert "model" in data
        assert "dtype" in data
