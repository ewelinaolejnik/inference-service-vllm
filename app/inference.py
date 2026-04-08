"""vLLM inference engine wrapper with performance tracking."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import torch

from app.config import AppConfig, Precision

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    output: str
    latency_ms: float
    prompt_tokens: int
    output_tokens: int


class InferenceEngine:
    """Wraps vLLM's LLM engine with latency tracking and GPU metrics."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._engine = None
        self._gpu_available = torch.cuda.is_available()

    @property
    def gpu_available(self) -> bool:
        return self._gpu_available

    def load_model(self) -> None:
        """Initialize the vLLM engine with the configured model and parameters."""
        from vllm import LLM

        dtype = self.config.model.dtype.value
        if dtype == "auto":
            dtype = "auto"

        engine_kwargs: dict = {
            "model": self.config.model.model_name,
            "dtype": dtype,
            "gpu_memory_utilization": self.config.engine.gpu_memory_utilization,
            "max_num_seqs": self.config.engine.max_num_seqs,
            "enforce_eager": self.config.engine.enforce_eager,
            "seed": self.config.engine.seed,
            "trust_remote_code": self.config.model.trust_remote_code,
        }

        if self.config.model.max_model_len is not None:
            engine_kwargs["max_model_len"] = self.config.model.max_model_len

        if self.config.engine.max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = (
                self.config.engine.max_num_batched_tokens
            )

        logger.info(
            "Loading model %s (dtype=%s, gpu_mem=%.2f)",
            self.config.model.model_name,
            dtype,
            self.config.engine.gpu_memory_utilization,
        )

        self._engine = LLM(**engine_kwargs)
        logger.info("Model loaded successfully.")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> InferenceResult:
        """Run inference on a single prompt and return the result with latency."""
        if self._engine is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_tokens or self.config.inference.max_tokens,
            temperature=temperature if temperature is not None else self.config.inference.temperature,
            top_p=top_p if top_p is not None else self.config.inference.top_p,
        )

        start = time.perf_counter()
        outputs = self._engine.generate([prompt], params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        result = outputs[0]
        generated_text = result.outputs[0].text
        prompt_tokens = len(result.prompt_token_ids)
        output_tokens = len(result.outputs[0].token_ids)

        return InferenceResult(
            output=generated_text,
            latency_ms=round(elapsed_ms, 2),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> list[InferenceResult]:
        """Run batched inference on multiple prompts."""
        if self._engine is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_tokens or self.config.inference.max_tokens,
            temperature=temperature if temperature is not None else self.config.inference.temperature,
            top_p=top_p if top_p is not None else self.config.inference.top_p,
        )

        start = time.perf_counter()
        outputs = self._engine.generate(prompts, params)
        total_elapsed_ms = (time.perf_counter() - start) * 1000

        results = []
        per_request_ms = total_elapsed_ms / len(outputs) if outputs else 0

        for result in outputs:
            generated_text = result.outputs[0].text
            prompt_tokens = len(result.prompt_token_ids)
            output_tokens = len(result.outputs[0].token_ids)
            results.append(
                InferenceResult(
                    output=generated_text,
                    latency_ms=round(per_request_ms, 2),
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                )
            )

        return results

    def get_gpu_memory_usage(self) -> dict:
        """Return current GPU memory usage statistics."""
        if not self._gpu_available:
            return {"error": "No GPU available"}

        free, total = torch.cuda.mem_get_info(0)
        used = total - free

        return {
            "total_mb": round(total / 1024**2, 1),
            "used_mb": round(used / 1024**2, 1),
            "free_mb": round(free / 1024**2, 1),
            "utilization_pct": round(used / total * 100, 1) if total > 0 else 0,
        }
