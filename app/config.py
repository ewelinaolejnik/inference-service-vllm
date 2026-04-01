"""Centralized configuration for the vLLM inference service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Precision(str, Enum):
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    AUTO = "auto"


@dataclass
class ModelConfig:
    model_name: str = os.getenv("MODEL_NAME", "facebook/opt-1.3b")
    dtype: Precision = Precision(os.getenv("MODEL_DTYPE", "auto"))
    max_model_len: Optional[int] = int(os.getenv("MAX_MODEL_LEN", "0")) or None
    trust_remote_code: bool = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"


@dataclass
class EngineConfig:
    gpu_memory_utilization: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.80"))
    max_num_batched_tokens: Optional[int] = (
        int(os.getenv("MAX_NUM_BATCHED_TOKENS", "0")) or None
    )
    max_num_seqs: int = int(os.getenv("MAX_NUM_SEQS", "64"))
    enforce_eager: bool = os.getenv("ENFORCE_EAGER", "false").lower() == "true"
    seed: int = int(os.getenv("ENGINE_SEED", "42"))


@dataclass
class ServerConfig:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")


@dataclass
class InferenceDefaults:
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    inference: InferenceDefaults = field(default_factory=InferenceDefaults)


def load_config() -> AppConfig:
    """Load application configuration from environment variables."""
    return AppConfig()
