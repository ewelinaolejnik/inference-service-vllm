"""FastAPI inference service powered by vLLM."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import load_config
from app.inference import InferenceEngine

logger = logging.getLogger(__name__)

config = load_config()
engine = InferenceEngine(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; clean up on shutdown."""
    logger.info("Starting model loading...")
    engine.load_model()
    logger.info("Model ready for inference.")
    yield
    logger.info("Shutting down inference service.")


app = FastAPI(
    title="vLLM Inference Service",
    description="High-performance LLM inference API backed by vLLM",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ───────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input text for generation")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    output: str
    latency_ms: float
    prompt_tokens: int
    output_tokens: int


class BatchGenerateRequest(BaseModel):
    prompts: list[str] = Field(..., min_length=1)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)


class BatchGenerateResponse(BaseModel):
    results: list[GenerateResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    gpu_available: bool


# ── Endpoints ────────────────────────────────────────────────────────────


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from a single prompt."""
    try:
        result = engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        return GenerateResponse(
            output=result.output,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            output_tokens=result.output_tokens,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


@app.post("/generate/batch", response_model=BatchGenerateResponse)
async def generate_batch(request: BatchGenerateRequest):
    """Generate text from multiple prompts in a single batch."""
    try:
        start = time.perf_counter()
        results = engine.generate_batch(
            prompts=request.prompts,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        total_ms = (time.perf_counter() - start) * 1000

        return BatchGenerateResponse(
            results=[
                GenerateResponse(
                    output=r.output,
                    latency_ms=r.latency_ms,
                    prompt_tokens=r.prompt_tokens,
                    output_tokens=r.output_tokens,
                )
                for r in results
            ],
            total_latency_ms=round(total_ms, 2),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Batch inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model=config.model.model_name,
        gpu_available=engine.gpu_available,
    )


@app.get("/metrics")
async def metrics():
    """Return GPU memory and configuration metrics."""
    gpu_mem = engine.get_gpu_memory_usage()
    return {
        "gpu_memory": gpu_mem,
        "model": config.model.model_name,
        "dtype": config.model.dtype.value,
        "gpu_memory_utilization": config.engine.gpu_memory_utilization,
        "max_num_seqs": config.engine.max_num_seqs,
        "max_num_batched_tokens": config.engine.max_num_batched_tokens,
    }
