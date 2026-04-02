# Optimizing LLM Inference with vLLM on Local GPU

A high-performance local inference service built with **vLLM** and **FastAPI**, designed to demonstrate measurable GPU inference optimizations for large language models.

## What is vLLM?

[vLLM](https://github.com/vllm-project/vllm) is an open-source library for fast LLM inference and serving. It uses **PagedAttention** to efficiently manage attention key/value memory, enabling high throughput and low latency compared to naive HuggingFace generation pipelines.

## Project Goal

Build a production-style inference API on a local GPU and systematically optimize latency and throughput using:

- **Continuous batching** -- vLLM's scheduler batches incoming requests automatically
- **Precision tuning** -- FP32 vs FP16 vs BF16 comparison
- **Configuration knobs** -- `gpu_memory_utilization`, `max_num_batched_tokens`, `max_num_seqs`
- **Model size comparison** -- smaller vs larger variants (e.g., OPT-1.3B vs OPT-6.7B)

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 13.x drivers
- ~8 GB VRAM for OPT-1.3B (more for larger models)

### Local Installation

```bash
git clone https://github.com/your-username/inference-service-vllm.git
cd inference-service-vllm

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
cp .env.example .env
```

### Docker

```bash
docker compose up --build
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Usage

### Start the API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### `POST /generate` -- Single Inference

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing in simple terms", "max_tokens": 100}'
```

Response:

```json
{
  "output": "Quantum computing uses quantum bits...",
  "latency_ms": 85.42,
  "prompt_tokens": 7,
  "output_tokens": 100
}
```

#### `POST /generate/batch` -- Batch Inference

```bash
curl -X POST http://localhost:8000/generate/batch \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello", "What is AI?", "Write a poem"], "max_tokens": 64}'
```

#### `GET /health` -- Health Check

```bash
curl http://localhost:8000/health
```

#### `GET /metrics` -- GPU & Config Metrics

```bash
curl http://localhost:8000/metrics
```

### CLI Tool

```bash
python run_inference.py --prompt "What is machine learning?" --max-tokens 100
python run_inference.py --benchmark
python run_inference.py --prompt "Hello" --model facebook/opt-350m
```

## Running Tests

```bash
pytest
pytest -v
pytest tests/test_api.py -v
```

Tests use mocked vLLM backends, so they run without a GPU.

## Environment Variables

All configuration is environment-driven (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `facebook/opt-1.3b` | HuggingFace model ID |
| `MODEL_DTYPE` | `auto` | Precision: `float32`, `float16`, `bfloat16`, `auto` |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of GPU memory for KV-cache |
| `MAX_NUM_SEQS` | `64` | Max concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `0` (auto) | Max tokens per batch iteration |
| `ENFORCE_EAGER` | `false` | Disable CUDA graphs (for debugging) |
