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

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
cp .env.example .env   # configure model, dtype, etc.
docker compose up --build inference
```

#### Available Docker services

| Command | Description |
|---------|-------------|
| `docker compose up inference` | Start the API server on port 8080 |
| `docker compose run --rm benchmark` | Run the full benchmark suite |
| `docker compose run --rm quick-test` | Single-prompt smoke test |
| `docker compose run --rm test` | Run unit tests (no GPU required) |

## Usage

### Start the API Server

```bash
# Local
uvicorn app.main:app --host 0.0.0.0 --port 8080

# Docker
docker compose up inference
```

### API Endpoints

#### `POST /generate` -- Single Inference

```bash
curl -X POST http://localhost:8080/generate \
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
curl -X POST http://localhost:8080/generate/batch \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello", "What is AI?", "Write a poem"], "max_tokens": 64}'
```

#### `GET /health` -- Health Check

```bash
curl http://localhost:8080/health
```

#### `GET /metrics` -- GPU & Config Metrics

```bash
curl http://localhost:8080/metrics
```

### CLI Tool

```bash
python run_inference.py --prompt "What is machine learning?" --max-tokens 100
python run_inference.py --prompt "Hello" --model facebook/opt-350m
python run_inference.py --benchmark
```

### Benchmarking

The benchmark suite measures inference performance across batch sizes and data types, collecting:

| Metric | Description |
|--------|-------------|
| Avg Latency (ms) | Mean end-to-end latency per batch |
| P50 / P95 (ms) | Median and 95th-percentile latency |
| RPS | Requests (batches) processed per second |
| Tokens/s | Output tokens generated per second |
| GPU Util % | Average GPU utilization during the run |
| Mem Used (MB) | Peak GPU memory consumption |

#### Run locally

```bash
python run_inference.py --benchmark \
  --batch-sizes 1,2,4,8 \
  --dtypes float16 \
  --iterations 5 \
  --warmup 2 \
  --output-dir benchmarks/results
```

#### Run with Docker

```bash
docker compose run --rm benchmark
```

Override defaults via environment variables:

```bash
BENCHMARK_DTYPES=float16,float32 BENCHMARK_ITERATIONS=10 docker compose run --rm benchmark
```

Or pass arguments directly:

```bash
docker compose run --rm benchmark \
  --benchmark --batch-sizes 1,4,16 --dtypes float16 --iterations 20 --warmup 3
```

Results (JSON and text reports) are saved to `benchmarks/results/` on the host via a volume mount.

| CLI Flag | Default | Description |
|----------|---------|-------------|
| `--batch-sizes` | `1,2,4,8` | Comma-separated batch sizes |
| `--dtypes` | `float16` | Comma-separated dtypes to test |
| `--iterations` | `5` | Measured iterations per experiment |
| `--warmup` | `2` | Warmup iterations (discarded) |
| `--output-dir` | `benchmarks/results` | Output directory |

## Running Tests

```bash
# Local
pytest -v

# Docker (no GPU needed)
docker compose run --rm test
```

Tests use mocked vLLM backends, so they run without a GPU.

## Environment Variables

All configuration is environment-driven (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `facebook/opt-1.3b` | HuggingFace model ID |
| `MODEL_DTYPE` | `auto` | Precision: `float32`, `float16`, `bfloat16`, `auto` |
| `GPU_MEMORY_UTILIZATION` | `0.80` | Fraction of GPU memory for KV-cache |
| `MAX_NUM_SEQS` | `64` | Max concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `0` (auto) | Max tokens per batch iteration |
| `ENFORCE_EAGER` | `false` | Disable CUDA graphs (for debugging) |
