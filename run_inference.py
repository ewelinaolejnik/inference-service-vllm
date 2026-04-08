#!/usr/bin/env python3
"""CLI tool for running inference and benchmarks.

Usage:
    python run_inference.py --prompt "Tell me about AI"
    python run_inference.py --prompt "Hello" --max-tokens 50
    python run_inference.py --benchmark
    python run_inference.py --benchmark --model facebook/opt-1.3b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from app.config import load_config
from app.inference import InferenceEngine


def run_single(args: argparse.Namespace) -> None:
    config = load_config()
    if args.model:
        config.model.model_name = args.model

    engine = InferenceEngine(config)
    engine.load_model()

    result = engine.generate(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(f"\n{'='*60}")
    print(f"Model:    {config.model.model_name}")
    print(f"Prompt:   {args.prompt}")
    print(f"{'='*60}")
    print(f"\nOutput:\n{result.output}\n")
    print(f"--- Performance ---")
    print(f"Latency:       {result.latency_ms:.2f} ms")
    print(f"Prompt tokens: {result.prompt_tokens}")
    print(f"Output tokens: {result.output_tokens}")

    gpu_mem = engine.get_gpu_memory_usage()
    if "error" not in gpu_mem:
        print(f"\n--- GPU Memory ---")
        print(f"Used:  {gpu_mem['used_mb']:.1f} MB")
        print(f"Free:  {gpu_mem['free_mb']:.1f} MB")
        print(f"Total: {gpu_mem['total_mb']:.1f} MB")
        print(f"Util:  {gpu_mem['utilization_pct']:.1f}%")


def run_benchmark(args: argparse.Namespace) -> None:
    from benchmarks.runner import run_standard_benchmarks

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dtypes = [d.strip() for d in args.dtypes.split(",")] if args.dtypes else None
    batch_sizes = (
        [int(x) for x in args.batch_sizes.split(",")]
        if args.batch_sizes
        else None
    )

    run_standard_benchmarks(
        model_name=args.model,
        max_tokens=args.max_tokens,
        batch_sizes=batch_sizes,
        dtypes=dtypes,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        output_dir=args.output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="vLLM Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--prompt", type=str, help="Prompt text for inference")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--benchmark", action="store_true", help="Run standard benchmarks")
    parser.add_argument("--batch-sizes", type=str, default=None, help="Comma-separated batch sizes (e.g. 1,2,4,8)")
    parser.add_argument("--dtypes", type=str, default=None, help="Comma-separated dtypes (e.g. float16,float32)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations (discarded)")
    parser.add_argument("--iterations", type=int, default=5, help="Measured iterations per experiment")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory for results")
    args = parser.parse_args()

    if not args.prompt and not args.benchmark:
        parser.error("Provide --prompt for inference or --benchmark to run benchmarks.")

    if args.benchmark:
        run_benchmark(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
