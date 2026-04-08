"""Rich benchmark reporting: tables, metric definitions, insights, and export."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from tabulate import tabulate

if TYPE_CHECKING:
    from benchmarks.runner import ExperimentResult

logger = logging.getLogger(__name__)

METRIC_DEFINITIONS = [
    "Avg Latency (ms): Average time to process one batch. Lower is better for responsiveness.",
    "P50 (ms): Median latency (50% of measurements are at or below this value). Represents typical latency.",
    "P95 (ms): 95th percentile latency (95% of measurements are at or below this value). Captures tail latency.",
    "RPS: Requests per second, i.e. how many requests the system handles each second. Higher is better.",
    "Tokens/s: Number of generated tokens per second. Main throughput metric for inference.",
    "GPU Util %: Average GPU utilization during the run. Higher values can indicate better compute saturation.",
    "Mem Used (MB): Peak GPU memory usage. Helps estimate OOM risk and configuration cost.",
    "Batch: Number of prompts processed at the same time. Larger batches usually improve throughput at the cost of latency.",
    "DType: Numeric precision (e.g. float16/float32). Lower precision often improves speed and memory efficiency.",
]


class ReportGenerator:

    def __init__(
        self,
        results: list[ExperimentResult],
        output_dir: str = "benchmarks/results",
    ):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── table ────────────────────────────────────────────────────────────

    def summary_table(self) -> str:
        headers = [
            "DType", "Batch", "Avg Latency (ms)",
            "P50 (ms)", "P95 (ms)", "RPS", "Tokens/s",
            "GPU Util %", "Mem Used (MB)",
        ]
        rows = []
        for r in self.results:
            rows.append([
                r.dtype,
                r.batch_size,
                f"{r.latency.avg_ms:.1f}",
                f"{r.latency.p50_ms:.1f}",
                f"{r.latency.p95_ms:.1f}",
                f"{r.throughput.rps:.2f}",
                f"{r.throughput.tokens_per_second:.1f}",
                f"{r.gpu_stats.avg_gpu_utilization_pct:.1f}",
                f"{r.gpu_stats.max_memory_used_mb:.0f}",
            ])
        return tabulate(rows, headers=headers, tablefmt="grid")

    # ── best configurations ──────────────────────────────────────────────

    def find_best_config(self) -> dict:
        if not self.results:
            return {}
        best_tp = max(self.results, key=lambda r: r.throughput.rps)
        best_lat = min(self.results, key=lambda r: r.latency.avg_ms)
        best_eff = max(
            self.results,
            key=lambda r: r.throughput.rps / max(r.gpu_stats.max_memory_used_mb, 1),
        )
        return {
            "best_throughput": {
                "dtype": best_tp.dtype,
                "batch_size": best_tp.batch_size,
                "rps": round(best_tp.throughput.rps, 2),
            },
            "best_latency": {
                "dtype": best_lat.dtype,
                "batch_size": best_lat.batch_size,
                "avg_ms": round(best_lat.latency.avg_ms, 2),
            },
            "best_memory_efficiency": {
                "dtype": best_eff.dtype,
                "batch_size": best_eff.batch_size,
            },
        }

    # ── insights ─────────────────────────────────────────────────────────

    def generate_insights(self) -> list[str]:
        if not self.results:
            return ["No results to analyze."]

        insights: list[str] = []
        best = self.find_best_config()

        sorted_by_batch = sorted(self.results, key=lambda r: r.batch_size)
        if len(sorted_by_batch) >= 2:
            first, last = sorted_by_batch[0], sorted_by_batch[-1]
            if first.throughput.rps > 0:
                speedup = last.throughput.rps / first.throughput.rps
                insights.append(
                    f"Increasing batch size from {first.batch_size} to "
                    f"{last.batch_size} changed throughput by {speedup:.1f}x "
                    f"({first.throughput.rps:.1f} -> {last.throughput.rps:.1f} RPS)."
                )

        fp16 = [r for r in self.results if r.dtype == "float16"]
        fp32 = [r for r in self.results if r.dtype == "float32"]
        if fp16 and fp32:
            avg16 = sum(r.gpu_stats.max_memory_used_mb for r in fp16) / len(fp16)
            avg32 = sum(r.gpu_stats.max_memory_used_mb for r in fp32) / len(fp32)
            if avg32 > 0:
                reduction = (1 - avg16 / avg32) * 100
                insights.append(
                    f"FP16 reduced GPU memory by ~{reduction:.0f}% vs FP32 "
                    f"({avg16:.0f} MB vs {avg32:.0f} MB average peak)."
                )

        bt = best.get("best_throughput", {})
        bl = best.get("best_latency", {})
        if bt:
            insights.append(
                f"Best throughput: dtype={bt['dtype']}, "
                f"batch_size={bt['batch_size']} ({bt['rps']} RPS)."
            )
        if bl:
            insights.append(
                f"Best latency: dtype={bl['dtype']}, "
                f"batch_size={bl['batch_size']} ({bl['avg_ms']} ms avg)."
            )
        return insights

    # ── text report ──────────────────────────────────────────────────────

    def text_report(self) -> str:
        sep = "=" * 72
        lines = [
            sep,
            "  vLLM INFERENCE BENCHMARK REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Model: {self.results[0].model_name}" if self.results else "",
            sep,
            "",
            "PERFORMANCE COMPARISON",
            "-" * 72,
            self.summary_table(),
            "",
            "METRIC DEFINITIONS",
            "-" * 72,
        ]
        for i, defn in enumerate(METRIC_DEFINITIONS, 1):
            lines.append(f"  {i}. {defn}")

        lines += [
            "",
            "RECOMMENDED CONFIGURATIONS",
            "-" * 72,
        ]
        for cat, info in self.find_best_config().items():
            lines.append(f"  {cat.replace('_', ' ').title()}: {info}")

        lines += [
            "",
            "KEY INSIGHTS",
            "-" * 72,
        ]
        for i, insight in enumerate(self.generate_insights(), 1):
            lines.append(f"  {i}. {insight}")

        lines += ["", sep]
        return "\n".join(lines)

    # ── persistence ──────────────────────────────────────────────────────

    def save_json(self, filename: str = "results.json") -> Path:
        path = self.output_dir / filename
        data = {
            "timestamp": datetime.now().isoformat(),
            "experiments": [r.as_dict() for r in self.results],
            "best_configs": self.find_best_config(),
            "insights": self.generate_insights(),
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("JSON results saved to %s", path)
        return path

    def save_text(self, filename: str = "report.txt") -> Path:
        path = self.output_dir / filename
        path.write_text(self.text_report())
        logger.info("Text report saved to %s", path)
        return path

    def print_report(self) -> None:
        print(self.text_report())
