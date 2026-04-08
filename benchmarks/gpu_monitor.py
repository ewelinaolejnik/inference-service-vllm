"""Background GPU monitoring via pynvml or nvidia-smi fallback."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUSnapshot:
    timestamp: float
    gpu_utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float = 0.0
    power_draw_w: float = 0.0

    @property
    def memory_utilization_pct(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100


@dataclass
class GPUStats:
    avg_gpu_utilization_pct: float = 0.0
    max_gpu_utilization_pct: float = 0.0
    avg_memory_used_mb: float = 0.0
    max_memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    avg_temperature_c: float = 0.0
    avg_power_draw_w: float = 0.0
    num_samples: int = 0

    def as_dict(self) -> dict:
        return {
            "avg_gpu_util_pct": round(self.avg_gpu_utilization_pct, 1),
            "max_gpu_util_pct": round(self.max_gpu_utilization_pct, 1),
            "avg_mem_used_mb": round(self.avg_memory_used_mb, 0),
            "max_mem_used_mb": round(self.max_memory_used_mb, 0),
            "mem_total_mb": round(self.memory_total_mb, 0),
            "avg_temp_c": round(self.avg_temperature_c, 1),
            "avg_power_w": round(self.avg_power_draw_w, 1),
            "num_samples": self.num_samples,
        }


class GPUMonitor:
    """Samples GPU metrics on a background thread while benchmarks run."""

    def __init__(self, interval: float = 0.5, gpu_index: int = 0):
        self._interval = interval
        self._gpu_index = gpu_index
        self._snapshots: list[GPUSnapshot] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._use_pynvml = False
        self._handle = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            self._use_pynvml = True
        except Exception:
            logger.debug("pynvml unavailable, falling back to nvidia-smi")

    def _sample_pynvml(self) -> GPUSnapshot:
        import pynvml

        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            temp = 0.0
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception:
            power = 0.0

        return GPUSnapshot(
            timestamp=time.time(),
            gpu_utilization_pct=float(util.gpu),
            memory_used_mb=mem.used / (1024**2),
            memory_total_mb=mem.total / (1024**2),
            temperature_c=float(temp),
            power_draw_w=float(power),
        )

    def _sample_nvidia_smi(self) -> GPUSnapshot | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self._gpu_index}",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,"
                    "temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return GPUSnapshot(
                timestamp=time.time(),
                gpu_utilization_pct=float(parts[0]),
                memory_used_mb=float(parts[1]),
                memory_total_mb=float(parts[2]),
                temperature_c=float(parts[3]) if len(parts) > 3 else 0.0,
                power_draw_w=float(parts[4]) if len(parts) > 4 else 0.0,
            )
        except Exception as exc:
            logger.debug("nvidia-smi sample failed: %s", exc)
            return None

    def _sample(self) -> GPUSnapshot | None:
        if self._use_pynvml:
            return self._sample_pynvml()
        return self._sample_nvidia_smi()

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            snapshot = self._sample()
            if snapshot is not None:
                self._snapshots.append(snapshot)
            self._stop_event.wait(self._interval)

    def start(self) -> None:
        self._snapshots.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> GPUStats:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        return self._aggregate()

    def _aggregate(self) -> GPUStats:
        if not self._snapshots:
            return GPUStats()
        snaps = self._snapshots
        return GPUStats(
            avg_gpu_utilization_pct=sum(s.gpu_utilization_pct for s in snaps) / len(snaps),
            max_gpu_utilization_pct=max(s.gpu_utilization_pct for s in snaps),
            avg_memory_used_mb=sum(s.memory_used_mb for s in snaps) / len(snaps),
            max_memory_used_mb=max(s.memory_used_mb for s in snaps),
            memory_total_mb=snaps[0].memory_total_mb,
            avg_temperature_c=sum(s.temperature_c for s in snaps) / len(snaps),
            avg_power_draw_w=sum(s.power_draw_w for s in snaps) / len(snaps),
            num_samples=len(snaps),
        )

    @property
    def snapshots(self) -> list[GPUSnapshot]:
        return list(self._snapshots)
