"""Debug wall timings synchronize CUDA at stage boundaries, never when disabled."""

import time
from contextlib import contextmanager

import numpy as np
import torch

from .ranges import range_scope


def statistics(values):
    array = np.asarray(values, dtype=float)
    return (
        {
            "count": len(values),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "total": float(array.sum()),
        }
        if len(array)
        else {"count": 0, "mean": None, "median": None, "p95": None, "total": 0.0}
    )


class BatchTimings:
    def __init__(self, device, enabled=False):
        self.device, self.enabled = torch.device(device), enabled
        self.rows, self.current, self.samples = [], {}, 0

    def synchronize(self):
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def begin(self):
        if self.enabled:
            self.synchronize()
            self.current = {}
            self.start = time.perf_counter()

    @contextmanager
    def stage(self, name):
        with range_scope("agri3d::" + name):
            if not self.enabled:
                yield
                return
            self.synchronize()
            start = time.perf_counter()
            try:
                yield
            finally:
                self.synchronize()
                self.current[name + "_ms"] = (time.perf_counter() - start) * 1000

    def finish(self, batch):
        if self.enabled:
            self.synchronize()
            self.current["total_batch_ms"] = (time.perf_counter() - self.start) * 1000
            for key, value in batch.get("_data_timing_ms", {}).items():
                self.current["cpu_" + key + "_per_sample_ms"] = float(
                    value.double().mean()
                )
            self.samples += len(batch["sample_id"])
            self.rows.append(self.current.copy())

    def summary(self):
        fields = sorted({key for row in self.rows for key in row})
        totals = sum(row["total_batch_ms"] for row in self.rows)
        return {
            "units": "milliseconds",
            "device": str(self.device),
            "cuda_timing": "synchronized wall time; serial debug measurement"
            if self.device.type == "cuda"
            else "not applicable; CPU wall time",
            "batches": len(self.rows),
            "samples": self.samples,
            "samples_per_second": self.samples / (totals / 1000) if totals else None,
            "stages": {
                field: statistics([r[field] for r in self.rows if field in r])
                for field in fields
            },
        }


def telemetry(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "gpu": None,
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "cuda_available": True,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
