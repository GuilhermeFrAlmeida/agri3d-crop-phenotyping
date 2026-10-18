"""Short loader/transfer/cache comparisons; no model, optimizer or checkpoint writes."""

import hashlib
import json
import time
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import torch

from agri3d.training.checkpoints import capture_rng, restore_rng, seed_everything
from agri3d.training.pointnet2_runner import (
    environment,
    experiment_contract,
    make_dataset,
    make_loader,
    read_manifest,
    select_device,
    write_json,
)

from .runtime import RuntimeOptions
from .timing import BatchTimings, telemetry


def batch_fingerprint(batch):
    """Exact sampled/augmented tensors and plant identities, excluding timings."""
    digest = hashlib.sha256(json.dumps(list(batch["sample_id"])).encode())
    for key in ("points", "labels", "valid_mask", "source_index", "centroid", "scale"):
        tensor = batch[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def benchmark_loaders(
    manifest,
    root,
    output_dir,
    config,
    split="train",
    workers=(0, 2, 4),
    pins=(False, True),
    warmup_batches=1,
    batches=5,
    device_name="cpu",
    cache_dir=None,
    compare_cache=False,
    prefetch_factor=None,
    persistent_workers=False,
    non_blocking=False,
    matrix=False,
):
    if split not in ("train", "val"):
        raise ValueError(
            "Loader benchmarking permits train/val only; test is forbidden"
        )
    if warmup_batches < 0 or batches < 1 or not workers or not pins:
        raise ValueError(
            "Require positive batches, warmup>=0 and nonempty configurations"
        )
    config.validate()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Benchmark requires new/empty output directory")
    if compare_cache and cache_dir is None:
        raise ValueError("compare_cache requires cache_dir")
    if compare_cache and Path(cache_dir).exists() and any(Path(cache_dir).iterdir()):
        raise ValueError("Cache comparison requires an empty cache directory")
    state = capture_rng()
    try:
        seed_everything(config.seed)
        device = select_device(device_name)
        records, digest = read_manifest(manifest)
        # Reference tensors are checked after timed windows, so hashing adds no timed work.
        reference = None
        results = []
        for count, pin in product(workers, pins):
            blocks = (False, True) if matrix else (non_blocking,)
            persistence = (
                (False, True)
                if matrix and count > 0
                else (persistent_workers if count else False,)
            )
            for blocking, persistent in product(blocks, persistence):
                identifier = (
                    f"w{count}_pin{int(pin)}_nb{int(blocking)}_p{int(persistent)}"
                )
                directory = (
                    str(Path(cache_dir) / identifier) if compare_cache else cache_dir
                )
                modes = (
                    [
                        ("uncached", None),
                        ("cache_population", directory),
                        ("cache_warm", directory),
                    ]
                    if compare_cache
                    else [("cache" if cache_dir else "uncached", directory)]
                )
                for mode, active_cache in modes:
                    cfg = replace(config, num_workers=count)
                    runtime = RuntimeOptions(
                        pin_memory=pin,
                        prefetch_factor=prefetch_factor if count else None,
                        persistent_workers=persistent,
                        non_blocking=blocking,
                        cache_dir=active_cache,
                        detailed_timing=True,
                    )
                    row = {
                        "num_workers": count,
                        "mode": mode,
                        "runtime": asdict(runtime),
                    }
                    loader = iterator = None
                    try:
                        cfg.validate()
                        runtime.validate(count)
                        dataset = make_dataset(records, root, split, cfg, runtime)
                        loader = make_loader(
                            dataset,
                            cfg,
                            False,
                            torch.Generator().manual_seed(cfg.seed),
                            runtime,
                        )
                        if count:
                            loader.timeout = 30
                        if warmup_batches + batches > len(loader):
                            raise ValueError("Requested batches exceed split length")
                        # Two bounded passes expose worker reuse. Same prefix, epoch, and points.
                        passes = []
                        fingerprints = None
                        for pass_index in range(2):
                            started = time.perf_counter()
                            iterator = iter(loader)
                            startup_ms = (time.perf_counter() - started) * 1000
                            for _ in range(warmup_batches):
                                batch = next(iterator)
                                for key in ("points", "labels", "valid_mask"):
                                    batch[key].to(device, non_blocking=blocking)
                            timer = BatchTimings(device, True)
                            retained = []
                            cache_hits, cache_misses = 0, 0
                            for _ in range(batches):
                                timer.begin()
                                with timer.stage("data_wait"):
                                    batch = next(iterator)
                                with timer.stage("h2d"):
                                    transferred = [
                                        batch[key].to(device, non_blocking=blocking)
                                        for key in ("points", "labels", "valid_mask")
                                    ]
                                del transferred
                                timer.finish(batch)
                                retained.append(batch)
                            timing = timer.summary()
                            # Outside timing: identity validation includes sampling and augmentation.
                            actual = [batch_fingerprint(b) for b in retained]
                            if reference is None:
                                reference = actual
                            if actual != reference:
                                raise ValueError(
                                    "Deterministic sample/tensor identity mismatch"
                                )
                            if fingerprints is not None and actual != fingerprints:
                                raise ValueError("Repeated pass identity mismatch")
                            fingerprints = actual
                            for b in retained:
                                cache_hits += int((b["_cache_hit"] == 1).sum())
                                cache_misses += int((b["_cache_hit"] == 0).sum())
                            passes.append(
                                {
                                    "pass": pass_index + 1,
                                    "iterator_startup_ms": startup_ms,
                                    "timing": timing,
                                    "actual_pinned": all(
                                        b["points"].is_pinned() for b in retained
                                    ),
                                    "cache_hits": cache_hits,
                                    "cache_misses": cache_misses,
                                }
                            )
                            del retained
                        # Population reports first traversal; warm reports reused traversal.
                        selected = (
                            passes[0]
                            if mode == "cache_population"
                            or (compare_cache and mode == "uncached")
                            else passes[-1]
                        )
                        stages = selected["timing"]["stages"]
                        row.update(
                            status="ok",
                            passes=passes,
                            timing=selected["timing"],
                            actual_pinned=selected["actual_pinned"],
                            sample_fingerprints=fingerprints,
                            identity_verified=True,
                            cache_hits=selected["cache_hits"],
                            cache_misses=selected["cache_misses"],
                            batch_preparation_median_ms=stages["data_wait_ms"][
                                "median"
                            ],
                            batch_preparation_p95_ms=stages["data_wait_ms"]["p95"],
                            h2d_median_ms=stages["h2d_ms"]["median"],
                            total_batch_median_ms=stages["total_batch_ms"]["median"],
                            plants_per_second=selected["timing"]["samples_per_second"],
                            non_blocking_cuda_effective=blocking
                            and device.type == "cuda"
                            and selected["actual_pinned"],
                        )
                    except (ValueError, TypeError, RuntimeError, OSError) as exc:
                        row.update(status="unavailable_or_failed", error=str(exc))
                    finally:
                        del iterator, loader
                    results.append(row)
        result = {
            "split": split,
            "manifest_sha256": digest,
            "config": asdict(config),
            "contract": experiment_contract(config, digest),
            "warmup_batches": warmup_batches,
            "measured_batches_per_pass": batches,
            "environment": environment(device),
            "telemetry": telemetry(device),
            "results": results,
            "note": "Two bounded passes per row; no model training. Synchronized loader+H2D throughput is not model throughput. OS file cache is uncontrolled; uncached means application cache disabled. Warmup excluded. Hashing occurs after each measured pass. Worker prefetch may read extra train/val plants. Cache population uses an isolated fresh subdirectory per configuration; report hit/miss counts. No automatic runtime recommendations.",
        }
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "benchmark.json", result)
        return result
    finally:
        restore_rng(state)
