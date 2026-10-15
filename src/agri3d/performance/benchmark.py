"""Short loader/transfer/cache comparisons; no model, optimizer or checkpoint writes."""

from dataclasses import asdict, replace
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
):
    if split not in ("train", "val"):
        raise ValueError(
            "Loader benchmarking permits train/val only; test is forbidden"
        )
    if warmup_batches < 0 or batches < 1 or not workers or not pins:
        raise ValueError(
            "Require positive batches, warmup>=0 and nonempty configurations"
        )
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Benchmark requires new/empty output directory")
    if compare_cache and cache_dir is None:
        raise ValueError("compare_cache requires cache_dir")
    state = capture_rng()
    try:
        seed_everything(config.seed)
        device = select_device(device_name)
        records, digest = read_manifest(manifest)
        results = []
        modes = (
            [("uncached", None)]
            if compare_cache
            else [("cache" if cache_dir else "uncached", cache_dir)]
        )
        if compare_cache:
            modes += [("cache_first_pass", cache_dir), ("cache_reuse", cache_dir)]
        for count in workers:
            for pin in pins:
                for mode, directory in modes:
                    cfg = replace(config, num_workers=count)
                    runtime = RuntimeOptions(
                        pin_memory=pin,
                        prefetch_factor=prefetch_factor if count else None,
                        persistent_workers=persistent_workers if count else False,
                        non_blocking=non_blocking,
                        cache_dir=directory,
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
                        iterator = iter(loader)
                        for _ in range(warmup_batches):
                            batch = next(iterator)
                            for key in ("points", "labels", "valid_mask"):
                                batch[key].to(device, non_blocking=non_blocking)
                        timer = BatchTimings(device, True)
                        observed_pinned = []
                        for _ in range(batches):
                            timer.begin()
                            with timer.stage("data_wait"):
                                batch = next(iterator)
                            observed_pinned.append(batch["points"].is_pinned())
                            with timer.stage("h2d"):
                                transferred = [
                                    batch[key].to(device, non_blocking=non_blocking)
                                    for key in ("points", "labels", "valid_mask")
                                ]
                            del transferred
                            timer.finish(batch)
                        row.update(
                            status="ok",
                            timing=timer.summary(),
                            actual_pinned=all(observed_pinned),
                        )
                    except (ValueError, RuntimeError, OSError) as exc:
                        row.update(status="unavailable_or_failed", error=str(exc))
                    finally:
                        # Iterator lifetime controls worker teardown; no persistent worker leak
                        # across the configuration matrix, including a bounded early exit.
                        del iterator, loader
                    results.append(row)
        result = {
            "split": split,
            "manifest_sha256": digest,
            "config": asdict(config),
            "contract": experiment_contract(config, digest),
            "warmup_batches": warmup_batches,
            "measured_batches": batches,
            "environment": environment(device),
            "telemetry": telemetry(device),
            "results": results,
            "note": "No automatic recommendations/config changes. Data wait includes residual prefetch; CPU per-sample times may overlap. First cache pass may already hit an existing cache; use an empty cache directory for cold experiments. Worker startup/warmup excluded.",
        }
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "benchmark.json", result)
        return result
    finally:
        restore_rng(state)
