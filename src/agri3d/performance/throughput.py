"""Disposable batch-size throughput comparisons, separate from scientific runs."""

import copy
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from agri3d.models import PointNet2Config, PointNet2Segmentation
from agri3d.preprocessing.cache import sha256_file
from agri3d.training.checkpoints import (
    capture_rng,
    load_checkpoint,
    restore_rng,
    seed_everything,
)
from agri3d.training.engine import run_epoch
from agri3d.training.pointnet2_runner import (
    environment,
    experiment_contract,
    make_dataset,
    make_loader,
    read_manifest,
    select_device,
    verify_checkpoint_contract,
    write_json,
)

from .runtime import RuntimeOptions
from .timing import telemetry


def benchmark_batch_sizes(
    manifest,
    root,
    output_dir,
    config,
    batch_sizes=(2, 4, 8),
    warmup_batches=1,
    batches=3,
    device_name="auto",
    runtime=None,
    checkpoint_path=None,
    split="train",
):
    if split != "train":
        raise ValueError(
            "Throughput training benchmark permits train only; test is forbidden"
        )
    if (
        type(warmup_batches) is not int
        or warmup_batches < 0
        or type(batches) is not int
        or batches < 1
    ):
        raise ValueError("Require warmup>=0 and batches>=1")
    if not batch_sizes or any(type(b) is not int or b < 1 for b in batch_sizes):
        raise ValueError("batch_sizes must be positive integers")
    config.validate()
    runtime = runtime or RuntimeOptions()
    runtime.validate(config.num_workers)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Throughput benchmark requires new/empty output directory")
    state = capture_rng()
    try:
        records, digest = read_manifest(manifest)
        contract = experiment_contract(config, digest)
        checksum = sha256_file(checkpoint_path) if checkpoint_path else None
        checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path else None
        if checkpoint:
            verify_checkpoint_contract(
                checkpoint, contract, "Throughput configuration differs from checkpoint"
            )
            if sha256_file(checkpoint_path) != checksum:
                raise ValueError("Checkpoint changed while loading")
        device = select_device(device_name)
        results = []

        def measure(cfg, detailed):
            seed_everything(cfg.seed)
            dataset = make_dataset(
                records, root, "train", cfg, replace(runtime, detailed_timing=detailed)
            )
            dataset.set_epoch(checkpoint.get("epoch", 0) if checkpoint else 0)
            if len(dataset) < (warmup_batches + batches) * cfg.batch_size:
                raise ValueError(
                    "Insufficient plants for requested full benchmark batches"
                )
            loader = make_loader(
                dataset, cfg, False, torch.Generator().manual_seed(cfg.seed), runtime
            )
            if cfg.num_workers:
                loader.timeout = 30
            model = PointNet2Segmentation(PointNet2Config(**cfg.model)).to(device)
            if checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            if checkpoint:
                optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state"]))
            iterator = iter(loader)
            if warmup_batches:
                run_epoch(
                    model,
                    iterator,
                    device,
                    optimizer,
                    max_batches=warmup_batches,
                    non_blocking=runtime.non_blocking,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            metrics = run_epoch(
                model,
                iterator,
                device,
                optimizer,
                max_batches=batches,
                non_blocking=runtime.non_blocking,
                detailed_timing=detailed,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            if metrics["batches"] != batches:
                raise ValueError("Loader ended before requested measured batches")
            return {
                "seconds": elapsed,
                "samples_per_second": batches * cfg.batch_size / elapsed,
                "batches_per_second": batches / elapsed,
                "metrics": metrics,
                "telemetry": telemetry(device),
            }

        for size in batch_sizes:
            cfg = replace(config, batch_size=size, device=device_name)
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                throughput = measure(cfg, False)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                diagnostic = measure(cfg, True)
                stages = diagnostic["metrics"]["timing"]["stages"]
                results.append(
                    {
                        "batch_size": size,
                        "status": "ok",
                        "throughput": throughput,
                        "diagnostic": diagnostic,
                        "forward_median_ms": stages["forward_ms"]["median"],
                        "backward_median_ms": stages["backward_ms"]["median"],
                        "epoch_estimate_seconds": len(
                            [r for r in records if r["split"] == "train"]
                        )
                        / throughput["samples_per_second"],
                    }
                )
            except (RuntimeError, ValueError, OSError) as exc:
                results.append(
                    {
                        "batch_size": size,
                        "status": "unavailable_or_failed",
                        "error": str(exc),
                    }
                )
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        result = {
            "scientific_contract": contract,
            "scientific_config": asdict(config),
            "runtime": asdict(runtime),
            "checkpoint_sha256": checksum,
            "environment": environment(device),
            "warmup_batches": warmup_batches,
            "measured_batches": batches,
            "results": results,
            "scope": "Disposable train-only workload; two independent passes per batch size. Primary throughput has no profiler/stage synchronization. Diagnostic pass synchronizes stages. No checkpoint/model writes, scheduler steps, or scientific results. Different batch size changes training/BatchNorm semantics and is never a drop-in runtime recommendation. Epoch estimates exclude validation/checkpoint overhead and are rough short-window extrapolations.",
        }
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "throughput.json", result)
        return result
    finally:
        restore_rng(state)
