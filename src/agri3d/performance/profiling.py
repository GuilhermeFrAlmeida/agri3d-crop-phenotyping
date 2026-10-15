"""Bounded disposable train/val profiling, never a persisted training experiment."""

import copy
from contextlib import nullcontext
from dataclasses import asdict
from functools import wraps
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
    RunConfig,
    environment,
    experiment_contract,
    make_dataset,
    make_loader,
    read_manifest,
    select_device,
    verify_checkpoint_contract,
    write_json,
)

from .ranges import profiling_ranges, range_scope
from .runtime import RuntimeOptions
from .timing import telemetry


def instrument_stages(model):
    for index, module in enumerate(model.abstraction):

        def attach(original, stage, npoint):
            @wraps(original)
            def forward(*args, **kwargs):
                count = args[0].shape[1]
                with range_scope(f"agri3d::sa{stage}_N{count}_S{min(count, npoint)}"):
                    return original(*args, **kwargs)

            return forward

        module.forward = attach(module.forward, index + 1, module.npoint)
    for index, module in enumerate(model.propagation):

        def attach(original, stage):
            @wraps(original)
            def forward(*args, **kwargs):
                with range_scope(f"agri3d::fp{stage}"):
                    return original(*args, **kwargs)

            return forward

        module.forward = attach(module.forward, index + 1)


def device_stat(event, name):
    # PyTorch 2.2 used CUDA-specific names; newer versions use device-neutral ones.
    if hasattr(event, name):
        return getattr(event, name)
    return getattr(event, name.replace("device", "cuda"), 0.0)


def operator_summary(profiler):
    rows = []
    for event in profiler.key_averages(group_by_input_shape=True):
        rows.append(
            {
                "name": event.key,
                "calls": event.count,
                "input_shapes": str(event.input_shapes),
                "self_cpu_us": event.self_cpu_time_total,
                "cpu_total_us": event.cpu_time_total,
                "self_device_us": device_stat(event, "self_device_time_total"),
                "device_total_us": device_stat(event, "device_time_total"),
                "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                "self_device_memory_bytes": device_stat(
                    event, "self_device_memory_usage"
                ),
            }
        )
    families = {
        "fps": ["agri3d::fps"],
        "square_distance": ["agri3d::square_distance"],
        "topk": ["aten::topk"],
        "indexing_gather": ["agri3d::index_points", "aten::gather"],
        "radius_query": ["agri3d::radius_query"],
        "interpolation": ["agri3d::interpolation"],
        "conv1d": ["conv1d"],
        "conv2d": ["conv2d"],
        "batch_norm": ["batch_norm"],
        "copies": ["aten::to", "aten::_to_copy", "aten::copy_"],
        "first_sa": ["agri3d::sa1_"],
    }
    return {
        "operators": sorted(
            rows, key=lambda r: max(r["self_cpu_us"], r["self_device_us"]), reverse=True
        ),
        "families": {
            name: [r for r in rows if any(pattern in r["name"] for pattern in patterns)]
            for name, patterns in families.items()
        },
        "note": "Ranges are inclusive/nested; do not add their totals. Self times rank leaf work. Copy operators are not exclusively H2D; inspect H2D range/trace lanes.",
    }


def profile_model(
    model,
    loader,
    device,
    warmup_batches,
    profile_batches,
    lr=0.001,
    training=False,
    runtime=None,
    optimizer_state=None,
    nvtx=False,
    use_torch_profiler=True,
):
    """Clone model, including BN buffers; original model and its modes are untouched."""
    runtime = runtime or RuntimeOptions()
    if warmup_batches < 0 or profile_batches < 1:
        raise ValueError("Require warmup>=0 and profile_batches>=1")
    if warmup_batches + profile_batches > len(loader):
        raise ValueError(
            "Requested warmup+profile batches exceed split length; reduce counts"
        )
    state = capture_rng()
    try:
        working = copy.deepcopy(model).to(device)
        instrument_stages(working)
        optimizer = torch.optim.Adam(working.parameters(), lr=lr) if training else None
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        iterator = iter(loader)
        if warmup_batches:
            run_epoch(
                working,
                iterator,
                device,
                optimizer,
                max_batches=warmup_batches,
                non_blocking=runtime.non_blocking,
            )
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.device(device).type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler_context = (
            torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            if use_torch_profiler
            else nullcontext()
        )
        with profiling_ranges(nvtx=nvtx), profiler_context as profiler:
            metrics = run_epoch(
                working,
                iterator,
                device,
                optimizer,
                max_batches=profile_batches,
                non_blocking=runtime.non_blocking,
                detailed_timing=runtime.detailed_timing,
                step_callback=profiler.step if profiler is not None else None,
            )
        if metrics["batches"] != profile_batches:
            raise ValueError("Loader ended before requested measured batch count")
        return profiler, {
            "warmup_batches": warmup_batches,
            "profile_batches": metrics["batches"],
            "metrics": metrics,
            "telemetry": telemetry(device),
            "operators": operator_summary(profiler)
            if profiler is not None
            else {
                "operators": [],
                "families": {},
                "note": "NVTX-only workload: inspect external Nsight Systems report; no concurrent CUPTI torch profiler",
            },
        }
    finally:
        restore_rng(state)


def profile_workload(
    manifest,
    root,
    output_dir,
    config=None,
    checkpoint_path=None,
    split="train",
    device_name="auto",
    num_workers=0,
    warmup_batches=1,
    profile_batches=2,
    runtime=None,
    nvtx=False,
    use_torch_profiler=True,
):
    if split not in ("train", "val"):
        raise ValueError("Profiling permits train/val only; test is forbidden")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Profiling requires new/empty output directory")
    runtime = runtime or RuntimeOptions()
    runtime.validate(num_workers)
    state = capture_rng()
    try:
        snapshot_hash = sha256_file(checkpoint_path) if checkpoint_path else None
        checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path else None
        if checkpoint_path and sha256_file(checkpoint_path) != snapshot_hash:
            raise ValueError("Checkpoint changed during load; profile a stable copy")
        config = config or (
            RunConfig(**checkpoint["config"]) if checkpoint else RunConfig()
        )
        config = copy.deepcopy(config)
        config.device, config.num_workers = device_name, num_workers
        config.validate()
        records, manifest_hash = read_manifest(manifest)
        contract = experiment_contract(config, manifest_hash)
        if checkpoint:
            verify_checkpoint_contract(
                checkpoint,
                contract,
                "Profile configuration/manifest differs from checkpoint",
            )
        seed_everything(config.seed)
        device = select_device(device_name)
        dataset = make_dataset(records, root, split, config, runtime)
        # Next training epoch's geometry; validation ignores epoch as before.
        dataset.set_epoch(checkpoint.get("epoch", 0) if checkpoint else 0)
        loader = make_loader(
            dataset,
            config,
            split == "train",
            torch.Generator().manual_seed(config.seed),
            runtime,
        )
        model = PointNet2Segmentation(PointNet2Config(**config.model))
        if checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        profiler, result = profile_model(
            model,
            loader,
            device,
            warmup_batches,
            profile_batches,
            config.lr,
            split == "train",
            runtime,
            checkpoint.get("optimizer_state") if checkpoint else None,
            nvtx,
            use_torch_profiler,
        )
        output.mkdir(parents=True, exist_ok=True)
        if profiler is not None:
            profiler.export_chrome_trace(str(output / "trace.json"))
        result.update(
            split=split,
            contract=contract,
            runtime=asdict(runtime),
            environment=environment(device),
            checkpoint_sha256=snapshot_hash,
            optimizer_source="checkpoint"
            if checkpoint and "optimizer_state" in checkpoint and split == "train"
            else "fresh_or_unused",
            scope="Disposable short workload, no model/checkpoint writes; no test access",
        )
        table = (
            profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
            if profiler is not None
            else "NVTX-only: inspect the external .nsys-rep file."
        )
        (output / "operators.txt").write_text(table)
        write_json(output / "summary.json", result)
        return result
    finally:
        restore_rng(state)
