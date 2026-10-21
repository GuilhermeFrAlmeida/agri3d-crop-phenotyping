"""Reproducible Phase 3 runs. Only train/val splits are loaded; test is excluded."""

import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from agri3d.models import PointNet2Config, PointNet2Segmentation
from agri3d.performance.runtime import RuntimeOptions, build_loader
from agri3d.performance.timing import telemetry
from agri3d.preprocessing.augmentation import AugmentationConfig
from agri3d.preprocessing.features import FeatureConfig
from agri3d.preprocessing.normals import NormalConfig, normal_provenance

from .checkpoints import (
    capture_rng,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
    seed_everything,
)
from .dataset import SegmentationDataset
from .engine import run_epoch
from .scheduling import SchedulerConfig, scheduler_summary
from .splits import validate_manifest


@dataclass
class RunConfig:
    points: int = 10000
    epochs: int = 100
    batch_size: int = 2
    lr: float = 0.001
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    overfit_samples: int = 0
    model: dict = field(default_factory=dict)
    augmentation: dict = field(default_factory=dict)
    scheduler: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)

    def validate(self):
        if any(
            not isinstance(x, int) or x < 1
            for x in (self.points, self.epochs, self.batch_size)
        ):
            raise ValueError("points, epochs and batch_size must be positive integers")
        if (
            self.points < 2
            or not math.isfinite(self.lr)
            or self.lr <= 0
            or self.num_workers < 0
        ):
            raise ValueError("Require points>=2, lr>0, num_workers>=0")
        if not isinstance(self.num_workers, int) or not isinstance(
            self.overfit_samples, int
        ):
            raise TypeError("num_workers and overfit_samples must be integers")
        if self.overfit_samples != 0 and not 2 <= self.overfit_samples <= 8:
            raise ValueError("Tiny-overfit requires 2–8 training plants")
        if not 0 <= self.seed < 2**32:
            raise ValueError("Invalid seed")
        PointNet2Config(**self.model)
        scheduler = SchedulerConfig.from_settings(self.scheduler)
        if scheduler.enabled and scheduler.min_lr > self.lr:
            raise ValueError("scheduler.min_lr must not exceed initial LR")
        if self.overfit_samples and scheduler.enabled:
            raise ValueError("Tiny-overfit requires disabled scheduler")
        features = FeatureConfig.from_settings(self.features)
        augmentation = AugmentationConfig.from_settings(self.augmentation)
        if features.channels and (augmentation.enabled or scheduler.enabled):
            raise ValueError(
                "Run 4 normal features require disabled augmentation and scheduler"
            )
        if self.overfit_samples and augmentation.enabled:
            raise ValueError(
                "Tiny-overfit requires disabled augmentation; use the Run 1 configuration"
            )


def build_model(config):
    """Only feature-dependent input dimensions change; geometry always stays XYZ."""
    return PointNet2Segmentation(
        PointNet2Config(**config.model),
        feature_channels=FeatureConfig.from_settings(config.features).channels,
    )


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_manifest(path: str | Path) -> tuple[list[dict], str]:
    raw = Path(path).read_bytes()
    records = validate_manifest(json.loads(raw))
    return records, hashlib.sha256(raw).hexdigest()


def experiment_contract(config: RunConfig, manifest_hash: str) -> dict:
    """Data/model invariants required for gate validation and comparable evaluation."""
    # Normalize tuples/lists so YAML and checkpoint configurations compare identically.
    model = json.loads(json.dumps(PointNet2Config(**config.model).to_dict()))
    contract = {
        "manifest_sha256": manifest_hash,
        "points": config.points,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "optimizer": {"name": "Adam", "lr": config.lr},
        "model": model,
        "annotation_mode": "maizefield3d",
        "features": FeatureConfig.from_settings(config.features).contract(),
        "classes": {"0": "leaf", "1": "stalk"},
        "sampling": "voxel",
        "voxel_size_normalized": 0.005,
        "sor": False,
        "augmentation": AugmentationConfig.from_settings(
            config.augmentation
        ).contract(),
    }
    scheduler = SchedulerConfig.from_settings(config.scheduler)
    # Keep disabled contracts byte-compatible with pre-scheduler runs.
    if scheduler.enabled:
        contract["scheduler"] = scheduler.contract()
    return contract


def make_dataset(
    records: list[dict], root: str | Path, split: str, config: RunConfig, runtime=None
):
    if split not in ("train", "val"):
        raise ValueError("Phase 3 development permits train/val only")
    runtime = runtime or RuntimeOptions()
    return SegmentationDataset(
        records,
        root,
        split,
        config.points,
        seed=config.seed,
        annotation_mode="maizefield3d",
        sor=False,
        augmentation=config.augmentation,
        features=config.features,
        cache_dir=runtime.cache_dir,
        profile_data=runtime.detailed_timing,
    )


def make_loader(
    dataset, config: RunConfig, shuffle: bool, generator: torch.Generator, runtime=None
):
    return build_loader(dataset, config, shuffle, generator, runtime)


def choose_overfit_indices(dataset, count: int, seed: int) -> list[int]:
    if len(dataset) < count:
        raise ValueError("Not enough training plants for requested overfit subset")
    ordered = sorted(range(len(dataset)), key=lambda i: dataset.records[i]["sample_id"])
    selected = np.random.default_rng(seed).choice(len(ordered), count, replace=False)
    return sorted(
        (ordered[int(i)] for i in selected),
        key=lambda i: dataset.records[i]["sample_id"],
    )


def overfit_status(initial_loss: float, metrics: dict) -> dict:
    """Explicit engineering gate, not a generalization claim."""
    ratio = metrics["loss"] / max(initial_loss, 1e-12)
    passed = ratio <= 0.2 and metrics["accuracy"] >= 0.98 and metrics["mIoU"] >= 0.95
    return {
        "passed": passed,
        "initial_loss": initial_loss,
        "loss_ratio": ratio,
        "accuracy": metrics["accuracy"],
        "mIoU": metrics["mIoU"],
        "criteria": "loss <=20% of initial; accuracy >=0.98; mIoU >=0.95",
    }


def verify_overfit_gate(checkpoint: dict, contract: dict) -> None:
    if checkpoint.get("mode") != "tiny_overfit" or not checkpoint.get(
        "overfit_status", {}
    ).get("passed"):
        raise ValueError("Regular training requires a passing tiny-overfit checkpoint")
    # The fixed memorization gate has neither augmentation nor scheduling.
    # All remaining data/model/optimizer invariants must match.
    expected = {**contract, "augmentation": None}
    expected.pop("scheduler", None)
    verify_checkpoint_contract(
        checkpoint,
        expected,
        "Overfit checkpoint data/model contract differs from this run",
    )


def contract_sha256(contract: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def verify_checkpoint_contract(checkpoint: dict, expected: dict, message: str) -> None:
    if checkpoint.get("contract") != expected:
        raise ValueError(message)
    saved_hash = checkpoint.get("contract_sha256")
    # Historical XYZ-only checkpoints predate the hash; newer feature policies require it.
    if (
        saved_hash is None
        and expected.get("augmentation") is None
        and "scheduler" not in expected
        and expected.get("features", "xyz") == "xyz"
    ):
        return
    if saved_hash != contract_sha256(expected):
        raise ValueError("Checkpoint contract SHA256 is missing or inconsistent")


def environment(device: torch.device) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown (not a git checkout)"
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "device": str(device),
        "code_commit": commit,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def train(
    manifest: str,
    root: str,
    output_dir: str,
    config: RunConfig,
    resume: str | None = None,
    overfit_checkpoint: str | None = None,
    runtime: RuntimeOptions | None = None,
) -> dict:
    config.validate()
    runtime = runtime or RuntimeOptions()
    runtime.validate(config.num_workers)
    seed_everything(config.seed)
    device = select_device(config.device)
    records, manifest_hash = read_manifest(manifest)
    contract = experiment_contract(config, manifest_hash)
    mode = "tiny_overfit" if config.overfit_samples else "regular"
    resumed = load_checkpoint(resume) if resume else None
    gate_evidence = None
    if resumed:
        verify_checkpoint_contract(
            resumed, contract, "Resume checkpoint data/model/mode contract differs"
        )
        if resumed.get("mode") != mode:
            raise ValueError("Resume checkpoint data/model/mode contract differs")
        old_config = resumed["config"]
        for key in ("batch_size", "lr", "num_workers", "overfit_samples"):
            if old_config[key] != getattr(config, key):
                raise ValueError(f"Resume requires unchanged {key}")
        if resumed["epoch"] >= config.epochs:
            raise ValueError("--epochs must exceed completed checkpoint epochs")
        gate_evidence = resumed.get("gate_evidence")
        if mode == "regular" and not gate_evidence:
            raise ValueError("Regular resume checkpoint has no overfit gate evidence")
    elif mode == "regular":
        if not overfit_checkpoint:
            raise ValueError(
                "Run tiny-overfit first, then supply --overfit-checkpoint overfit_passed.pt"
            )
        gate = load_checkpoint(overfit_checkpoint)
        verify_overfit_gate(gate, contract)
        gate_evidence = {
            "checkpoint_sha256": hashlib.sha256(
                Path(overfit_checkpoint).read_bytes()
            ).hexdigest(),
            "overfit_status": gate["overfit_status"],
            "sample_ids": gate["train_sample_ids"],
            "contract": gate["contract"],
            "contract_sha256": contract_sha256(gate["contract"]),
        }

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError(
            "Output directory is nonempty; use a new directory or --resume"
        )
    if resume and Path(resume).resolve() != (output / "last.pt").resolve():
        raise ValueError("Resume from output-dir/last.pt to preserve one run history")
    output.mkdir(parents=True, exist_ok=True)

    training = make_dataset(records, root, "train", config, runtime)
    train_view = training
    if config.overfit_samples:
        train_view = Subset(
            training,
            choose_overfit_indices(training, config.overfit_samples, config.seed),
        )
    train_ids = (
        [training.records[i]["sample_id"] for i in train_view.indices]
        if isinstance(train_view, Subset)
        else [r["sample_id"] for r in training.records]
    )
    validation = (
        make_dataset(records, root, "val", config, runtime)
        if mode == "regular"
        else None
    )
    shuffle_rng = torch.Generator().manual_seed(config.seed)
    eval_rng = torch.Generator().manual_seed(config.seed + 1)
    train_loader = make_loader(train_view, config, True, shuffle_rng, runtime)
    eval_loader = make_loader(
        train_view if validation is None else validation,
        config,
        False,
        eval_rng,
        runtime,
    )
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = SchedulerConfig.from_settings(config.scheduler).build(optimizer)
    history, start_epoch, best_miou, best_loss = [], 0, -1.0, float("inf")
    initial_metrics = None
    if resumed:
        model.load_state_dict(resumed["model_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        if scheduler is not None:
            if not isinstance(resumed.get("scheduler_state"), dict):
                raise ValueError("Scheduled resume requires scheduler_state")
            scheduler.load_state_dict(resumed["scheduler_state"])
        history, start_epoch = resumed["history"], resumed["epoch"]
        best_miou, best_loss = resumed["best_miou"], resumed["best_loss"]
        initial_metrics = resumed["initial_metrics"]
        shuffle_rng.set_state(resumed["loader_rng"]["shuffle"])
        eval_rng.set_state(resumed["loader_rng"]["eval"])
        restore_rng(resumed["rng"])
    elif mode == "tiny_overfit":
        training.set_epoch(0)
        initial_metrics = run_epoch(
            model,
            eval_loader,
            device,
            non_blocking=runtime.non_blocking,
            detailed_timing=runtime.detailed_timing,
        )
    run_info = {
        "config": asdict(config),
        "runtime": asdict(runtime),
        "contract": contract,
        "contract_sha256": contract_sha256(contract),
        "mode": mode,
        "train_sample_ids": train_ids,
        "initial_metrics": initial_metrics,
        "split_counts": {
            s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")
        },
        "parameters": sum(p.numel() for p in model.parameters()),
        "point_feature_channels": model.feature_channels,
        "feature_provenance": normal_provenance(
            NormalConfig.from_settings(config.features.get("normals"))
        )
        if model.feature_channels
        else None,
        "parameters_added_by_point_features": model.feature_channels
        * (
            model.abstraction[0].mlp[0].out_channels
            + model.propagation[-1].mlp[0].out_channels
        ),
        "environment": environment(device),
        "gate_evidence": gate_evidence,
        "scope": "fixed sampled points; no full-cloud projection; no test evaluation",
    }
    write_json(output / "run.json", run_info)
    for epoch in range(start_epoch, config.epochs):
        epoch_lr = optimizer.param_groups[0]["lr"]
        training.set_epoch(0 if mode == "tiny_overfit" else epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            non_blocking=runtime.non_blocking,
            detailed_timing=runtime.detailed_timing,
        )
        # Tiny-overfit is scored after updates in eval mode, on the same fixed train points.
        selected_metrics = run_epoch(
            model,
            eval_loader,
            device,
            non_blocking=runtime.non_blocking,
            detailed_timing=runtime.detailed_timing,
        )
        status = (
            overfit_status(initial_metrics["loss"], selected_metrics)
            if initial_metrics
            else None
        )
        selection = selected_metrics["mIoU"]
        # Exactly once, after validation; next epoch uses any reduced LR.
        if scheduler is not None:
            scheduler.step(selection)
        next_lr = optimizer.param_groups[0]["lr"]
        improved = selection > best_miou or (
            selection == best_miou and selected_metrics["loss"] < best_loss
        )
        if improved:
            best_miou, best_loss = selection, selected_metrics["loss"]
        record = {
            "epoch": epoch + 1,
            "lr": epoch_lr,
            "next_lr": next_lr,
            "lr_changed": next_lr != epoch_lr,
            "scheduler": scheduler_summary(scheduler),
            "train": train_metrics,
            "train_eval" if mode == "tiny_overfit" else "val": selected_metrics,
            "overfit_status": status,
        }
        if runtime.telemetry:
            record["telemetry"] = telemetry(device)
        history.append(record)
        payload = {
            **run_info,
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()
            if scheduler is not None
            else None,
            "best_miou": best_miou,
            "best_loss": best_loss,
            "selection_metric": "train_mIoU" if mode == "tiny_overfit" else "val_mIoU",
            "overfit_status": status,
            "history": history,
            "rng": capture_rng(),
            "loader_rng": {
                "shuffle": shuffle_rng.get_state(),
                "eval": eval_rng.get_state(),
            },
        }
        save_checkpoint(output / "last.pt", payload)
        if improved:
            save_checkpoint(output / "best.pt", payload)
        if status and status["passed"]:
            save_checkpoint(output / "overfit_passed.pt", payload)
        write_json(output / "metrics.json", history)
        print(json.dumps(record, allow_nan=False), flush=True)
    metric_key = "train_eval" if mode == "tiny_overfit" else "val"
    selected_record = max(
        history, key=lambda row: (row[metric_key]["mIoU"], -row[metric_key]["loss"])
    )
    summary = {
        "best_epoch": selected_record["epoch"],
        "lr_at_best_epoch": selected_record.get("lr"),
        "final_lr": optimizer.param_groups[0]["lr"],
        "lr_reduction_epochs": [
            row["epoch"] for row in history if row.get("lr_changed")
        ],
        "scheduler": scheduler_summary(scheduler),
        "mode": mode,
        "epochs_completed": config.epochs,
        "best_mIoU": best_miou,
        "last_overfit_status": history[-1]["overfit_status"],
        "overfit_gate_checkpoint": "overfit_passed.pt"
        if mode == "tiny_overfit" and (output / "overfit_passed.pt").exists()
        else None,
        "next_step": "Inspect tiny-overfit results before regular training"
        if mode == "tiny_overfit"
        else "Inspect frozen-validation results; leave test untouched",
    }
    write_json(output / "summary.json", summary)
    return summary


def validate_checkpoint(
    manifest: str,
    root: str,
    checkpoint_path: str,
    output_path: str,
    device_name: str = "auto",
    num_workers: int = 0,
) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("mode") != "regular":
        raise ValueError(
            "Validation command expects a regular-run checkpoint; inspect tiny-overfit train_eval instead"
        )
    config = RunConfig(**checkpoint["config"])
    config.device, config.num_workers = device_name, num_workers
    config.validate()
    seed_everything(config.seed)
    records, manifest_hash = read_manifest(manifest)
    verify_checkpoint_contract(
        checkpoint,
        experiment_contract(config, manifest_hash),
        "Validation manifest differs from checkpoint contract",
    )
    device = select_device(device_name)
    dataset = make_dataset(records, root, "val", config)
    loader = make_loader(
        dataset, config, False, torch.Generator().manual_seed(config.seed + 1)
    )
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    result = {
        "split": "val",
        "scope": "fixed sampled points; not full-cloud projection",
        "checkpoint_sha256": hashlib.sha256(
            Path(checkpoint_path).read_bytes()
        ).hexdigest(),
        "contract": checkpoint["contract"],
        "metrics": run_epoch(model, loader, device),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result
