"""CPU correctness/recovery tests; performance claims require separate measurement."""

import argparse
import copy
import importlib.util
import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from test_pointnet2_analysis import fixture_checkpoint
from test_pointnet2_training import config, fixture_data

from agri3d.models import PointNet2Config, PointNet2Segmentation
from agri3d.performance.benchmark import benchmark_loaders
from agri3d.performance.inspection import inspect_checkpoint
from agri3d.performance.profiling import profile_model, profile_workload
from agri3d.performance.ranges import profiling_ranges
from agri3d.performance.runtime import (
    EpochDataLoader,
    RuntimeOptions,
    add_runtime_arguments,
    runtime_from_args,
)
from agri3d.performance.timing import BatchTimings, telemetry
from agri3d.preprocessing.cache import NormalizedCloudCache
from agri3d.training.checkpoints import (
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)
from agri3d.training.dataset import SegmentationDataset
from agri3d.training.engine import run_epoch
from agri3d.training.pointnet2_runner import (
    experiment_contract,
    make_dataset,
    make_loader,
    read_manifest,
    train,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def equivalent(a, b):
    assert set(a) == set(b)
    for key in a:
        if isinstance(a[key], torch.Tensor):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        else:
            assert a[key] == b[key]


def test_inspector_no_dataset_or_model(tmp_path, monkeypatch):
    checkpoint = tmp_path / "last.pt"
    history = [
        {"epoch": epoch, "val": {"mIoU": miou, "loss": loss}}
        for epoch, miou, loss in [(68, 0.82, 0.2), (69, 0.84, 0.19), (70, 0.83, 0.18)]
    ]
    cfg = config(100)
    cfg.overfit_samples = 0
    contract = experiment_contract(cfg, "manifest-hash")
    save_checkpoint(
        checkpoint,
        {
            "mode": "regular",
            "epoch": 70,
            "config": asdict(cfg),
            "history": history,
            "best_miou": 0.84,
            "best_loss": 0.19,
            "contract": contract,
            "environment": {"device": "cuda", "gpu": "recorded GPU"},
        },
    )
    before = checkpoint.read_bytes()
    monkeypatch.setattr(
        SegmentationDataset, "__init__", lambda *_a, **_k: pytest.fail("Dataset loaded")
    )
    result = inspect_checkpoint(checkpoint, 2)
    assert result["completed_epoch"] == 70 and result["next_completed_epoch"] == 71
    assert (
        result["selected_epoch"] == 69 and result["epochs_since_last_improvement"] == 1
    )
    assert result["best_validation_loss"] == 0.19  # Not minimum historical .18.
    assert len(result["recent_validation_trajectory"]) == 2
    assert result["contract_hash_valid"] is None
    assert checkpoint.read_bytes() == before
    command = [
        sys.executable,
        "-c",
        "import sys; from agri3d.performance.inspection import inspect_checkpoint; inspect_checkpoint(sys.argv[1]); assert 'agri3d.training.dataset' not in sys.modules",
        str(checkpoint),
    ]
    subprocess.run(command, check=True, capture_output=True)


def test_inspector_ties_missing_history(tmp_path):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        {
            "mode": "regular",
            "epoch": 4,
            "best_miou": 0.8,
            "best_loss": 0.1,
            "history": [
                {"epoch": 1, "val": {"mIoU": 0.8, "loss": 0.2}},
                {"epoch": 2, "val": {"mIoU": 0.8, "loss": 0.1}},
            ],
        },
    )
    assert inspect_checkpoint(path)["best_epoch"] == 2
    save_checkpoint(path, {"mode": "regular", "epoch": 70, "best_miou": 0.8})
    assert inspect_checkpoint(path)["best_epoch"] is None


def test_cache_equivalence_epochs_and_provenance(tmp_path):
    manifest, root = fixture_data(tmp_path)
    records, _ = read_manifest(manifest)
    cfg = config()
    cfg.augmentation = {"enabled": True}
    cfg.overfit_samples = 0
    for split in ("train", "val"):
        base = make_dataset(records, root, split, cfg)
        cached = make_dataset(
            records, root, split, cfg, RuntimeOptions(cache_dir=str(tmp_path / "cache"))
        )
        for epoch in (0, 2, 7):
            base.set_epoch(epoch)
            cached.set_epoch(epoch)
            equivalent(base[0], cached[0])
            equivalent(base[0], cached[0])
        entries = list((tmp_path / "cache").glob("*.npz"))
        assert entries
        with np.load(entries[0], allow_pickle=False) as data:
            identity = json.loads(str(data["identity"]))
            assert identity["source_sha256"] and identity["pipeline_sha256"]
            assert not any(
                key.endswith("valid_mask") for key in data.files
            )  # Before sampling.
    source = tmp_path / "0.ply"
    cache = NormalizedCloudCache(tmp_path / "cache2")
    old = cache.identity(source)
    # Content edit with identical size and preserved mtime still invalidates.
    status = source.stat()
    raw = source.read_bytes()
    changed = raw.replace(b"255", b"254", 1)
    assert changed != raw and len(changed) == len(raw)
    source.write_bytes(changed)
    os.utime(source, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert cache.identity(source)["source_sha256"] != old["source_sha256"]
    cache.pipeline_hash = "changed-pipeline"
    assert cache.identity(source)["pipeline_sha256"] != old["pipeline_sha256"]
    # No test file exists or is required.
    assert not (tmp_path / "4.ply").exists()


def test_cache_corruption_and_immutability(tmp_path):
    manifest, root = fixture_data(tmp_path)
    records, _ = read_manifest(manifest)
    ds = make_dataset(
        records,
        root,
        "train",
        config(),
        RuntimeOptions(cache_dir=str(tmp_path / "cache")),
    )
    first = ds[0]
    first["points"].fill_(999)
    assert not torch.all(ds[0]["points"] == 999)
    entry = next((tmp_path / "cache").glob("*.npz"))
    with np.load(entry, allow_pickle=False) as file:
        arrays = {key: file[key] for key in file.files}
    arrays["xyz"][0, 0] = 999
    np.savez_compressed(entry, **arrays)
    with pytest.raises(ValueError, match="Invalid normalized cache"):
        ds[0]


def test_runtime_parsing_and_defaults():
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    options = runtime_from_args(
        parser.parse_args(
            [
                "--pin-memory",
                "--non-blocking",
                "--prefetch-factor",
                "3",
                "--persistent-workers",
            ]
        )
    )
    options.validate(2)
    assert options.pin_memory and options.non_blocking and options.prefetch_factor == 3
    assert options.persistent_workers
    with pytest.raises(ValueError):
        options.validate(0)
    assert asdict(runtime_from_args(parser.parse_args([]))) == asdict(RuntimeOptions())
    future = yaml.safe_load(
        Path("configs/pointnet2_future_options.example.yaml").read_text()
    )
    assert (
        not future["early_stopping"]["enabled"]
        and not future["amp"]["enabled"]
        and not future["tf32"]["enabled"]
    )


def test_cpu_timing_and_model_parity(tmp_path):
    manifest, root = fixture_data(tmp_path)
    records, _ = read_manifest(manifest)
    cfg = config()
    seed_everything(42)
    model = PointNet2Segmentation(PointNet2Config(**cfg.model))
    before = copy.deepcopy(model.state_dict())
    results = []
    for detailed in (False, True):
        loader = make_loader(
            make_dataset(
                records, root, "train", cfg, RuntimeOptions(detailed_timing=detailed)
            ),
            cfg,
            False,
            torch.Generator().manual_seed(42),
        )
        model.load_state_dict(before)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        results.append(
            run_epoch(
                model,
                loader,
                torch.device("cpu"),
                optimizer,
                detailed_timing=detailed,
                max_batches=1,
            )
        )
        if not detailed:
            expected_weights = copy.deepcopy(model.state_dict())
        else:
            for name, value in expected_weights.items():
                torch.testing.assert_close(
                    value, model.state_dict()[name], rtol=0, atol=0
                )
    assert results[0]["loss"] == results[1]["loss"]
    timing = results[1]["timing"]
    assert timing["batches"] == 1 and timing["samples_per_second"] > 0
    for stage in (
        "data_wait_ms",
        "h2d_ms",
        "forward_ms",
        "loss_ms",
        "backward_ms",
        "gradient_check_ms",
        "optimizer_ms",
        "metrics_ms",
        "total_batch_ms",
        "cpu_normalized_cloud_per_sample_ms",
    ):
        assert timing["stages"][stage]["mean"] >= 0
    assert "timing" not in results[0]
    assert telemetry("cpu")["allocated_bytes"] is None


def test_profile_bounded_readonly_and_trace(tmp_path):
    manifest, checkpoint, records, cfg = fixture_checkpoint(tmp_path)
    before = checkpoint.read_bytes()
    result = profile_workload(
        str(manifest),
        str(tmp_path),
        str(tmp_path / "profile"),
        checkpoint_path=str(checkpoint),
        split="val",
        device_name="cpu",
        warmup_batches=1,
        profile_batches=1,
        runtime=RuntimeOptions(detailed_timing=True),
    )
    assert result["profile_batches"] == 1 and result["warmup_batches"] == 1
    assert (tmp_path / "profile/trace.json").exists() and (
        tmp_path / "profile/operators.txt"
    ).exists()
    assert (
        result["operators"]["families"]["fps"]
        and result["operators"]["families"]["first_sa"]
    )
    assert checkpoint.read_bytes() == before
    loader = make_loader(
        make_dataset(records, tmp_path, "val", cfg),
        cfg,
        False,
        torch.Generator().manual_seed(42),
    )
    model = PointNet2Segmentation(PointNet2Config(**cfg.model)).eval()
    state = copy.deepcopy(model.state_dict())
    seed_everything(21)
    rng_before = torch.get_rng_state().clone()
    profile_model(model, loader, torch.device("cpu"), 0, 1, training=True)
    assert not model.training and torch.equal(rng_before, torch.get_rng_state())
    for name, value in state.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="exceed"):
        profile_model(model, loader, torch.device("cpu"), 10, 10)


def test_profile_exact_batch_access_count():
    class Counting(torch.utils.data.Dataset):
        def __init__(self):
            self.calls = 0

        def __len__(self):
            return 20

        def __getitem__(self, index):
            self.calls += 1
            return {
                "points": torch.ones(8, 3),
                "labels": torch.zeros(8, dtype=torch.long),
                "valid_mask": torch.ones(8, dtype=torch.bool),
                "sample_id": str(index),
            }

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)
            self.abstraction = torch.nn.ModuleList()
            self.propagation = torch.nn.ModuleList()

        def forward(self, points):
            return self.linear(points)

    dataset = Counting()
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    _, result = profile_model(Tiny(), loader, torch.device("cpu"), 1, 2, training=True)
    assert dataset.calls == 6 and result["profile_batches"] == 2


def test_benchmark_matrix_cache_and_test_protection(tmp_path):
    manifest, root = fixture_data(tmp_path)
    result = benchmark_loaders(
        manifest,
        root,
        tmp_path / "bench",
        config(),
        workers=(0,),
        pins=(False, True),
        warmup_batches=0,
        batches=1,
        cache_dir=str(tmp_path / "cache"),
        compare_cache=True,
    )
    assert len(result["results"]) == 6
    assert all(row["status"] == "ok" for row in result["results"])
    for split in ("test", "invalid"):
        with pytest.raises(ValueError, match="test"):
            profile_workload("missing", "missing", "missing", split=split)
        with pytest.raises(ValueError, match="test"):
            benchmark_loaders("missing", "missing", "missing", config(), split=split)


def test_resume_from_70_next_is_71_no_gate_required(tmp_path):
    manifest, root = fixture_data(tmp_path)
    directory = tmp_path / "run"
    cfg = replace(config(1), overfit_samples=0, augmentation={"enabled": True})
    _, digest = read_manifest(manifest)
    gate = tmp_path / "gate.pt"
    # Structural fixture, not a produced tiny-overfit performance claim.
    save_checkpoint(
        gate,
        {
            "mode": "tiny_overfit",
            "overfit_status": {"passed": True},
            "contract": experiment_contract(replace(cfg, augmentation={}), digest),
            "train_sample_ids": ["0", "1"],
        },
    )
    train(manifest, root, str(directory), cfg, overfit_checkpoint=str(gate))
    checkpoint = load_checkpoint(directory / "last.pt")
    checkpoint["epoch"] = 70
    checkpoint["config"]["epochs"] = 100
    # An epoch counter fixture checks continuation arithmetic, not 70 epochs of training.
    checkpoint.pop("runtime", None)  # Historical checkpoint predates Phase 3P.
    save_checkpoint(directory / "last.pt", checkpoint)
    resumed = replace(cfg, epochs=71)
    train(
        manifest,
        root,
        str(directory),
        resumed,
        resume=str(directory / "last.pt"),
        runtime=RuntimeOptions(
            cache_dir=str(tmp_path / "cache"), detailed_timing=True, telemetry=True
        ),
    )
    after = load_checkpoint(directory / "last.pt")
    assert after["epoch"] == 71 and after["history"][-1]["epoch"] == 71
    assert "telemetry" in after["history"][-1]
    assert all(
        key in after
        for key in (
            "optimizer_state",
            "rng",
            "loader_rng",
            "history",
            "best_miou",
            "best_loss",
            "contract",
        )
    )


def test_inactive_ranges_numerical_identity():
    from agri3d.models.pointnet2_ops import farthest_point_sample

    xyz = torch.randn(2, 10, 3)
    plain = farthest_point_sample(xyz, 4)
    with profiling_ranges():
        ranged = farthest_point_sample(xyz, 4)
    torch.testing.assert_close(plain, ranged, rtol=0, atol=0)


def test_nsight_missing_gracefully(monkeypatch):
    specification = importlib.util.spec_from_file_location(
        "nsys_helper", "scripts/nsys_pointnet2.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    status = module.nsight_status()
    assert status["available"] is False and "torch profiler" in status["reason"]


def test_persistent_loader_rng_draw_without_workers(monkeypatch):
    # Unit check for RNG accounting; real spawn/epoch propagation is checked below.
    loader = object.__new__(EpochDataLoader)
    object.__setattr__(loader, "persistent_workers", True)
    object.__setattr__(loader, "generator", torch.Generator().manual_seed(42))
    monkeypatch.setattr(torch.utils.data.DataLoader, "__iter__", lambda _: iter(()))
    expected = torch.Generator().manual_seed(42)
    iter(loader)
    assert torch.equal(loader.generator.get_state(), expected.get_state())
    iter(loader)
    torch.empty((), dtype=torch.int64).random_(generator=expected)
    assert torch.equal(loader.generator.get_state(), expected.get_state())


def test_persistent_workers_epoch_and_rng_parity(tmp_path):
    try:
        sock = socket.socket(socket.AF_UNIX)
        sock.close()
    except PermissionError:
        pytest.skip("Workspace blocks Unix sockets required for tensor workers")
    manifest, root = fixture_data(tmp_path)
    records, _ = read_manifest(manifest)
    cfg = replace(
        config(), num_workers=2, overfit_samples=0, augmentation={"enabled": True}
    )
    datasets = [make_dataset(records, root, "train", cfg) for _ in range(2)]
    generators = [torch.Generator().manual_seed(42) for _ in range(2)]
    loaders = [
        make_loader(
            dataset, cfg, True, gen, RuntimeOptions(persistent_workers=bool(index))
        )
        for index, (dataset, gen) in enumerate(zip(datasets, generators))
    ]
    previous = None
    for epoch in (0, 1):
        for dataset in datasets:
            dataset.set_epoch(epoch)
        batches = [list(loader) for loader in loaders]
        for a, b in zip(*batches):
            equivalent(a, b)
        assert torch.equal(generators[0].get_state(), generators[1].get_state())
        if previous is not None:
            assert not torch.equal(previous, batches[1][0]["points"])
        previous = batches[1][0]["points"]


def test_default_timing_never_synchronizes(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *_a, **_k: pytest.fail("Unexpected CUDA sync")
    )
    timer = BatchTimings("cuda", False)
    timer.begin()
    with timer.stage("forward"):
        pass
    timer.finish({})
    assert timer.rows == []


def test_runtime_cache_and_timing_preserve_exact_regular_resume(tmp_path):
    manifest, root = fixture_data(tmp_path)
    _, digest = read_manifest(manifest)
    cfg = replace(config(2), overfit_samples=0, augmentation={"enabled": True})
    gate = tmp_path / "gate.pt"
    save_checkpoint(
        gate,
        {
            "mode": "tiny_overfit",
            "overfit_status": {"passed": True},
            "contract": experiment_contract(replace(cfg, augmentation={}), digest),
            "train_sample_ids": ["0", "1"],
        },
    )
    a, b = tmp_path / "default", tmp_path / "resumed"
    train(manifest, root, str(a), cfg, overfit_checkpoint=str(gate))
    train(manifest, root, str(b), replace(cfg, epochs=1), overfit_checkpoint=str(gate))
    legacy = load_checkpoint(b / "last.pt")
    legacy.pop("runtime")
    save_checkpoint(b / "last.pt", legacy)
    train(
        manifest,
        root,
        str(b),
        cfg,
        resume=str(b / "last.pt"),
        runtime=RuntimeOptions(
            cache_dir=str(tmp_path / "cache"),
            detailed_timing=True,
            non_blocking=True,
            telemetry=True,
        ),
    )
    full, resumed = load_checkpoint(a / "last.pt"), load_checkpoint(b / "last.pt")
    for name, value in full["model_state"].items():
        torch.testing.assert_close(value, resumed["model_state"][name], rtol=0, atol=0)
    assert (
        full["history"][-1]["train"]["loss"] == resumed["history"][-1]["train"]["loss"]
    )
    assert full["contract"] == resumed["contract"]
    for name in full["loader_rng"]:
        torch.testing.assert_close(
            full["loader_rng"][name], resumed["loader_rng"][name], rtol=0, atol=0
        )


def test_nsight_workload_does_not_start_torch_profiler(tmp_path, monkeypatch):
    manifest, checkpoint, _, _ = fixture_checkpoint(tmp_path)
    monkeypatch.setattr(
        torch.profiler, "profile", lambda **_k: pytest.fail("Concurrent CUPTI profiler")
    )
    result = profile_workload(
        str(manifest),
        str(tmp_path),
        str(tmp_path / "nvtx"),
        checkpoint_path=str(checkpoint),
        split="val",
        device_name="cpu",
        warmup_batches=0,
        profile_batches=1,
        nvtx=True,
        use_torch_profiler=False,
    )
    assert result["profile_batches"] == 1
    assert not (tmp_path / "nvtx/trace.json").exists()
    assert "NVTX-only" in result["operators"]["note"]
