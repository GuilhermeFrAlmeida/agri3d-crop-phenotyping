"""Bounded performance plumbing, not evidence of GPU speedups."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_pointnet2_training import config, fixture_data

from agri3d.models import PointNet2Config, PointNet2Segmentation
from agri3d.performance.benchmark import benchmark_loaders
from agri3d.performance.profiling import profile_model
from agri3d.performance.runtime import RuntimeOptions
from agri3d.performance.throughput import benchmark_batch_sizes
from agri3d.training.checkpoints import capture_rng, seed_everything
from agri3d.training.engine import run_epoch
from agri3d.training.pointnet2_runner import (
    experiment_contract,
    make_dataset,
    make_loader,
    read_manifest,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def larger_fixture(tmp_path):
    manifest, root = fixture_data(tmp_path)
    records = json.loads(Path(manifest).read_text())
    for i in range(5, 17):
        (tmp_path / f"{i}.ply").write_bytes((tmp_path / "0.ply").read_bytes())
        records.append(
            {
                "sample_id": str(i),
                "path": f"{i}.ply",
                "group_id": str(i),
                "split": "train",
            }
        )
    Path(manifest).write_text(json.dumps(records))
    return manifest, root


def test_one_window_retains_all_batches_and_ranges(tmp_path, monkeypatch):
    manifest, root = larger_fixture(tmp_path)
    records, _ = read_manifest(manifest)
    cfg = config()
    loader = make_loader(
        make_dataset(records, root, "train", cfg),
        cfg,
        False,
        torch.Generator().manual_seed(42),
    )
    model = PointNet2Segmentation(PointNet2Config(**cfg.model))
    before = copy.deepcopy(model.state_dict())
    monkeypatch.setattr(
        torch.profiler.profile,
        "step",
        lambda *_a, **_k: pytest.fail("cycle step must not clear events"),
    )
    _, report = profile_model(
        model,
        loader,
        "cpu",
        1,
        3,
        training=True,
        runtime=RuntimeOptions(detailed_timing=True),
    )
    operators = report["operators"]
    forward = next(r for r in operators["operators"] if r["name"] == "agri3d::forward")
    assert forward["calls"] == 3
    assert operators["family_totals"]["fps"]["calls"] == 12
    assert len(operators["set_abstraction"]) == 4
    assert len(operators["stage_fps"]) == 4
    assert all(r["calls"] == 3 for r in operators["stage_fps"])
    assert operators["set_abstraction"][0]["input_N"] == cfg.points
    assert operators["set_abstraction"][0]["centroids"] == cfg.model["npoints"][0]
    assert operators["family_totals"]["h2d"]["calls"] == 3
    assert operators["forward_components"]["neural_mlp"]["calls"] == 24
    for name in (
        "agri3d::h2d",
        "agri3d::forward",
        "agri3d::backward",
        "agri3d::optimizer",
    ):
        row = next(r for r in operators["operators"] if r["name"] == name)
        assert row["calls"] == 3 and row["cpu_mean_us"] > 0
        assert row["cuda_total_us"] == 0
        assert "self_cpu_us" in row and "self_cuda_us" in row
    shares = report["metrics"]["timing"]["stage_percent_of_batch"]
    assert all(
        k in shares
        for k in ("data_wait", "h2d", "forward", "backward", "optimizer", "metrics")
    )
    assert (
        99.9
        <= sum(shares.values()) + report["metrics"]["timing"]["unattributed_percent"]
        <= 100.1
    )
    for name, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)


def test_loader_matrix_fingerprints_cache_population_warm(tmp_path):
    manifest, root = fixture_data(tmp_path)
    cfg = replace(config(), overfit_samples=0, augmentation={"enabled": True})
    result = benchmark_loaders(
        manifest,
        root,
        tmp_path / "matrix",
        cfg,
        workers=(0,),
        pins=(False, True),
        warmup_batches=0,
        batches=1,
        cache_dir=str(tmp_path / "cache"),
        compare_cache=True,
        matrix=True,
    )
    rows = result["results"]
    assert len(rows) == 12 and all(r["status"] == "ok" for r in rows)
    assert len({tuple(r["sample_fingerprints"]) for r in rows}) == 1
    for row in rows:
        assert row["identity_verified"]
        assert row["plants_per_second"] > 0
        assert row["batch_preparation_p95_ms"] >= row["batch_preparation_median_ms"]
        assert len(row["passes"]) == 2
        if row["mode"] == "cache_population":
            assert row["cache_hits"] == 0 and row["cache_misses"] == 2
        if row["mode"] == "cache_warm":
            assert row["cache_hits"] == 2 and row["cache_misses"] == 0
    with pytest.raises(ValueError, match="empty cache"):
        benchmark_loaders(
            manifest,
            root,
            tmp_path / "bad",
            cfg,
            workers=(0,),
            pins=(False,),
            cache_dir=str(tmp_path / "cache"),
            compare_cache=True,
        )


def test_cache_manifest_identity_and_pre_sampling_equivalence(tmp_path):
    manifest, root = fixture_data(tmp_path)
    records, _ = read_manifest(manifest)
    cfg = config()
    plain = make_dataset(records, root, "train", cfg)
    cached = make_dataset(
        records, root, "train", cfg, RuntimeOptions(cache_dir=str(tmp_path / "cache"))
    )
    expected = plain._normalized_cloud(tmp_path / "0.ply")
    actual = cached.cache.get(
        tmp_path / "0.ply",
        lambda: cached._normalized_cloud(tmp_path / "0.ply"),
        {"sample_id": "0", "manifest": "one"},
    )
    import numpy as np

    np.testing.assert_array_equal(actual.xyz, expected.xyz)
    for key in expected.attributes:
        np.testing.assert_array_equal(actual.attributes[key], expected.attributes[key])
    assert cached.cache.last_hit is False
    cached.cache.get(
        tmp_path / "0.ply",
        lambda: pytest.fail("cache miss"),
        {"sample_id": "0", "manifest": "one"},
    )
    assert cached.cache.last_hit is True
    cached.cache.get(
        tmp_path / "0.ply", lambda: expected, {"sample_id": "0", "manifest": "two"}
    )
    assert cached.cache.last_hit is False
    cached[0]
    entries = list((tmp_path / "cache").glob("*.npz"))
    assert len(entries) == 3
    identities = [
        json.loads(str(np.load(p, allow_pickle=False)["identity"])) for p in entries
    ]
    assert any(
        i["sample_identity"].get("manifest_sha256") == cached.manifest_hash
        for i in identities
    )
    assert all(
        i["cache_version"] == 2 and i["storage"] == "uncompressed_npz"
        for i in identities
    )


def test_transfer_and_pinning_paths_preserve_validation_and_updates(
    tmp_path, monkeypatch
):
    manifest, root = fixture_data(tmp_path)
    records, digest = read_manifest(manifest)
    cfg = config()
    contract = experiment_contract(cfg, digest)
    original_to = torch.Tensor.to
    calls = []

    def traced_to(self, *args, **kwargs):
        if "non_blocking" in kwargs:
            calls.append(kwargs["non_blocking"])
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", traced_to)
    seed_everything(42)
    model = PointNet2Segmentation(PointNet2Config(**cfg.model))
    original = copy.deepcopy(model.state_dict())
    outputs, weights = [], []
    for runtime in (
        RuntimeOptions(),
        RuntimeOptions(
            pin_memory=True, non_blocking=True, cache_dir=str(tmp_path / "cache")
        ),
    ):
        model.load_state_dict(original)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        loader = make_loader(
            make_dataset(records, root, "train", cfg, runtime),
            cfg,
            False,
            torch.Generator().manual_seed(42),
            runtime,
        )
        assert loader.pin_memory == runtime.pin_memory
        run_epoch(
            model,
            loader,
            torch.device("cpu"),
            optimizer,
            non_blocking=runtime.non_blocking,
        )
        weights.append(copy.deepcopy(model.state_dict()))
        val = make_loader(
            make_dataset(records, root, "val", cfg, runtime),
            cfg,
            False,
            torch.Generator().manual_seed(43),
            runtime,
        )
        outputs.append(
            run_epoch(
                model, val, torch.device("cpu"), non_blocking=runtime.non_blocking
            )
        )
    assert True in calls and False in calls
    for key in weights[0]:
        torch.testing.assert_close(weights[0][key], weights[1][key], atol=0, rtol=0)
    assert {k: v for k, v in outputs[0].items() if k != "seconds"} == {
        k: v for k, v in outputs[1].items() if k != "seconds"
    }
    assert experiment_contract(cfg, digest) == contract


def test_batch_benchmark_bounded_readonly_and_contract(tmp_path):
    manifest, root = larger_fixture(tmp_path)
    cfg = config()
    before = capture_rng()["torch"]
    result = benchmark_batch_sizes(
        manifest,
        root,
        tmp_path / "throughput",
        cfg,
        batch_sizes=(2, 4, 8),
        warmup_batches=0,
        batches=1,
        device_name="cpu",
    )
    assert torch.equal(before, capture_rng()["torch"])
    assert cfg.batch_size == 2
    assert result["scientific_contract"]["batch_size"] == 2
    for row in result["results"]:
        assert row["status"] == "ok"
        assert row["throughput"]["metrics"]["batches"] == 1
        assert row["throughput"]["samples_per_second"] > 0
        assert "timing" not in row["throughput"]["metrics"]
        assert row["diagnostic"]["metrics"]["timing"]["batches"] == 1
        assert row["forward_median_ms"] > 0 and row["backward_median_ms"] > 0
        assert row["throughput"]["telemetry"]["peak_allocated_bytes"] is None
    assert not list((tmp_path / "throughput").glob("*.pt"))
    assert not (tmp_path / "4.ply").exists()
    with pytest.raises(ValueError, match="test"):
        benchmark_batch_sizes("missing", "missing", "missing", cfg, split="test")


@pytest.mark.parametrize(
    "runtime,workers",
    [
        (RuntimeOptions(), True),
        (RuntimeOptions(prefetch_factor=0), 2),
        (RuntimeOptions(persistent_workers=True), 0),
        (RuntimeOptions(pin_memory=1), 0),
    ],
)
def test_invalid_loader_controls(runtime, workers):
    with pytest.raises((TypeError, ValueError)):
        runtime.validate(workers)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for actual pinned H2D equivalence",
)
def test_cuda_pinned_nonblocking_exact_update_and_validation(tmp_path):
    manifest, root = fixture_data(tmp_path)
    records, _ = read_manifest(manifest)
    cfg = config()
    seed_everything(42)
    model = PointNet2Segmentation(PointNet2Config(**cfg.model)).cuda()
    original = copy.deepcopy(model.state_dict())
    states, metrics = [], []
    for runtime in (
        RuntimeOptions(),
        RuntimeOptions(
            pin_memory=True, non_blocking=True, cache_dir=str(tmp_path / "cache")
        ),
    ):
        model.load_state_dict(original)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        loader = make_loader(
            make_dataset(records, root, "train", cfg, runtime),
            cfg,
            False,
            torch.Generator().manual_seed(42),
            runtime,
        )
        if runtime.pin_memory:
            assert next(iter(loader))["points"].is_pinned()
        run_epoch(
            model, loader, torch.device("cuda"), opt, non_blocking=runtime.non_blocking
        )
        states.append(copy.deepcopy(model.state_dict()))
        val = make_loader(
            make_dataset(records, root, "val", cfg, runtime),
            cfg,
            False,
            torch.Generator().manual_seed(43),
            runtime,
        )
        result = run_epoch(
            model, val, torch.device("cuda"), non_blocking=runtime.non_blocking
        )
        metrics.append({k: v for k, v in result.items() if k != "seconds"})
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)
    assert metrics[0] == metrics[1]


def test_throughput_checkpoint_is_readonly(tmp_path):
    from agri3d.training.checkpoints import save_checkpoint

    manifest, root = fixture_data(tmp_path)
    cfg = replace(config(), overfit_samples=0)
    _, digest = read_manifest(manifest)
    model = PointNet2Segmentation(PointNet2Config(**cfg.model))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    checkpoint = tmp_path / "snapshot.pt"
    save_checkpoint(
        checkpoint,
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "contract": experiment_contract(cfg, digest),
            "epoch": 0,
        },
    )
    before = checkpoint.read_bytes()
    report = benchmark_batch_sizes(
        manifest,
        root,
        tmp_path / "bench_ckpt",
        cfg,
        batch_sizes=(2,),
        warmup_batches=0,
        batches=1,
        device_name="cpu",
        checkpoint_path=str(checkpoint),
    )
    assert report["results"][0]["status"] == "ok"
    assert checkpoint.read_bytes() == before
    assert report["checkpoint_sha256"]
