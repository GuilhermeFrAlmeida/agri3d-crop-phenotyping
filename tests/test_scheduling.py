"""Controlled scheduler behavior and exact CPU recovery; synthetic train/val only."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from test_pointnet2_training import config, fixture_data

from agri3d.analysis.pointnet2 import analyze_checkpoint
from agri3d.performance.inspection import inspect_checkpoint
from agri3d.training import pointnet2_runner as runner
from agri3d.training.checkpoints import load_checkpoint, save_checkpoint
from agri3d.training.scheduling import SchedulerConfig


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_run3_only_scheduler_changes():
    a = yaml.safe_load(Path("configs/pointnet2.yaml").read_text())
    b = yaml.safe_load(Path("configs/pointnet2_run3_scheduler.yaml").read_text())
    settings = b.pop("scheduler")
    assert a == b
    assert SchedulerConfig.from_settings(settings) == SchedulerConfig(enabled=True)
    legacy = runner.experiment_contract(runner.RunConfig(), "manifest")
    assert "scheduler" not in legacy
    assert legacy == runner.experiment_contract(
        runner.RunConfig(scheduler={"enabled": False}), "manifest"
    )
    assert (
        SchedulerConfig().build(torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]))
        is None
    )


def test_plateau_absolute_threshold_and_floor():
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.001)
    scheduler = SchedulerConfig(enabled=True).build(opt)
    scheduler.step(0.8)  # establishes baseline
    for _ in range(8):
        scheduler.step(0.8004)  # below absolute improvement threshold
        assert opt.param_groups[0]["lr"] == 0.001
    scheduler.step(0.8004)  # patience=8 permits 8 bad epochs; reduces on 9th
    assert opt.param_groups[0]["lr"] == 0.0005
    scheduler.step(0.801)  # significant improvement resets bad count
    assert scheduler.num_bad_epochs == 0
    for _ in range(100):
        scheduler.step(0.801)
    assert opt.param_groups[0]["lr"] == 0.00001


@pytest.mark.parametrize(
    "settings",
    [
        {"enabled": "true"},
        {"name": "cosine"},
        {"mode": "min"},
        {"threshold_mode": "rel"},
        {"factor": 1},
        {"patience": -1},
        {"patience": True},
        {"threshold": float("nan")},
        {"min_lr": -1},
        {"cooldown": 1.5},
        {"eps": True},
        {"monitor": "train_loss"},
    ],
)
def test_invalid_scheduler(settings):
    with pytest.raises((TypeError, ValueError)):
        SchedulerConfig.from_settings(settings)


def structural_gate(tmp_path, cfg, digest):
    gate = tmp_path / "gate.pt"
    save_checkpoint(
        gate,
        {
            "mode": "tiny_overfit",
            "overfit_status": {"passed": True},
            "contract": runner.experiment_contract(
                replace(cfg, scheduler={}, augmentation={}), digest
            ),
            "train_sample_ids": ["0", "1"],
        },
    )
    return str(gate)


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            if key != "seconds":  # Wall-clock durations are not reproducible state.
                assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


def test_scheduled_resume_monitoring_and_compatibility(tmp_path, monkeypatch):
    manifest, root = fixture_data(tmp_path)
    _, digest = runner.read_manifest(manifest)
    cfg = replace(
        config(4),
        overfit_samples=0,
        scheduler={"enabled": True, "patience": 0, "threshold": 1.0},
    )
    gate = structural_gate(tmp_path, cfg, digest)
    real_epoch = runner.run_epoch
    monitored = []
    real_step = torch.optim.lr_scheduler.ReduceLROnPlateau.step

    def step(self, value, *args, **kwargs):
        monitored.append(value)
        return real_step(self, value, *args, **kwargs)

    def epoch(*args, **kwargs):
        result = real_epoch(*args, **kwargs)
        # Controlled validation plateau with genuine parameter/Adam updates.
        # Training mIoU differs, so stepping on train metrics fails the spy check.
        if len(args) == 3:
            result["mIoU"] = 0.6
        else:
            result["mIoU"] = 0.9
        return result

    monkeypatch.setattr(runner, "run_epoch", epoch)
    monkeypatch.setattr(torch.optim.lr_scheduler.ReduceLROnPlateau, "step", step)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    runner.train(manifest, root, str(full), cfg, overfit_checkpoint=gate)
    runner.train(
        manifest, root, str(resumed), replace(cfg, epochs=2), overfit_checkpoint=gate
    )
    partial = load_checkpoint(resumed / "last.pt")
    assert partial["scheduler_state"]["num_bad_epochs"] == 0
    assert partial["optimizer_state"]["param_groups"][0]["lr"] == 0.0005
    runner.train(manifest, root, str(resumed), cfg, resume=str(resumed / "last.pt"))
    a, b = load_checkpoint(full / "last.pt"), load_checkpoint(resumed / "last.pt")
    assert monitored == [0.6] * 8
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "loader_rng",
        "history",
        "best_miou",
        "best_loss",
    ):
        assert_equal(a[key], b[key])
    assert_equal(a["rng"], b["rng"])
    assert [r["lr"] for r in a["history"]] == [0.001, 0.001, 0.0005, 0.00025]
    assert [r["next_lr"] for r in a["history"]] == [0.001, 0.0005, 0.00025, 0.000125]
    assert a["contract"]["scheduler"]["monitor"] == "val_mIoU"
    assert load_checkpoint(full / "best.pt")["scheduler_state"] is not None
    import json

    summary = json.loads((full / "summary.json").read_text())
    assert summary["lr_reduction_epochs"] == [2, 3, 4]
    assert summary["final_lr"] == 0.000125
    assert summary["lr_at_best_epoch"] == a["history"][summary["best_epoch"] - 1]["lr"]
    inspection = inspect_checkpoint(full / "last.pt")
    assert inspection["scheduler_contract"] == a["contract"]["scheduler"]
    for settings in (
        {},
        {"enabled": True, "patience": 1},
        {"enabled": True, "threshold": 0.1},
    ):
        with pytest.raises(ValueError, match="contract differs"):
            runner.train(
                manifest,
                root,
                str(resumed),
                replace(cfg, epochs=5, scheduler=settings),
                resume=str(resumed / "last.pt"),
            )
    missing = dict(b)
    missing.pop("scheduler_state")
    save_checkpoint(resumed / "last.pt", missing)
    with pytest.raises(ValueError, match="scheduler_state"):
        runner.train(
            manifest,
            root,
            str(resumed),
            replace(cfg, epochs=5),
            resume=str(resumed / "last.pt"),
        )
    # Same contract is accepted by validation and per-plant analysis.
    runner.validate_checkpoint(
        manifest, root, str(full / "best.pt"), str(tmp_path / "val.json"), "cpu"
    )
    analyze_checkpoint(
        manifest,
        root,
        str(full / "best.pt"),
        str(tmp_path / "analysis"),
        device_name="cpu",
    )
    assert not (tmp_path / "4.ply").exists()
    with pytest.raises(ValueError, match="train/val"):
        runner.make_dataset([], root, "test", cfg)
    with pytest.raises(ValueError, match="Tiny-overfit"):
        replace(cfg, overfit_samples=2).validate()


@pytest.mark.parametrize("augmentation", [{}, {"enabled": True}])
def test_disabled_matches_legacy_checkpoint_training(tmp_path, augmentation):
    manifest, root = fixture_data(tmp_path)
    _, digest = runner.read_manifest(manifest)
    cfg = replace(config(2), overfit_samples=0, augmentation=augmentation)
    gate = structural_gate(tmp_path, cfg, digest)
    full, legacy = tmp_path / "full", tmp_path / "legacy"
    runner.train(manifest, root, str(full), cfg, overfit_checkpoint=gate)
    runner.train(
        manifest, root, str(legacy), replace(cfg, epochs=1), overfit_checkpoint=gate
    )
    saved = load_checkpoint(legacy / "last.pt")
    saved["config"].pop("scheduler")
    saved.pop("scheduler_state")
    if not augmentation:
        saved.pop("contract_sha256")  # Earliest Run 1 format.
    save_checkpoint(legacy / "last.pt", saved)
    runner.validate_checkpoint(
        manifest, root, str(legacy / "last.pt"), str(tmp_path / "val.json"), "cpu"
    )
    analyze_checkpoint(
        manifest,
        root,
        str(legacy / "last.pt"),
        str(tmp_path / "analysis"),
        device_name="cpu",
    )
    runner.train(
        manifest,
        root,
        str(legacy),
        replace(cfg, scheduler={"enabled": False}),
        resume=str(legacy / "last.pt"),
    )
    a, b = load_checkpoint(full / "last.pt"), load_checkpoint(legacy / "last.pt")
    for key in ("model_state", "optimizer_state", "loader_rng", "history"):
        assert_equal(a[key], b[key])
    assert all(row["lr"] == 0.001 and not row["lr_changed"] for row in b["history"])
    assert b["scheduler_state"] is None
