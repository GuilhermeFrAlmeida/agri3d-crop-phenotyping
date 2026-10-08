"""End-to-end synthetic PLY runs; real data/test split is never used."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from agri3d.training.checkpoints import (
    capture_rng,
    load_checkpoint,
    restore_rng,
    seed_everything,
)
from agri3d.training.pointnet2_runner import (
    RunConfig,
    choose_overfit_indices,
    experiment_contract,
    make_dataset,
    overfit_status,
    read_manifest,
    train,
    validate_checkpoint,
    verify_overfit_gate,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_data(root):
    records = []
    for i, split in enumerate(["train", "train", "train", "val", "test"]):
        name = f"{i}.ply"
        if split != "test":  # Test data deliberately absent; must not be required/read.
            rng = np.random.default_rng(i)
            xyz = rng.normal(size=(40, 3)).astype(np.float32)
            data = np.zeros(
                40,
                dtype=[
                    ("x", "f4"),
                    ("y", "f4"),
                    ("z", "f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ],
            )
            for j, key in enumerate(("x", "y", "z")):
                data[key] = xyz[:, j]
            data["green"] = np.where(xyz[:, 2] > 0, 0, 255)
            PlyData([PlyElement.describe(data, "vertex")], text=True).write(
                str(root / name)
            )
        records.append(
            {"path": name, "sample_id": str(i), "group_id": str(i), "split": split}
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(records))
    return str(manifest), str(root)


def config(epochs=1):
    return RunConfig(
        points=24,
        epochs=epochs,
        batch_size=2,
        overfit_samples=2,
        device="cpu",
        model={"npoints": [12, 6, 3, 2], "nsamples": [4, 4, 4, 4], "chunk_size": 6},
    )


def test_rng_roundtrip():
    seed_everything(2)
    state = capture_rng()
    expected = (random.random(), np.random.rand(), torch.rand(4))
    restore_rng(state)
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)


def test_subset_train_only_and_gate(tmp_path):
    manifest, root = fixture_data(tmp_path)
    records, digest = read_manifest(manifest)
    ds = make_dataset(records, root, "train", config())
    idx = choose_overfit_indices(ds, 2, 42)
    assert idx == choose_overfit_indices(ds, 2, 42)
    assert all(ds.records[i]["split"] == "train" for i in idx)
    with pytest.raises(ValueError):
        make_dataset(records, root, "test", config())
    contract = experiment_contract(config(), digest)
    with pytest.raises(ValueError):
        verify_overfit_gate({}, contract)
    with pytest.raises(ValueError):
        train(manifest, root, str(tmp_path / "regular"), RunConfig())
    assert overfit_status(1.0, {"loss": 0.1, "accuracy": 0.99, "mIoU": 0.98})["passed"]
    assert not overfit_status(1.0, {"loss": 0.1, "accuracy": 0.90, "mIoU": 0.98})[
        "passed"
    ]


def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    manifest, root = fixture_data(tmp_path)
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    train(manifest, root, str(uninterrupted), config(2))
    train(manifest, root, str(resumed), config(1))
    train(manifest, root, str(resumed), config(2), resume=str(resumed / "last.pt"))
    a, b = (
        load_checkpoint(uninterrupted / "last.pt"),
        load_checkpoint(resumed / "last.pt"),
    )
    assert b["epoch"] == 2 and b["selection_metric"] == "train_mIoU"
    assert len(b["train_sample_ids"]) == 2 and all(
        x in ["0", "1", "2"] for x in b["train_sample_ids"]
    )
    assert b["initial_metrics"]["loss"] == a["initial_metrics"]["loss"]
    for name, value in a["model_state"].items():
        torch.testing.assert_close(value, b["model_state"][name], rtol=0, atol=0)
    assert (
        a["history"][-1]["train_eval"]["loss"] == b["history"][-1]["train_eval"]["loss"]
    )
    assert (resumed / "best.pt").exists() and (resumed / "metrics.json").exists()
    with pytest.raises(ValueError, match="regular-run"):
        validate_checkpoint(
            manifest, root, str(resumed / "best.pt"), str(tmp_path / "val.json")
        )
    changed = config(3)
    changed.points = 25
    with pytest.raises(ValueError, match="contract differs"):
        train(manifest, root, str(resumed), changed, resume=str(resumed / "last.pt"))


def test_regular_validation_and_test_exclusion(tmp_path, monkeypatch):
    # Gate verification is tested separately. This isolates regular orchestration
    # without claiming a one-epoch synthetic run meets the overfit thresholds.
    manifest, root = fixture_data(tmp_path)
    tiny = tmp_path / "tiny"
    train(manifest, root, str(tiny), config())
    monkeypatch.setattr(
        "agri3d.training.pointnet2_runner.verify_overfit_gate", lambda *_: None
    )
    cfg = config()
    cfg.overfit_samples = 0
    out = tmp_path / "regular"
    train(manifest, root, str(out), cfg, overfit_checkpoint=str(tiny / "last.pt"))
    saved = load_checkpoint(out / "best.pt")
    assert saved["selection_metric"] == "val_mIoU"
    assert saved["history"][0]["val"]["evaluated_points"] == 24
    result = validate_checkpoint(
        manifest, root, str(out / "best.pt"), str(tmp_path / "validation.json"), "cpu"
    )
    assert result["split"] == "val"
    assert result["metrics"]["mIoU"] == saved["history"][0]["val"]["mIoU"]
    assert not (Path(root) / "4.ply").exists()
    Path(manifest).write_text(Path(manifest).read_text() + "\n")
    with pytest.raises(ValueError, match="manifest differs"):
        validate_checkpoint(
            manifest, root, str(out / "best.pt"), str(tmp_path / "bad.json"), "cpu"
        )
