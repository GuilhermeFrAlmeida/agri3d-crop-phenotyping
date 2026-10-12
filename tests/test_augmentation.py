"""Geometry/alignment and Run 1/Run 2 contract regressions on synthetic data."""

import hashlib
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
import yaml
from test_pointnet2_training import config, fixture_data

from agri3d.analysis.pointnet2 import analyze_checkpoint
from agri3d.io.ply import PLYReader, PointCloudData
from agri3d.preprocessing.augmentation import (
    AugmentationConfig,
    augment_cloud,
    augment_xyz,
)
from agri3d.preprocessing.filtering import CloudFilter
from agri3d.preprocessing.labels import decode_maizefield3d_labels
from agri3d.preprocessing.normalization import CloudNormalizer
from agri3d.preprocessing.sampling import FixedSizeSampler
from agri3d.training.checkpoints import load_checkpoint, save_checkpoint
from agri3d.training.dataset import SegmentationDataset
from agri3d.training.pointnet2_runner import (
    RunConfig,
    contract_sha256,
    experiment_contract,
    make_dataset,
    read_manifest,
    train,
    validate_checkpoint,
    verify_checkpoint_contract,
    verify_overfit_gate,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def cloud():
    xyz = np.random.default_rng(3).normal(size=(20, 3)).astype(np.float32)
    return PointCloudData(
        xyz,
        {
            "label": np.arange(20, dtype=np.int64) % 2,
            "valid_mask": np.array([True] * 16 + [False] * 4),
            "source_index": np.arange(20, dtype=np.int64),
            "colors": np.zeros((20, 3), dtype=np.uint8),
        },
        {"centroid_shift": [1.0, 2.0, 3.0], "scale_factor": 2.0},
    )


def test_z_rotation_distances_and_vertical_axis():
    xyz = cloud().xyz
    cfg = AugmentationConfig(enabled=True, scale_min=1.0, scale_max=1.0, jitter_sigma=0)
    rotated, applied = augment_xyz(xyz, np.random.default_rng(4), cfg)
    before = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    after = np.linalg.norm(rotated[:, None] - rotated[None], axis=-1)
    np.testing.assert_allclose(after, before, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(rotated[:, 2], xyz[:, 2])
    assert 0 <= applied["rotation_z_radians"] < 2 * np.pi
    assert not np.array_equal(rotated[:, :2], xyz[:, :2])


def test_isotropic_scale_range_and_fixed_factor():
    xyz = cloud().xyz
    cfg = AugmentationConfig(
        enabled=True, rotate_z=False, scale_min=1.02, scale_max=1.02, jitter_sigma=0
    )
    scaled, applied = augment_xyz(xyz, np.random.default_rng(4), cfg)
    np.testing.assert_allclose(scaled, xyz * np.float32(1.02), rtol=0, atol=0)
    assert applied["isotropic_scale"] == 1.02
    default = replace(cfg, scale_min=0.95, scale_max=1.05)
    for seed in range(10):
        scaled, applied = augment_xyz(xyz, np.random.default_rng(seed), default)
        assert 0.95 <= applied["isotropic_scale"] <= 1.05
        np.testing.assert_allclose(
            scaled, xyz * np.float32(applied["isotropic_scale"]), rtol=0, atol=0
        )


def test_jitter_clipped_and_duplicate_source_consistency():
    xyz = np.zeros((100, 3), dtype=np.float32)
    cfg = AugmentationConfig(
        enabled=True,
        rotate_z=False,
        scale_min=1,
        scale_max=1,
        jitter_sigma=0.5,
        jitter_clip=0.02,
    )
    result, _ = augment_xyz(xyz, np.random.default_rng(3), cfg)
    assert np.abs(result).max() <= np.float32(0.02)
    assert np.any(np.abs(result) == np.float32(0.02))
    assert np.std(result) > 0
    identities = np.repeat(np.arange(50), 2)
    result, _ = augment_xyz(xyz, np.random.default_rng(3), cfg, identities)
    np.testing.assert_array_equal(result[::2], result[1::2])


@pytest.mark.parametrize("enabled", [True, False])
def test_no_inplace_mutation_and_attribute_preservation(enabled):
    original = cloud()
    saved_xyz = original.xyz.copy()
    saved_attrs = {key: value.copy() for key, value in original.attributes.items()}
    rng = np.random.default_rng(4)
    result = augment_cloud(original, rng, AugmentationConfig(enabled=enabled))
    np.testing.assert_array_equal(original.xyz, saved_xyz)
    for key, value in saved_attrs.items():
        np.testing.assert_array_equal(result.attributes[key], value)
        assert not np.shares_memory(original.attributes[key], result.attributes[key])
    assert not np.shares_memory(result.xyz, original.xyz)
    result.metadata["centroid_shift"][0] = 99
    assert original.metadata["centroid_shift"] == [1.0, 2.0, 3.0]
    if not enabled:
        np.testing.assert_array_equal(result.xyz, original.xyz)
        assert (
            rng.random() == np.random.default_rng(4).random()
        )  # No RNG draw when disabled.
    else:
        assert not np.array_equal(result.xyz, original.xyz)


@pytest.mark.parametrize(
    "settings",
    [
        {"scale_min": 0},
        {"scale_min": 2, "scale_max": 1},
        {"jitter_sigma": -0.1},
        {"jitter_clip": -0.1},
        {"jitter_sigma": float("nan")},
        {"scale_min": True},
        {"enabled": "true"},
        {"rotate_z": 1},
        {"rotate_x": True},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises((ValueError, TypeError)):
        AugmentationConfig.from_settings(settings)


def test_dataset_seed_epoch_alignment_validation_and_legacy_behavior(tmp_path):
    manifest, root = fixture_data(tmp_path)
    kwargs = {
        "manifest": manifest,
        "root": root,
        "num_points": 64,
        "seed": 42,
        "annotation_mode": "maizefield3d",
    }
    baseline = SegmentationDataset(split="train", **kwargs)
    disabled = SegmentationDataset(
        split="train", augmentation={"enabled": False}, **kwargs
    )
    augmented = SegmentationDataset(
        split="train", augmentation={"enabled": True}, **kwargs
    )
    other_instance = SegmentationDataset(
        split="train", augmentation={"enabled": True}, **kwargs
    )
    previous = None
    for epoch in (0, 1):
        for dataset in (baseline, disabled, augmented, other_instance):
            dataset.set_epoch(epoch)
        plain, off, on, repeated = baseline[0], disabled[0], augmented[0], augmented[0]
        for key in plain:
            if isinstance(plain[key], torch.Tensor):
                torch.testing.assert_close(plain[key], off[key], rtol=0, atol=0)
                if key != "points":
                    torch.testing.assert_close(plain[key], on[key], rtol=0, atol=0)
        torch.testing.assert_close(on["points"], repeated["points"], rtol=0, atol=0)
        torch.testing.assert_close(
            on["points"], other_instance[0]["points"], rtol=0, atol=0
        )
        assert not torch.equal(plain["points"], on["points"])
        for identity in on["source_index"].unique():
            duplicates = on["points"][on["source_index"] == identity]
            assert torch.equal(duplicates, duplicates[0].expand_as(duplicates))
        if previous is not None:
            # First 40 valid points keep their source order, so this measures augmentation,
            # not merely a change to random sampled point order/padded repeats.
            assert not torch.equal(on["points"][:40], previous[:40])
        previous = on["points"]
        # Explicit pre-augmentation Run 1 sequence, independent of Dataset implementation.
        record = baseline.records[0]
        digest = hashlib.sha256(f"42:{epoch}:{record['sample_id']}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        pcd = PLYReader(tmp_path / record["path"]).load_point_cloud()
        pcd = decode_maizefield3d_labels(CloudFilter().remove_nan_inf(pcd))
        pcd = CloudNormalizer(scale_to_unit_sphere=True).process(pcd)
        pcd = FixedSizeSampler(64).sample(pcd, rng)
        np.testing.assert_array_equal(plain["points"].numpy(), pcd.xyz)
    validation = SegmentationDataset(
        split="val", augmentation={"enabled": True}, **kwargs
    )
    val = validation[0]
    validation.set_epoch(20)
    for key in ("points", "labels", "valid_mask", "source_index", "centroid", "scale"):
        torch.testing.assert_close(val[key], validation[0][key], rtol=0, atol=0)
    unaug_val = SegmentationDataset(split="val", **kwargs)[0]
    torch.testing.assert_close(val["points"], unaug_val["points"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="test"):
        SegmentationDataset(split="test", augmentation={"enabled": True}, **kwargs)


def test_gate_legacy_compatibility_and_tiny_rejection():
    plain = RunConfig()
    augmented = RunConfig(augmentation={"enabled": True})
    old_contract = experiment_contract(plain, "hash")
    new_contract = experiment_contract(augmented, "hash")
    assert old_contract["augmentation"] is None
    assert (
        experiment_contract(RunConfig(augmentation={"enabled": False}), "hash")
        == old_contract
    )
    assert {**new_contract, "augmentation": None} == old_contract
    gate = {
        "mode": "tiny_overfit",
        "overfit_status": {"passed": True},
        "contract": old_contract,
    }
    verify_overfit_gate(gate, new_contract)
    verify_checkpoint_contract({"contract": old_contract}, old_contract, "mismatch")
    with pytest.raises(ValueError, match="differs"):
        verify_overfit_gate({**gate, "contract": new_contract}, new_contract)
    with pytest.raises(ValueError, match="differs"):
        verify_overfit_gate(gate, {**new_contract, "points": 999})
    with pytest.raises(ValueError, match="Tiny-overfit"):
        RunConfig(overfit_samples=2, augmentation={"enabled": True}).validate()
    with pytest.raises(ValueError, match="SHA256"):
        verify_checkpoint_contract({"contract": new_contract}, new_contract, "mismatch")
    assert contract_sha256(new_contract) != contract_sha256(old_contract)


def test_run2_config_only_conceptual_difference():
    from pathlib import Path

    run1 = yaml.safe_load(Path("configs/pointnet2.yaml").read_text())
    run2 = yaml.safe_load(Path("configs/pointnet2_run2_aug.yaml").read_text())
    settings = run2.pop("augmentation")
    assert run2 == run1
    assert settings == asdict(AugmentationConfig(enabled=True))


def test_augmented_resume_matches_uninterrupted_and_validates_unaugmented(tmp_path):
    manifest, root = fixture_data(tmp_path)
    records, digest = read_manifest(manifest)
    cfg = config(2)
    cfg.overfit_samples = 0
    cfg.augmentation = {"enabled": True}
    # Structural synthetic gate fixture; this is not a produced overfit result.
    unaug_cfg = replace(cfg, augmentation={})
    gate = tmp_path / "gate.pt"
    save_checkpoint(
        gate,
        {
            "mode": "tiny_overfit",
            "overfit_status": {"passed": True},
            "contract": experiment_contract(unaug_cfg, digest),
            "train_sample_ids": ["0", "1"],
        },
    )
    a_dir, b_dir = tmp_path / "full", tmp_path / "resumed"
    train(manifest, root, str(a_dir), cfg, overfit_checkpoint=str(gate))
    train(
        manifest, root, str(b_dir), replace(cfg, epochs=1), overfit_checkpoint=str(gate)
    )
    train(manifest, root, str(b_dir), cfg, resume=str(b_dir / "last.pt"))
    a, b = load_checkpoint(a_dir / "last.pt"), load_checkpoint(b_dir / "last.pt")
    assert a["contract"]["augmentation"]["enabled"] is True
    assert a["contract_sha256"] == contract_sha256(a["contract"])
    for key in a["model_state"]:
        torch.testing.assert_close(
            a["model_state"][key], b["model_state"][key], rtol=0, atol=0
        )
    assert a["history"][-1]["train"]["loss"] == b["history"][-1]["train"]["loss"]
    validation = validate_checkpoint(
        manifest, root, str(a_dir / "last.pt"), str(tmp_path / "val.json"), "cpu"
    )
    assert validation["metrics"]["mIoU"] == a["history"][-1]["val"]["mIoU"]
    augmented_val = make_dataset(records, root, "val", cfg)[0]
    plain_val = make_dataset(records, root, "val", unaug_cfg)[0]
    torch.testing.assert_close(
        augmented_val["points"], plain_val["points"], rtol=0, atol=0
    )
    for augmentation in ({"enabled": False}, {"enabled": True, "jitter_sigma": 0.006}):
        with pytest.raises(ValueError, match="contract differs"):
            train(
                manifest,
                root,
                str(b_dir),
                replace(cfg, epochs=3, augmentation=augmentation),
                resume=str(b_dir / "last.pt"),
            )
    bad_hash = {**b, "contract_sha256": "bad"}
    save_checkpoint(b_dir / "last.pt", bad_hash)
    with pytest.raises(ValueError, match="SHA256"):
        train(
            manifest,
            root,
            str(b_dir),
            replace(cfg, epochs=3),
            resume=str(b_dir / "last.pt"),
        )
    analysis = analyze_checkpoint(
        manifest,
        root,
        str(a_dir / "last.pt"),
        str(tmp_path / "analysis"),
        device_name="cpu",
    )
    assert analysis["metrics"]["mIoU"] == validation["metrics"]["mIoU"]
    assert not (tmp_path / "4.ply").exists()


def test_legacy_run1_checkpoint_without_config_field_or_hash(tmp_path):
    manifest, root = fixture_data(tmp_path)
    _, digest = read_manifest(manifest)
    cfg = config(1)
    cfg.overfit_samples = 0
    gate = tmp_path / "gate.pt"
    save_checkpoint(
        gate,
        {
            "mode": "tiny_overfit",
            "overfit_status": {"passed": True},
            "contract": experiment_contract(cfg, digest),
            "train_sample_ids": ["0", "1"],
        },
    )
    output = tmp_path / "legacy"
    train(manifest, root, str(output), cfg, overfit_checkpoint=str(gate))
    legacy = load_checkpoint(output / "last.pt")
    legacy["config"].pop("augmentation")
    legacy.pop("contract_sha256")
    save_checkpoint(output / "last.pt", legacy)
    validate_checkpoint(
        manifest,
        root,
        str(output / "last.pt"),
        str(tmp_path / "legacy_val.json"),
        "cpu",
    )
    analyze_checkpoint(
        manifest,
        root,
        str(output / "last.pt"),
        str(tmp_path / "legacy_analysis"),
        device_name="cpu",
    )
    train(
        manifest,
        root,
        str(output),
        replace(cfg, epochs=2),
        resume=str(output / "last.pt"),
    )
    resumed = load_checkpoint(output / "last.pt")
    assert resumed["epoch"] == 2 and resumed["contract"]["augmentation"] is None
