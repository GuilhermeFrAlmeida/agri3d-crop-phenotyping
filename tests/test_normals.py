"""Geometry-only normal estimation and Run 4 contracts on synthetic clouds."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from plyfile import PlyData, PlyElement
from test_pointnet2_training import config, fixture_data

from agri3d.analysis.normal_audit import audit_normals
from agri3d.analysis.pointnet2 import analyze_checkpoint
from agri3d.models import PointNet2Config, PointNet2Segmentation
from agri3d.performance.runtime import RuntimeOptions
from agri3d.preprocessing.features import FeatureConfig
from agri3d.preprocessing.normals import (
    NormalConfig,
    canonical_orientation,
    estimate_normals,
)
from agri3d.training.checkpoints import load_checkpoint, save_checkpoint
from agri3d.training.pointnet2_runner import (
    RunConfig,
    build_model,
    contract_sha256,
    experiment_contract,
    make_dataset,
    read_manifest,
    train,
    validate_checkpoint,
    verify_checkpoint_contract,
    verify_overfit_gate,
)

FEATURES = {"mode": "xyz_normals", "normals": {"k": 32, "orientation": "deterministic"}}


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_plane_unit_orientation_and_repeatability():
    rng = np.random.default_rng(5)
    xyz = np.column_stack([rng.normal(size=(300, 2)), np.zeros(300)]).astype("float32")
    a, degenerate = estimate_normals(xyz)
    b, other = estimate_normals(xyz, chunk_size=17)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(degenerate, other)
    assert not degenerate.any()
    np.testing.assert_array_equal(a, np.tile([0, 0, 1], (len(xyz), 1)))
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1, rtol=0, atol=1e-6)
    # Mathematical sign, independent of arbitrary eigensolver +/-.
    vectors = np.array([[-1, 1, 0], [0, -2, 2], [0, 0, -1]], dtype=float)
    np.testing.assert_array_equal(
        canonical_orientation(vectors), canonical_orientation(-vectors)
    )
    np.testing.assert_array_equal(
        canonical_orientation(vectors), [[1, -1, 0], [0, 2, -2], [0, 0, 1]]
    )


@pytest.mark.parametrize(
    "xyz",
    [
        np.zeros((1, 3)),
        np.zeros((50, 3)),
        np.column_stack([np.arange(40), np.zeros((40, 2))]),
        np.array([[0, 0, 0], [1, 0, 0]]),
    ],
)
def test_degenerate_clouds_explicit_fallback(xyz):
    normals, degenerate = estimate_normals(xyz)
    assert degenerate.all()
    np.testing.assert_array_equal(normals, np.tile([0, 0, 1], (len(xyz), 1)))


def test_ties_and_isotropic_neighborhoods_repeat():
    xyz = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [0, 0, 0],
        ],
        dtype="float32",
    )
    a, bad = estimate_normals(xyz, {"k": 7})
    assert bad.all()
    for chunk in (1, 2, 100):
        b, _ = estimate_normals(xyz, {"k": 7}, chunk)
        np.testing.assert_array_equal(a, b)
    # Boundary ties in a lattice exercise radius expansion and stable index order.
    grid = np.array(
        [[x, y, z] for x in range(4) for y in range(4) for z in range(2)],
        dtype="float32",
    )
    a, _ = estimate_normals(grid, {"k": 8})
    b, _ = estimate_normals(grid, {"k": 8}, chunk_size=1)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "settings",
    [
        {"k": 2},
        {"k": True},
        {"orientation": "random"},
        {"source": "ply"},
        {"method": "rgb"},
        {"policy_version": 2},
    ],
)
def test_invalid_policies(settings):
    with pytest.raises((ValueError, TypeError)):
        NormalConfig.from_settings(settings)
    with pytest.raises(ValueError):
        estimate_normals(np.array([[np.nan, 0, 0]]))


def test_dataset_full_resolution_before_normalize_sampling_alignment_no_rgb(
    tmp_path, monkeypatch
):
    manifest, root = fixture_data(tmp_path)
    records, _ = read_manifest(manifest)
    # Deliberate nonfinite source points; original source indices retain gaps.
    data = PlyData.read(tmp_path / "0.ply")["vertex"].data.copy()
    data["x"][3] = np.nan
    data["z"][7] = np.inf
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(
        str(tmp_path / "0.ply")
    )
    import agri3d.training.dataset as module

    real = module.estimate_normals
    inputs = []

    def spy(xyz, *args, **kwargs):
        inputs.append(xyz.copy())
        return real(xyz, *args, **kwargs)

    monkeypatch.setattr(module, "estimate_normals", spy)
    cfg = replace(config(), points=64, features=FEATURES)
    normals = make_dataset(records, root, "train", cfg)
    plain = make_dataset(records, root, "train", replace(cfg, features={}))
    raw = normals._normalized_cloud(tmp_path / "0.ply")
    sample = normals[0]
    baseline = plain[0]
    assert inputs[0].shape == (38, 3)
    expected = np.column_stack([data[k] for k in ("x", "y", "z")])
    expected = expected[np.isfinite(expected).all(1)]
    np.testing.assert_array_equal(inputs[0], expected)
    before_normalization, _ = real(expected)
    np.testing.assert_array_equal(raw.attributes["normals"], before_normalization)
    for key in baseline:
        if isinstance(baseline[key], torch.Tensor):
            torch.testing.assert_close(sample[key], baseline[key], rtol=0, atol=0)
        else:
            assert sample[key] == baseline[key]
    positions = np.searchsorted(
        raw.attributes["source_index"], sample["source_index"].numpy()
    )
    np.testing.assert_array_equal(
        sample["features"], raw.attributes["normals"][positions]
    )
    assert sample["valid_mask"].sum() == 38
    # RGB controls supervision, but changing it cannot change the normal features.
    data["red"][:] = 255
    data["green"][:] = 0
    data["blue"][:] = 0
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(
        str(tmp_path / "0.ply")
    )
    changed = normals[0]
    torch.testing.assert_close(sample["features"], changed["features"], rtol=0, atol=0)
    assert not torch.equal(sample["labels"], changed["labels"])
    normals.set_epoch(7)
    again = normals[0]
    positions = np.searchsorted(
        raw.attributes["source_index"], again["source_index"].numpy()
    )
    np.testing.assert_array_equal(
        again["features"], raw.attributes["normals"][positions]
    )
    with pytest.raises(ValueError, match="train/val"):
        make_dataset(records, root, "test", cfg)


def test_cache_separates_policies_and_matches_exactly(tmp_path):
    manifest, root = fixture_data(tmp_path)
    records, _ = read_manifest(manifest)
    runtime = RuntimeOptions(cache_dir=str(tmp_path / "cache"))
    cfg = replace(config(), features=FEATURES)
    normal = make_dataset(records, root, "train", cfg, runtime)
    uncached = make_dataset(records, root, "train", cfg)
    plain = make_dataset(records, root, "train", replace(cfg, features={}), runtime)
    plain[0]
    for epoch in (0, 3):
        normal.set_epoch(epoch)
        uncached.set_epoch(epoch)
        a, b = normal[0], uncached[0]
        for key in a:
            if isinstance(a[key], torch.Tensor):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        normal[0]
        assert normal.cache.last_hit
    entries = list((tmp_path / "cache").glob("*.npz"))
    assert len(entries) == 2
    identities = []
    for entry in entries:
        with np.load(entry, allow_pickle=False) as f:
            identities.append(json.loads(str(f["identity"])))
    detailed = next(i for i in identities if i["features"] != "xyz")
    assert detailed["feature_provenance"]["policy"]["k"] == 32
    assert detailed["feature_provenance"]["implementation_sha256"]
    assert detailed["feature_provenance"]["scipy_version"]
    different = make_dataset(
        records,
        root,
        "train",
        replace(cfg, features={"mode": "xyz_normals", "normals": {"k": 16}}),
        runtime,
    )
    different[0]
    assert not different.cache.last_hit
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 3
    old = normal.cache.identity(tmp_path / "0.ply")
    normal.cache.feature_provenance = {
        **normal.cache.feature_provenance,
        "implementation_sha256": "changed",
    }
    assert normal.cache.identity(tmp_path / "0.ply") != old
    for split in ("train", "val"):
        a = make_dataset(records, root, split, cfg, runtime)
        b = make_dataset(records, root, split, cfg)
        for key in ("points", "features", "source_index", "valid_mask", "labels"):
            torch.testing.assert_close(a[0][key], b[0][key], rtol=0, atol=0)


def legacy_forward(model, xyz):
    coords, features = [xyz], [None]
    for stage in model.abstraction:
        a, b = stage(coords[-1], features[-1])
        coords.append(a)
        features.append(b)
    propagated = features[-1]
    for i, stage in enumerate(model.propagation):
        level = 3 - i
        propagated = stage(
            coords[level],
            coords[level + 1],
            xyz if level == 0 else features[level],
            propagated,
        )
    return model.classifier(propagated.transpose(1, 2)).transpose(1, 2).contiguous()


def test_model_legacy_state_and_outputs_and_parameter_change():
    cfg = config()
    plain = build_model(cfg).eval()
    copy_model = PointNet2Segmentation(
        PointNet2Config(**cfg.model), feature_channels=0
    ).eval()
    copy_model.load_state_dict(plain.state_dict(), strict=True)
    xyz = torch.randn(2, 24, 3)
    torch.testing.assert_close(
        plain(xyz), legacy_forward(copy_model, xyz), atol=0, rtol=0
    )
    with pytest.raises(ValueError, match="XYZ-only"):
        plain(xyz, torch.randn_like(xyz))
    normal = build_model(replace(cfg, features=FEATURES)).eval()
    features = torch.nn.functional.normalize(torch.randn_like(xyz), dim=-1)
    assert normal(xyz, features).shape == (2, 24, 2)
    assert (
        sum(p.numel() for p in normal.parameters())
        - sum(p.numel() for p in plain.parameters())
        == 480
    )
    with pytest.raises(ValueError, match="align"):
        normal(xyz)
    with pytest.raises(ValueError, match="align"):
        normal(xyz, features[:, :-1])
    with pytest.raises(ValueError, match="finite"):
        normal(xyz, torch.full_like(features, float("nan")))
    with pytest.raises(RuntimeError):
        normal.load_state_dict(plain.state_dict(), strict=True)


def test_geometry_queries_ignore_normal_features(monkeypatch):
    from agri3d.models import pointnet2_ops as ops

    seen = []
    original = ops.farthest_point_sample

    def fps(xyz, *a, **k):
        assert xyz.shape[-1] == 3
        seen.append(xyz.detach().clone())
        return original(xyz, *a, **k)

    monkeypatch.setattr(ops, "farthest_point_sample", fps)
    from agri3d.models import pointnet2 as network

    radius_calls, interpolation_calls = [], []
    original_radius, original_interpolate = (
        ops.query_ball_point,
        network.interpolate_features,
    )

    def radius(r, k, xyz, centers, *args, **kwargs):
        assert xyz.shape[-1] == centers.shape[-1] == 3
        radius_calls.append((xyz.detach().clone(), centers.detach().clone()))
        return original_radius(r, k, xyz, centers, *args, **kwargs)

    def interpolate(dense, sparse, *args, **kwargs):
        assert dense.shape[-1] == sparse.shape[-1] == 3
        interpolation_calls.append((dense.detach().clone(), sparse.detach().clone()))
        return original_interpolate(dense, sparse, *args, **kwargs)

    monkeypatch.setattr(ops, "query_ball_point", radius)
    monkeypatch.setattr(network, "interpolate_features", interpolate)
    cfg = replace(config(), features=FEATURES)
    model = build_model(cfg).eval()
    xyz = torch.randn(2, 24, 3)
    features = torch.nn.functional.normalize(torch.randn_like(xyz), dim=-1)
    model(xyz, features)
    model(xyz, -features)
    for a, b in zip(seen[:4], seen[4:]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(seen[0], xyz, atol=0, rtol=0)
    for calls in (radius_calls, interpolation_calls):
        for a, b in zip(calls[:4], calls[4:]):
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_run4_config_and_gate_and_hash_contract():
    base = yaml.safe_load(Path("configs/pointnet2.yaml").read_text())
    run4 = yaml.safe_load(Path("configs/pointnet2_run4_normals.yaml").read_text())
    features = run4.pop("features")
    assert base == run4
    cfg = RunConfig(**{**run4, "features": features})
    contract = experiment_contract(cfg, "hash")
    assert (
        contract["features"]["normal_estimation"]["orientation"]
        == "largest_abs_component_positive"
    )
    assert experiment_contract(RunConfig(), "hash")["features"] == "xyz"
    gate = {
        "mode": "tiny_overfit",
        "overfit_status": {"passed": True},
        "contract": contract,
        "contract_sha256": contract_sha256(contract),
    }
    verify_overfit_gate(gate, contract)
    with pytest.raises(ValueError, match="differs"):
        verify_overfit_gate(
            {**gate, "contract": experiment_contract(RunConfig(), "hash")}, contract
        )
    with pytest.raises(ValueError, match="SHA256"):
        verify_checkpoint_contract({"contract": contract}, contract, "mismatch")
    changed = experiment_contract(
        replace(cfg, features={"mode": "xyz_normals", "normals": {"k": 16}}), "hash"
    )
    with pytest.raises(ValueError, match="differs"):
        verify_overfit_gate(gate, changed)
    with pytest.raises(ValueError):
        replace(cfg, augmentation={"enabled": True}).validate()
    with pytest.raises(ValueError):
        replace(cfg, scheduler={"enabled": True}).validate()
    with pytest.raises(ValueError, match="RGB"):
        FeatureConfig.from_settings({"mode": "rgb"})


def test_normal_audit_and_training_resume_validation_analysis(tmp_path):
    manifest, root = fixture_data(tmp_path)
    cfg = replace(config(2), features=FEATURES)
    report = audit_normals(
        manifest,
        root,
        tmp_path / "audit.json",
        cfg,
        max_plants=2,
        runtime=RuntimeOptions(cache_dir=str(tmp_path / "cache")),
    )
    assert report["passed"] and report["plants_checked"] == 2
    assert (
        report["finite_normal_ratio"] == 1
        and report["norm_distribution"]["min"] > 0.99999
    )
    assert all(
        row["deterministic"] and row["source_alignment_verified"]
        for row in report["plants"]
    )
    with pytest.raises(ValueError, match="test"):
        audit_normals("missing", "missing", "missing", cfg, split="test")
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(manifest, root, str(full), cfg)
    train(manifest, root, str(resumed), replace(cfg, epochs=1))
    train(
        manifest,
        root,
        str(resumed),
        cfg,
        resume=str(resumed / "last.pt"),
        runtime=RuntimeOptions(
            cache_dir=str(tmp_path / "cache"), pin_memory=True, non_blocking=True
        ),
    )
    a, b = load_checkpoint(full / "last.pt"), load_checkpoint(resumed / "last.pt")
    for key in a["model_state"]:
        torch.testing.assert_close(
            a["model_state"][key], b["model_state"][key], atol=0, rtol=0
        )
    assert (
        a["history"][-1]["train_eval"]["loss"] == b["history"][-1]["train_eval"]["loss"]
    )
    # Structural passing-gate fixture, not a scientific overfit success claim.
    gate = tmp_path / "gate.pt"
    passing = copy.deepcopy(a)
    passing["overfit_status"]["passed"] = True
    save_checkpoint(gate, passing)
    regular = replace(cfg, epochs=1, overfit_samples=0)
    out = tmp_path / "regular"
    train(manifest, root, str(out), regular, overfit_checkpoint=str(gate))
    val = validate_checkpoint(
        manifest, root, str(out / "best.pt"), str(tmp_path / "val.json"), "cpu"
    )
    analysis = analyze_checkpoint(
        manifest,
        root,
        str(out / "best.pt"),
        str(tmp_path / "analysis"),
        device_name="cpu",
    )
    assert val["metrics"]["mIoU"] == analysis["metrics"]["mIoU"]
    assert (
        val["metrics"]["mIoU"]
        == load_checkpoint(out / "best.pt")["history"][-1]["val"]["mIoU"]
    )
    with pytest.raises(ValueError, match="contract differs"):
        train(
            manifest,
            root,
            str(out),
            replace(regular, epochs=2, features={}),
            resume=str(out / "last.pt"),
        )
    with pytest.raises(ValueError, match="contract differs"):
        train(
            manifest,
            root,
            str(out),
            replace(
                regular,
                epochs=2,
                features={"mode": "xyz_normals", "normals": {"k": 16}},
            ),
            resume=str(out / "last.pt"),
        )
    assert not (tmp_path / "4.ply").exists()


@pytest.mark.parametrize(
    "policy",
    [{}, {"augmentation": {"enabled": True}}, {"scheduler": {"enabled": True}}],
)
def test_actual_legacy_config_without_feature_field_validates_and_analyzes(
    tmp_path, policy
):
    from test_pointnet2_analysis import fixture_checkpoint

    manifest, path, _, cfg = fixture_checkpoint(tmp_path)
    cfg = replace(cfg, **policy)
    saved = load_checkpoint(path)
    saved["config"].update(policy)
    saved["config"].pop("features")
    saved["contract"] = experiment_contract(cfg, read_manifest(manifest)[1])
    saved["contract_sha256"] = contract_sha256(saved["contract"])
    assert saved["contract"]["features"] == "xyz"
    save_checkpoint(path, saved)
    a = validate_checkpoint(
        str(manifest), str(tmp_path), str(path), str(tmp_path / "val.json"), "cpu"
    )
    b = analyze_checkpoint(
        str(manifest),
        str(tmp_path),
        str(path),
        str(tmp_path / "analysis"),
        device_name="cpu",
    )
    assert a["metrics"]["mIoU"] == b["metrics"]["mIoU"]
    assert not (tmp_path / "plant4.ply").exists()


def test_eigensolver_sign_is_never_used_as_orientation(monkeypatch):
    xyz = np.random.default_rng(8).normal(size=(70, 3)).astype("float32")
    expected, _ = estimate_normals(xyz)
    original = np.linalg.eigh

    def inverted(*args, **kwargs):
        values, vectors = original(*args, **kwargs)
        return values, -vectors

    monkeypatch.setattr(np.linalg, "eigh", inverted)
    actual, _ = estimate_normals(xyz)
    np.testing.assert_array_equal(expected, actual)


def test_knn_pca_matches_brute_force_tie_policy():
    xyz = np.array(
        [[x, y, z] for x in range(4) for y in range(4) for z in range(2)],
        dtype="float64",
    )
    actual, bad = estimate_normals(xyz, {"k": 8})
    for i in range(len(xyz)):
        distances = ((xyz - xyz[i]) ** 2).sum(1)
        candidates = np.arange(len(xyz))
        candidates = candidates[candidates != i]
        order = np.lexsort((candidates, distances[candidates]))
        indices = np.r_[i, candidates[order[:7]]]
        local = xyz[indices] - xyz[indices].mean(0)
        values, vectors = np.linalg.eigh(local.T @ local / 8)
        if values[2] <= 0 or values[1] - values[0] <= 1e-8 * values[2]:
            assert bad[i]
        else:
            expected = canonical_orientation(vectors[:, 0][None])[0]
            np.testing.assert_allclose(actual[i], expected, atol=1e-6, rtol=0)
