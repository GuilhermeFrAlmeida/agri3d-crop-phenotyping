"""Synthetic downstream qualification. No real dataset/checkpoint or test PLY is opened."""
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from agri3d.inference.export import (
    ExportNotQualified,
    export_pointnet2,
    load_qualification,
)
from agri3d.inference.pointnet2_backend import PyTorchPointNet2Predictor
from agri3d.inference.projection import nearest_indices, project_nearest
from agri3d.phenotyping.instances import InstanceExtractor
from agri3d.phenotyping.measurements import (
    PhenotypeExtractor,
    canopy_volume,
    leaf_lengths,
)
from agri3d.phenotyping.references import (
    decode_verified_instances,
    instance_metrics,
    load_trait_references,
)
from agri3d.pipeline.analyze import analyze_plant
from agri3d.pipeline.artifacts import write_plant
from agri3d.pipeline.batch import evaluate_pipeline
from agri3d.pipeline.config import InstanceConfig, PipelineConfig, TraitConfig
from agri3d.pipeline.provenance import sha256_file
from agri3d.training.checkpoints import load_checkpoint, save_checkpoint
from agri3d.training.pointnet2_runner import (
    RunConfig,
    build_model,
    contract_sha256,
    experiment_contract,
    make_dataset,
)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def write_ply(path, xyz, colors=None):
    dtype = [(a, "f4") for a in "xyz"] + [(c, "u1") for c in ("red", "green", "blue")]
    vertices = np.empty(len(xyz), dtype=dtype)
    for i, name in enumerate("xyz"):
        vertices[name] = xyz[:, i]
    if colors is None:
        colors = np.where((xyz[:, 0] > .05)[:, None], [[200, 20, 10]], [[0, 0, 0]])
    for i, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, i]
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def fixture(tmp, full=False):
    xyz = np.random.default_rng(42).uniform(-.2, .2, (80, 3)).astype("float32")
    xyz[:, 2] += 1
    write_ply(tmp / "val.ply", xyz)
    records = [dict(path="val.ply", sample_id="plant", group_id="g", split="val"),
               dict(path="MISSING_TEST.ply", sample_id="test", group_id="heldout", split="test")]
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps(records))
    mh = sha256_file(manifest)
    cfg = RunConfig() if full else RunConfig(points=32, model={"npoints": [16, 8, 4, 2],
        "nsamples": [4, 4, 4, 4], "radii": [.1, .2, .4, .8], "chunk_size": 8})
    torch.manual_seed(42)
    contract = experiment_contract(cfg, mh)
    checkpoint = tmp / "model.pt"
    save_checkpoint(checkpoint, {"mode": "regular", "config": asdict(cfg), "contract": contract,
        "contract_sha256": contract_sha256(contract), "model_state": build_model(cfg).state_dict(),
        "selection_metric": "val_mIoU", "epoch": 76, "best_miou": .83,
        "history": [{"epoch": 76, "val": {"mIoU": .83}}]})
    return manifest, checkpoint, records, cfg


def test_backend_exact_inference_metadata_and_source_unchanged(tmp_path):
    _, checkpoint, _, cfg = fixture(tmp_path)
    before = sha256_file(checkpoint)
    p = PyTorchPointNet2Predictor(checkpoint, require_frozen_run1=False, supplied_metadata={"note": "explicit"})
    xyz = np.random.default_rng(1).normal(size=(cfg.points, 3)).astype("float32")
    a, b = p.predict(xyz), p.predict(xyz)
    np.testing.assert_array_equal(a.logits, b.logits)
    with torch.no_grad():
        direct = p.model(torch.from_numpy(xyz)[None])[0].numpy()
    np.testing.assert_array_equal(a.logits, direct)
    assert p.metadata["validation_mIoU"] == .83 and p.metadata["checkpoint_sha256"] == before
    assert p.metadata["user_supplied_metadata"] == {"note": "explicit"}
    assert sha256_file(checkpoint) == before
    with pytest.raises(ValueError, match="frozen Run 1"):
        PyTorchPointNet2Predictor(checkpoint)
    for x in [xyz[:1], xyz.astype(float), np.full_like(xyz, np.nan)]:
        with pytest.raises(ValueError):
            p.predict(x)
    with pytest.raises(ValueError):
        p.predict(xyz, xyz)


@pytest.mark.parametrize("change", ["contract", "hash", "mode", "features", "augmentation"])
def test_checkpoint_rejections(tmp_path, change):
    _, path, _, _ = fixture(tmp_path)
    data = load_checkpoint(path)
    if change == "contract":
        data["contract"]["seed"] = 17
    elif change == "hash":
        data["contract_sha256"] = "bad"
    elif change == "mode":
        data["mode"] = "tiny_overfit"
    else:
        data["config"][change] = ({"mode": "xyz_normals"} if change == "features" else {"enabled": True})
    save_checkpoint(path, data)
    with pytest.raises(ValueError):
        PyTorchPointNet2Predictor(path, require_frozen_run1=False)


def test_projection_exact_source_ties_determinism_and_metric_invariance():
    full = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [.5, 0, 0]], dtype="float32")
    ids = np.array([0, 1, 2])
    labels = np.array([0, 1, 1])
    p = np.eye(2)[labels]
    a, probabilities = project_nearest(full, full[ids], labels, ids, p)
    np.testing.assert_array_equal(a[ids], labels)
    np.testing.assert_array_equal(probabilities[ids], p)
    np.testing.assert_array_equal(a, project_nearest(full, full[ids], labels, ids)[0])
    np.testing.assert_array_equal(a, project_nearest(full * 2 + 4, (full * 2 + 4)[ids], labels, ids)[0])
    assert a[-1] == 0  # equal-distance tie: sampled slot zero
    np.testing.assert_array_equal(nearest_indices(full[:1], full), np.zeros(4))


@pytest.mark.parametrize("case", ["empty", "nan", "labels", "indices", "alignment", "duplicate", "probabilities"])
def test_projection_validation(case):
    xyz = np.arange(12, dtype=float).reshape(4, 3)
    ids, labels, sample, prob = np.array([0, 2]), np.array([0, 1]), xyz[[0, 2]].copy(), None
    if case == "empty": xyz = xyz[:0]
    if case == "nan": xyz[0] = np.nan
    if case == "labels": labels[0] = 8
    if case == "indices": ids[0] = -1
    if case == "alignment": sample[0] += 1
    if case == "duplicate": ids[1] = 0
    if case == "probabilities": prob = np.zeros((2, 2))
    with pytest.raises(ValueError):
        project_nearest(xyz, sample, labels, ids, prob)


def test_instances_disconnected_touching_noise_and_full_accounting():
    line = np.column_stack([np.linspace(0, .1, 21), np.zeros((21, 2))])
    xyz = np.concatenate([line, line + [0, .2, 0], [[2, 2, 2]], [[0, 0, -.1]]])
    semantic = np.zeros(len(xyz), dtype=int)
    semantic[-1] = 1
    extractor = InstanceExtractor(InstanceConfig(neighborhood_radius_m=.012, min_leaf_points=5))
    a, b = extractor.extract(xyz, semantic), extractor.extract(xyz, semantic)
    assert len(a.leaves) == 2 and a.metadata["rejected_small_components"] == 1
    assert len(a.unassigned_indices) == 1 and len(a.stalk_indices) == 1
    np.testing.assert_array_equal(a.instance_labels(len(xyz)), b.instance_labels(len(xyz)))
    bridge = np.column_stack([np.zeros(19), np.linspace(.01, .19, 19), np.zeros(19)])
    touched = extractor.extract(np.vstack([xyz[:-2], bridge]), np.zeros(61, dtype=int))
    assert len(touched.leaves) == 1  # documented baseline failure mode, not hidden split logic
    empty_leaf = extractor.extract(xyz, np.ones(len(xyz), dtype=int))
    assert not empty_leaf.leaves and (empty_leaf.instance_labels(len(xyz)) == 0).all()


def test_measurements_units_known_geometry_degeneracy_and_lengths():
    cube = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)])
    volume = canopy_volume(cube, .5)
    assert volume["convex_hull_m3"] == pytest.approx(1)
    assert volume["voxel_occupancy_m3"] == pytest.approx(1)
    line = np.column_stack([np.linspace(0, 2, 101), np.zeros((101, 2))])
    cfg = TraitConfig(geodesic_voxel_size_m=.001, geodesic_radius_m=.025)
    lengths = leaf_lengths(line, cfg)
    assert lengths["length_pca_m"] == pytest.approx(2)
    assert lengths["length_geodesic_m"] == pytest.approx(2)
    degenerate = leaf_lengths(np.zeros((3, 3)), cfg)
    assert degenerate["length_pca_m"] is None and degenerate["length_geodesic_m"] is None
    assert canopy_volume(line, .1)["convex_hull_m3"] is None
    assert leaf_lengths(line, TraitConfig(geodesic_max_nodes=2))["length_geodesic_m"] is None
    structure = InstanceExtractor(InstanceConfig(min_leaf_points=1)).extract(cube, np.zeros(8, dtype=int))
    traits = PhenotypeExtractor().extract(cube * 3 + [0, 0, 4], structure).to_dict()
    assert traits["phenotypes"]["height_m"] == 3 and traits["phenotypes"]["z_min"] == 4
    assert traits["phenotypes"]["point_count_used"] == 8
    assert traits["phenotypes"]["canopy_volume"]["convex_hull_m3"] == pytest.approx(27)
    with pytest.raises(ValueError): canopy_volume(cube, 0)


def test_explicit_verified_instance_and_trait_reference_adapters(tmp_path):
    fixture(tmp_path)
    from agri3d.io.ply import PLYReader
    cloud = PLYReader(tmp_path / "val.ply").load_point_cloud()
    policy = {"schema_version": 1, "evidence": "reviewed fixture mapping",
              "rgb_to_instance": {"0,0,0": 0, "200,20,10": 1}}
    path = tmp_path / "palette.json"
    path.write_text(json.dumps(policy))
    labels, _ = decode_verified_instances(cloud, path)
    scores = instance_metrics(labels, labels)
    assert scores["matched_instance_iou"] == 1 and scores["absolute_leaf_count_error"] == 0
    policy["rgb_to_instance"].pop("200,20,10")
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="Unmapped"):
        decode_verified_instances(cloud, path)
    csv = tmp_path / "traits.csv"
    csv.write_text("sample_id,height_m,leaf_count\nplant,2.3,4\n")
    assert load_trait_references(csv, "verified meters")["plant"]["leaf_count"] == 4
    with pytest.raises(ValueError): load_trait_references(csv, None)


def test_pipeline_sampling_parity_filtering_alignment_no_rgb_leak(tmp_path):
    manifest, path, records, cfg = fixture(tmp_path)
    p = PyTorchPointNet2Predictor(path, require_frozen_run1=False)
    a = analyze_plant(tmp_path / "val.ply", p, sample_id="plant", evaluate_semantics=True)
    sample = make_dataset(records, tmp_path, "val", cfg)[0]
    np.testing.assert_array_equal(sample["points"].numpy(), a.sampled.xyz)
    np.testing.assert_array_equal(sample["source_index"].numpy(), a.sampled.attributes["source_index"])
    ids = a.sampled.attributes["source_index"]
    np.testing.assert_array_equal(a.semantic_labels[ids], a.prediction.labels)
    assert a.report["phenotypes"]["height_m"] == pytest.approx(np.ptp(a.cloud.xyz[:, 2]))
    source = tmp_path / "val.ply"
    xyz = a.cloud.xyz.copy()
    write_ply(source, xyz, np.zeros((len(xyz), 3), dtype=int))
    b = analyze_plant(source, p, sample_id="plant", evaluate_semantics=True)
    np.testing.assert_array_equal(a.prediction.logits, b.prediction.logits)
    xyz[3] = np.nan
    write_ply(source, xyz)
    c = analyze_plant(source, p, sample_id="plant")
    assert c.report["clean_point_count"] == 79 and 3 not in c.cloud.attributes["source_index"]
    assert 3 not in c.sampled.attributes["source_index"]
    assert sha256_file(manifest) == p.metadata["manifest_sha256"]


def test_pipeline_artifacts_batch_oracle_predicted_and_no_test_access(tmp_path):
    manifest, path, _, _ = fixture(tmp_path)
    p = PyTorchPointNet2Predictor(path, require_frozen_run1=False)
    result = analyze_plant(tmp_path / "val.ply", p, sample_id="plant", evaluate_semantics=True)
    input_hash = sha256_file(tmp_path / "val.ply")
    out = tmp_path / "single"
    write_plant(out, result, visualize=True)
    inventory = json.loads((out / "artifacts.json").read_text())["files"]
    for name, info in inventory.items():
        assert sha256_file(out / name) == info["sha256"]
        assert (out / name).stat().st_size == info["size_bytes"]
    ply = PlyData.read(out / "structure.ply")["vertex"]
    assert {"semantic_label", "instance_id", "source_index"} <= set(ply.data.dtype.names)
    np.testing.assert_array_equal(ply["semantic_label"], result.semantic_labels)
    assert sha256_file(tmp_path / "val.ply") == input_hash
    with pytest.raises(ValueError): write_plant(out, result)
    a = evaluate_pipeline(manifest, tmp_path, p, tmp_path / "batch", max_plants=1, visualize=True)
    assert a["successful_plant_modes"] == 2 and not a["failures"]
    assert set(a["by_semantic_source"]) == {"ground_truth", "run1_projected_prediction"}
    assert "sampled_10k" in a["semantic_metrics"] and "projected_full_cloud" in a["semantic_metrics"]
    b = evaluate_pipeline(manifest, tmp_path, p, tmp_path / "batch_repeat", max_plants=1)
    assert a["semantic_metrics"] == b["semantic_metrics"]
    with pytest.raises(ValueError, match="validation only"):
        evaluate_pipeline("absent", "absent", None, "absent", split="test")
    raw = json.loads(manifest.read_text())
    raw.append(dict(path="absent.ply", sample_id="absent", group_id="missing", split="val"))
    manifest.write_text(json.dumps(raw))
    p.metadata["manifest_sha256"] = sha256_file(manifest)
    failed = evaluate_pipeline(manifest, tmp_path, p, tmp_path / "failures", mode="predicted")
    assert failed["failures"] == 1 and failed["successful_plant_modes"] == 1
    with pytest.raises(FileNotFoundError):
        evaluate_pipeline(manifest, tmp_path, p, tmp_path / "failfast", fail_fast=True)
    manifest.write_text(" ")
    with pytest.raises(ValueError): evaluate_pipeline(manifest, tmp_path, p, tmp_path / "invalid")


def test_official_cli_single_and_test_rejection(tmp_path):
    _, path, _, _ = fixture(tmp_path, full=True)
    command = [sys.executable, "-m", "agri3d.pipeline", "analyze", "--input", str(tmp_path / "val.ply"),
               "--checkpoint", str(path), "--output", str(tmp_path / "cli")]
    # CPU fixed 10k forward, random synthetic weights, no training or biological claim.
    import os
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run(command, check=True, capture_output=True, env=env, timeout=120)
    assert (tmp_path / "cli" / "phenotype.json").exists()
    for script in ([sys.executable, "scripts/evaluate_phenotyping_pipeline.py"],
                   [sys.executable, "-m", "agri3d.pipeline", "analyze-batch"]):
        r = subprocess.run(script + ["--manifest", "missing", "--root", "missing", "--checkpoint", "missing",
                                     "--output-dir", "missing", "--split", "test"], capture_output=True, text=True)
        assert r.returncode == 2 and "invalid choice" in r.stderr


@pytest.mark.parametrize("data", [{"projection": "vote"}, {"vertical_axis": "y"}, {"seed": 7}])
def test_config_rejections(data):
    with pytest.raises(ValueError): PipelineConfig(**data)


def test_backend_neutral_boundary():
    for folder in ("pipeline", "phenotyping"):
        for path in (Path("src/agri3d") / folder).glob("*.py"):
            if path.name == "__main__.py": continue  # sole CLI composition root
            source = path.read_text()
            assert "PointNet2Segmentation" not in source and "pointnet2_runner" not in source


def test_export_failure_is_explicit_and_backend_rejects(tmp_path, monkeypatch):
    _, path, _, _ = fixture(tmp_path)
    p = PyTorchPointNet2Predictor(path, require_frozen_run1=False)
    def blocked(*_a, **_k):
        raise RuntimeError("data-dependent torch.isfinite tensor boolean guard")
    monkeypatch.setattr(torch.onnx, "export", blocked)
    report = export_pointnet2(p, tmp_path / "export")
    assert report["status"] == "NOT QUALIFIED" and "isfinite" in report["blocker"]
    with pytest.raises(ExportNotQualified):
        load_qualification(path, tmp_path / "export" / "qualification.json")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_optional_backend(tmp_path):
    _, path, _, cfg = fixture(tmp_path)
    p = PyTorchPointNet2Predictor(path, "cuda", require_frozen_run1=False)
    a = p.predict(np.zeros((cfg.points, 3), dtype="float32"))
    assert a.logits.shape == (cfg.points, 2)


def test_onnx_checker_runtime_qualifier_parity_and_mismatch(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    from agri3d.inference.export import qualify_model
    from agri3d.inference.onnx_backend import OnnxSemanticPredictor
    from agri3d.inference.predictor import SemanticPrediction
    from agri3d.pipeline.artifacts import write_json

    weights = np.array([[1, -1], [.3, -.3], [0, 0]], dtype="float32")
    class LinearFixture:
        input_points = 16
        metadata = {"fixture": "synthetic linear operator, not PointNet++"}
        def predict(self, xyz):
            logits = xyz @ weights
            exp = np.exp(logits - logits.max(1, keepdims=True))
            return SemanticPrediction(logits.argmax(1), logits, exp / exp.sum(1, keepdims=True),
                                      np.arange(len(xyz)), self.metadata)
    def write_model(path, w):
        graph = helper.make_graph([helper.make_node("MatMul", ["xyz", "weights"], ["logits"])],
            "fixture", [helper.make_tensor_value_info("xyz", TensorProto.FLOAT, [1, 16, 3])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 16, 2])],
            [numpy_helper.from_array(w, "weights")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 10
        onnx.save(model, path)
    model = tmp_path / "fixture.onnx"
    write_model(model, weights)
    report = qualify_model(LinearFixture(), model)
    assert report["status"] == "QUALIFIED" and len(report["cases"]) == 3
    assert all(x["label_agreement"] == 1 for x in report["cases"])
    report_file = tmp_path / "qualification.json"
    write_json(report_file, report)
    runtime = OnnxSemanticPredictor(model, report_file)
    runtime.predict(np.ones((16, 3), dtype="float32")).validate(16)
    write_model(model, -weights)
    with pytest.raises(ExportNotQualified): load_qualification(model, report_file)
    failed = qualify_model(LinearFixture(), model)
    assert failed["status"] == "NOT QUALIFIED"


def test_semantic_prediction_invalid_contracts():
    from agri3d.inference.predictor import SemanticPrediction
    good = SemanticPrediction(np.array([0, 1]), np.eye(2), np.eye(2), np.arange(2), {})
    good.validate(2)
    import copy
    for field, value in [("probabilities", np.zeros((2, 2))), ("labels", np.array([1, 0])),
                         ("source_indices", np.array([1, 0])), ("logits", np.full((2, 2), np.nan))]:
        bad = copy.deepcopy(good)
        setattr(bad, field, value)
        with pytest.raises(ValueError): bad.validate(2)


def test_trait_reference_evaluation_and_oracle_without_model(tmp_path):
    manifest, _, _, _ = fixture(tmp_path)
    refs = tmp_path / "references.csv"
    refs.write_text("sample_id,height_m,leaf_count\nplant,2.0,2\n")
    result = evaluate_pipeline(manifest, tmp_path, None, tmp_path / "oracle", mode="oracle",
                              reference_traits=refs, reference_evidence="reviewed synthetic meters")
    oracle = result["by_semantic_source"]["ground_truth"]
    assert oracle["trait_reference_errors"]["height_m_error"]["count"] == 1
    assert "semantic_metrics" not in result


def test_disconnected_geodesic_and_empty_inputs():
    disconnected = np.array([[0, 0, 0], [3, 0, 0]], dtype=float)
    measured = leaf_lengths(disconnected, TraitConfig())
    assert measured["length_geodesic_m"] is None
    assert measured["geodesic_unavailable_reason"] == "disconnected graph"
    with pytest.raises(ValueError): InstanceExtractor().extract(np.empty((0, 3)), np.array([], dtype=int))


def test_legacy_xyz_loader_and_selected_weight_guard(tmp_path):
    _, path, _, _ = fixture(tmp_path, full=True)
    payload = load_checkpoint(path)
    payload.pop("contract_sha256")
    for key in ("features", "scheduler", "augmentation"):
        payload["config"].pop(key)
    save_checkpoint(path, payload)
    p = PyTorchPointNet2Predictor(path)
    assert p.metadata["selected_weights_verified"]
    payload["best_miou"] = .9
    save_checkpoint(path, payload)
    with pytest.raises(ValueError, match="selected checkpoint"):
        PyTorchPointNet2Predictor(path)


def test_verified_palette_batch_quantitative_path(tmp_path):
    manifest, path, _, _ = fixture(tmp_path)
    palette = tmp_path / "palette.json"
    palette.write_text(json.dumps({"schema_version": 1, "evidence": "synthetic exact palette",
                                  "rgb_to_instance": {"0,0,0": 0, "200,20,10": 1}}))
    p = PyTorchPointNet2Predictor(path, require_frozen_run1=False)
    result = evaluate_pipeline(manifest, tmp_path, p, tmp_path / "verified", instance_palette=palette)
    assert result["failures"] == 0
    for source in ("ground_truth", "run1_projected_prediction"):
        metrics = result["by_semantic_source"][source]["verified_instance_metrics"]
        assert metrics["plants"] == 1 and metrics["leaf_count_mae"] >= 0
