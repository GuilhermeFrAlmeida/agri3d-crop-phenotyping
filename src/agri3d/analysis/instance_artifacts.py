"""Read-only reuse of the exact bounded Phase 4/5 val5 outputs."""
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData

from agri3d.pipeline.config import PipelineConfig
from agri3d.pipeline.provenance import sha256_file
from agri3d.preprocessing.labels import validate_labels

PLANTS = ("0006", "0014", "0015", "0022", "0029")
SOURCES = {"oracle": "ground_truth", "predicted": "run1_projected_prediction"}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass
class ReusedPlant:
    sample_id: str
    source: str
    xyz: np.ndarray
    semantic: np.ndarray
    original_indices: np.ndarray
    baseline_instances: np.ndarray
    report: dict
    geometry_sha256: str


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes input root")
    return path


def load_val5(root, config=None, plants=None, split="val"):
    if split != "val":
        raise ValueError("Phase 4A permits the fixed validation five only; test is inaccessible")
    plants = list(PLANTS if plants is None else plants)
    if len(plants) != 5 or set(plants) != set(PLANTS):
        raise ValueError("Use exactly 0006 0014 0015 0022 0029; no subset replacement")
    root = Path(root).resolve()
    config = config or PipelineConfig()
    reused = {}

    def checked_file(path):
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        info = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
        reused[relative] = info
        return raw, info

    root_prov = json.loads(checked_file(root / "provenance.json")[0])
    original_config = json.loads(checked_file(root / "configuration.json")[0])
    if original_config != config.to_dict():
        raise ValueError("Input preprocessing/phenotype/default instance config differs from supplied config")
    baseline = {"method": "geometric_graph_v1", "neighborhood_radius_m": .015,
                "max_neighbors": 16, "min_leaf_points": 100, "attachment_radius_m": .03}
    if original_config["instances"] != baseline:
        raise ValueError("Source must be the original Phase 4/5 default, not a promoted candidate")
    selected = root_prov.get("selected_sample_ids", [])
    if len(selected) != 5 or set(selected) != set(PLANTS):
        raise ValueError("Input provenance does not identify the frozen val5 cohort")
    if root_prov.get("split", "val") != "val":
        raise ValueError("Test/non-validation artifacts rejected")
    if root_prov.get("instance_gt_status") != "unverified" or root_prov.get("instance_palette_sha256"):
        raise ValueError("This diagnostic protocol expects unverified instance GT; review verified references separately")
    model = root_prov.get("semantic_model") or {}
    if (model.get("model_family") != "pointnet2_ssg" or model.get("feature_mode") != "xyz"
            or model.get("input_points") != 10000 or model.get("selected_weights_verified") is not True
            or model.get("repository_metadata", {}).get("frozen_run1_policy_enforced") is not True
            or model.get("repository_metadata", {}).get("compatible_experiment_policy") != "run1_xyz"
            or not model.get("checkpoint_sha256") or not root_prov.get("manifest_sha256")
            or model.get("manifest_sha256") != root_prov["manifest_sha256"]):
        raise ValueError("Missing/inconsistent frozen XYZ model/manifest provenance")
    with open(root / "per_plant.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    checked_file(root / "per_plant.csv")
    index = {}
    for row in rows:
        key = (row.get("sample_id"), row.get("semantic_source"))
        if key[0] not in PLANTS or key[1] not in SOURCES.values() or key in index:
            raise ValueError("Unexpected/duplicate plant or semantic mode")
        index[key] = row
    expected = {(plant, source) for plant in PLANTS for source in SOURCES.values()}
    if set(index) != expected:
        raise ValueError("Both original semantic modes must exist for all five plants")
    records = []
    for source in SOURCES.values():
        for plant in PLANTS:
            folder = _inside(root, index[(plant, source)]["artifact_dir"])
            inventory = json.loads(checked_file(folder / "artifacts.json")[0]).get("files", {})
            required = ("phenotype.json", "structure.ply", "semantic_labels.npy", "instance_labels.npy")
            for name in required:
                path = _inside(root, (folder / name).relative_to(root))
                _, info = checked_file(path)
                if inventory.get(name) != info:
                    raise ValueError(f"Artifact checksum/size mismatch: {plant}/{source}/{name}")
            report = json.loads((folder / "phenotype.json").read_text())
            if (report.get("schema_version") != 1 or report.get("plant_id") != plant
                    or report.get("semantic_source") != source
                    or report.get("split", "val") != "val"
                    or report.get("provenance", {}).get("split", "val") != "val"):
                raise ValueError("Artifact identity/schema/split mismatch")
            if report["provenance"].get("configuration") != original_config:
                raise ValueError("Per-plant configuration mismatch")
            if report["provenance"].get("manifest_sha256") != root_prov["manifest_sha256"]:
                raise ValueError("Per-plant manifest provenance mismatch")
            if source == SOURCES["predicted"] and (
                    report.get("semantic_model", {}).get("checkpoint_sha256") != model["checkpoint_sha256"]):
                raise ValueError("Predicted artifact checkpoint mismatch")
            vertex = PlyData.read(str(folder / "structure.ply"))["vertex"]
            required_fields = {"x", "y", "z", "source_index", "semantic_label", "instance_id"}
            if not required_fields <= set(vertex.data.dtype.names):
                raise ValueError("Structure PLY lacks aligned scalar labels/source rows")
            xyz = np.column_stack([vertex[a] for a in "xyz"]).astype(np.float32)
            raw_ids = np.asarray(vertex["source_index"], dtype=np.int64).copy()
            semantic = validate_labels(np.load(folder / "semantic_labels.npy", allow_pickle=False))
            baseline_labels = np.load(folder / "instance_labels.npy", allow_pickle=False)
            n = len(xyz)
            if (not n or xyz.shape != (n, 3) or not np.isfinite(xyz).all()
                    or len(raw_ids) != n or np.any(raw_ids < 0) or np.any(np.diff(raw_ids) <= 0)
                    or semantic.shape != (n,) or baseline_labels.shape != (n,)
                    or baseline_labels.dtype.kind not in "iu" or np.any(baseline_labels < -1)
                    or report.get("clean_point_count") != n):
                raise ValueError("Invalid full-cloud geometry/label/source alignment")
            if (not np.array_equal(semantic, vertex["semantic_label"])
                    or not np.array_equal(baseline_labels, vertex["instance_id"])
                    or not np.array_equal(baseline_labels == 0, semantic == 1)):
                raise ValueError("PLY/NPY labels or stalk accounting differ")
            if report["structure"]["accepted_leaf_instances"] != len(np.unique(baseline_labels[baseline_labels > 0])):
                raise ValueError("Saved component count differs from saved partition")
            fingerprint = hashlib.sha256(xyz.tobytes() + raw_ids.tobytes()).hexdigest()
            for array in (xyz, raw_ids, semantic, baseline_labels):
                array.flags.writeable = False
            records.append(ReusedPlant(plant, source, xyz, semantic, raw_ids,
                                       baseline_labels, report, fingerprint))
    for plant in PLANTS:
        a, b = [r for r in records if r.sample_id == plant]
        if (a.geometry_sha256 != b.geometry_sha256
                or a.report["provenance"]["input_sha256"] != b.report["provenance"]["input_sha256"]):
            raise ValueError("Oracle/predicted geometry or recorded raw input hash differs")
    provenance = {
        "selected_sample_ids": list(PLANTS), "scope": "same prior validation five only",
        "split_assertion": "val, caller-supplied prior validation artifacts; original batch lacks an explicit split field",
        "checkpoint_sha256": model["checkpoint_sha256"], "manifest_sha256": root_prov["manifest_sha256"],
        "source_code_commit": root_prov.get("code_commit"), "source_versions": root_prov.get("versions"),
        "reused_files": reused, "input_artifact_fingerprint": canonical_hash(reused),
        "not_recomputed": ["PLY cleaning", "normalization", "10k sampling", "neural inference", "semantic projection", "phenotype formulas"],
        "instance_gt_status": "unverified; original annotation RGB is not in these reusable PLYs",
    }
    return records, provenance


def verify_unchanged(root, source):
    root = Path(root).resolve()
    for relative, info in source["reused_files"].items():
        path = _inside(root, relative)
        if path.stat().st_size != info["size_bytes"] or sha256_file(path) != info["sha256"]:
            raise ValueError("Input artifact changed during sweep: " + relative)
