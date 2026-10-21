"""Bounded full-resolution normal audit on train/val only, with repeat checks."""

from pathlib import Path

import numpy as np

from agri3d.io.ply import PLYReader
from agri3d.preprocessing.filtering import CloudFilter
from agri3d.preprocessing.normals import (
    normal_implementation_fingerprint,
    validate_normals,
)
from agri3d.training.pointnet2_runner import (
    experiment_contract,
    make_dataset,
    read_manifest,
    write_json,
)


def audit_normals(
    manifest, root, output_path, config, split="train", max_plants=5, runtime=None
):
    if split not in ("train", "val"):
        raise ValueError("Normal audit permits train/val only; test is forbidden")
    if type(max_plants) is not int or max_plants < 1:
        raise ValueError("max_plants must be positive")
    config.validate()
    output = Path(output_path)
    if output.exists():
        raise ValueError("Normal audit requires a new output file")
    records, digest = read_manifest(manifest)
    dataset = make_dataset(records, root, split, config, runtime)
    if not dataset.feature_policy.channels:
        raise ValueError("Normal audit requires xyz_normals mode")
    rows, lengths = [], []
    component_sum, component_square = np.zeros(3), np.zeros(3)
    component_min, component_max = np.full(3, np.inf), np.full(3, -np.inf)
    for index, record in enumerate(dataset.records[:max_plants]):
        path = dataset.root / record["path"]
        # Load/populate cache through the actual Dataset path, then recompute directly.
        sample = dataset[index]
        cloud = dataset._normalized_cloud(path)
        normals = cloud.attributes["normals"]
        validate_normals(normals, len(cloud.xyz))
        repeated = dataset._normalized_cloud(path)
        deterministic = np.array_equal(normals, repeated.attributes["normals"])
        raw_cleaned = CloudFilter().remove_nan_inf(PLYReader(path).load_point_cloud())
        aligned = np.array_equal(
            cloud.attributes["source_index"], raw_cleaned.attributes["source_index"]
        )
        source = sample["source_index"].numpy()
        positions = np.searchsorted(cloud.attributes["source_index"], source)
        aligned = aligned and np.array_equal(
            normals[positions], sample["features"].numpy()
        )
        aligned = aligned and np.array_equal(
            cloud.xyz[positions], sample["points"].numpy()
        )
        aligned = aligned and np.array_equal(
            cloud.attributes["label"][positions], sample["labels"].numpy()
        )
        if not deterministic or not aligned:
            raise ValueError(
                f"Normal determinism/alignment audit failed: {record['sample_id']}"
            )
        values = np.linalg.norm(normals.astype(np.float64), axis=1)
        lengths.append(values)
        component_sum += normals.astype(np.float64).sum(0)
        component_square += (normals.astype(np.float64) ** 2).sum(0)
        component_min = np.minimum(component_min, normals.min(0))
        component_max = np.maximum(component_max, normals.max(0))
        rows.append(
            {
                "sample_id": record["sample_id"],
                "cleaned_points": len(normals),
                "finite_normal_ratio": float(np.isfinite(normals).all(1).mean()),
                "degenerate_neighborhood_count": int(
                    cloud.attributes["normal_degenerate"].sum()
                ),
                "effective_k": cloud.metadata["normal_estimation"]["effective_k"],
                "deterministic": deterministic,
                "source_alignment_verified": aligned,
            }
        )
    values = np.concatenate(lengths)
    total = len(values)
    mean = component_sum / total
    result = {
        "split": split,
        "plants_checked": len(rows),
        "points_checked": total,
        "finite_normal_ratio": 1.0,
        "unit_norm_tolerance": 1e-6,
        "norm_distribution": {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p01": float(np.percentile(values, 1)),
            "median": float(np.median(values)),
            "p99": float(np.percentile(values, 99)),
            "max": float(values.max()),
        },
        "degenerate_neighborhood_count": sum(
            r["degenerate_neighborhood_count"] for r in rows
        ),
        "normal_component_statistics": {
            "axes": ["x", "y", "z"],
            "min": component_min.tolist(),
            "max": component_max.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(component_square / total - mean**2, 0)).tolist(),
        },
        "deterministic": True,
        "passed": True,
        "plants": rows,
        "contract": experiment_contract(config, digest),
        "normal_implementation_sha256": normal_implementation_fingerprint(),
        "scope": "Full cleaned clouds; normals recomputed twice plus Dataset sampling/cache alignment. No training or test reads. Canonical signs and fallback normals are not physical outward orientation; degeneracy counts require review.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result
