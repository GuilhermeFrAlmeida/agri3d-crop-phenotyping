"""Validation-only evaluation, oracle/predicted separation and controlled summaries."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from agri3d.phenotyping.references import (
    decode_verified_instances,
    instance_metrics,
    load_trait_references,
)
from agri3d.training.evaluate import SegmentationEvaluator
from agri3d.training.splits import validate_manifest

from .analyze import analyze_plant
from .artifacts import write_json, write_plant
from .config import PipelineConfig
from .provenance import provenance, sha256_file


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {"count": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())} if len(a) else None


def write_csv(path, rows):
    fields = sorted({k for r in rows for k in r}) or ["sample_id"]
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_pipeline(manifest, root, predictor, output_dir, config=None, split="val",
                      max_plants=None, mode="both", fail_fast=False, visualize=False,
                      instance_palette=None, reference_traits=None, reference_evidence=None,
                      code_commit=None):
    # Reject test before checkpoint/dataset/reference I/O.
    if split != "val":
        raise ValueError("Phenotyping evaluation permits validation only; test is inaccessible")
    if max_plants is not None and (type(max_plants) is not int or max_plants < 1):
        raise ValueError("max_plants must be positive")
    if mode not in ("both", "predicted", "oracle"):
        raise ValueError("Invalid evaluation mode")
    config = config or PipelineConfig()
    raw_manifest = Path(manifest).read_bytes()
    records = validate_manifest(json.loads(raw_manifest))
    manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    if predictor is not None and predictor.metadata.get("manifest_sha256") != manifest_hash:
        raise ValueError("Frozen manifest differs from model provenance")
    if mode != "oracle" and predictor is None:
        raise ValueError("Predicted mode requires a predictor")
    selected = sorted((r for r in records if r["split"] == "val"), key=lambda r: r["sample_id"])
    if not selected:
        raise ValueError("Manifest contains no validation plants")
    selected = selected[:max_plants]
    references = load_trait_references(reference_traits, reference_evidence) if reference_traits else {}
    out = Path(output_dir)
    if out.exists():
        raise ValueError("Evaluation output must not exist")
    out.mkdir(parents=True)
    prov = {**provenance(code_commit), "manifest_sha256": manifest_hash,
            "selected_sample_ids": [r["sample_id"] for r in selected],
            "semantic_model": predictor.metadata if predictor else None,
            "instance_palette_sha256": sha256_file(instance_palette) if instance_palette else None,
            "reference_traits_sha256": sha256_file(reference_traits) if reference_traits else None,
            "reference_evidence": reference_evidence,
            "instance_gt_status": "explicit verified palette" if instance_palette else "unverified",
            "phenotype_reference_status": "explicit external references" if references else "unavailable"}
    write_json(out / "configuration.json", config.to_dict())
    write_json(out / "provenance.json", prov)
    rows, timings, failures = [], [], []
    sampled_eval, projected_eval = SegmentationEvaluator(), SegmentationEvaluator()
    root = Path(root).resolve()
    for record in selected:
        path = (root / record["path"]).resolve()
        for source in (["ground_truth", "run1_projected_prediction"] if mode == "both" else
                       ["ground_truth" if mode == "oracle" else "run1_projected_prediction"]):
            try:
                if not path.is_relative_to(root):
                    raise ValueError("Manifest path escapes root")
                result = analyze_plant(path, predictor, config, record["sample_id"], source, True, code_commit)
                instance_gt = None
                if instance_palette:
                    instance_gt, policy = decode_verified_instances(result.cloud, instance_palette)
                    result.report["instance_metrics"] = instance_metrics(result.instance_labels, instance_gt)
                    result.report["provenance"]["instance_policy"] = policy
                result.report["provenance"].update(manifest_sha256=manifest_hash,
                    instance_gt_status=prov["instance_gt_status"],
                    phenotype_reference_status=prov["phenotype_reference_status"])
                row = {"sample_id": record["sample_id"], "semantic_source": source,
                       "height_m": result.report["phenotypes"]["height_m"],
                       "leaf_count": len(result.structure.leaves),
                       "unassigned_fraction": result.structure.metadata["unassigned_fraction_of_leaf"]}
                volume = result.report["phenotypes"]["canopy_volume"]
                row.update(volume_convex_hull_m3=volume["convex_hull_m3"], volume_voxel_m3=volume["voxel_occupancy_m3"])
                if volume["convex_hull_m3"] is not None:
                    row["volume_hull_minus_voxel_m3"] = volume["convex_hull_m3"] - volume["voxel_occupancy_m3"]
                differences = [leaf["length_geodesic_m"] - leaf["length_pca_m"] for leaf in result.report["leaves"]
                               if leaf["length_geodesic_m"] is not None and leaf["length_pca_m"] is not None]
                row["length_geodesic_minus_pca_mean_m"] = float(np.mean(differences)) if differences else None
                result.report["reference_traits"] = references.get(record["sample_id"])
                result.report["provenance"]["reference_evidence"] = reference_evidence
                for name, value in references.get(record["sample_id"], {}).items():
                    row[name + "_error"] = row[name] - value
                for domain, metrics in result.report["semantic_metrics"].items():
                    for metric, value in metrics.items():
                        row[domain + "_" + metric] = value
                if instance_gt is not None:
                    row.update({"instance_" + k: v for k, v in result.report["instance_metrics"].items()
                                if isinstance(v, (int, float)) or v is None})
                key = hashlib.sha256(record["sample_id"].encode()).hexdigest()[:20]
                row["artifact_dir"] = str(Path("plants") / key / source)
                write_plant(out / row["artifact_dir"], result, visualize)
                if result.prediction is not None:
                    raw_ids = result.sampled.attributes["source_index"][result.sampled.attributes["valid_mask"]]
                    ids = np.searchsorted(result.cloud.attributes["source_index"], raw_ids)
                    sampled_eval.update(result.prediction.labels[result.sampled.attributes["valid_mask"]], result.target_labels[ids])
                    projected_eval.update(result.semantic_labels, result.target_labels)
                rows.append(row)
                timings.append({"sample_id": record["sample_id"], "semantic_source": source, **result.report["timing_ms"]})
            except Exception as exc:
                failures.append({"sample_id": record["sample_id"], "semantic_source": source,
                                 "error_type": type(exc).__name__, "error": str(exc)})
                write_json(out / "failures.json", failures)
                if fail_fast:
                    _finish(out, rows, timings, failures, sampled_eval, projected_eval)
                    raise
    return _finish(out, rows, timings, failures, sampled_eval, projected_eval)


def _finish(out, rows, timings, failures, sampled_eval, projected_eval):
    summary = {"status": "IMPLEMENTED — NOT YET EXPERIMENTALLY QUALIFIED",
               "successful_plant_modes": len(rows), "failures": len(failures), "by_semantic_source": {}}
    for source in sorted({r["semantic_source"] for r in rows}):
        subset = [r for r in rows if r["semantic_source"] == source]
        subset_times = [t for t in timings if t["semantic_source"] == source]
        traits = {}
        for key in ("height_m_error", "leaf_count_error"):
            errors = [r[key] for r in subset if key in r]
            if errors:
                a = np.asarray(errors)
                traits[key] = {"count": len(a), "mae": float(np.abs(a).mean()),
                               "rmse": float(np.sqrt((a*a).mean())), "bias": float(a.mean()),
                               "exact_match_fraction": float((a == 0).mean())}
        verified = [r for r in subset if "instance_absolute_leaf_count_error" in r]
        instance_summary = None
        if verified:
            instance_summary = {
                "plants": len(verified),
                "leaf_count_mae": float(np.mean([r["instance_absolute_leaf_count_error"] for r in verified])),
                "over_segmentation_count_total": sum(r["instance_over_segmentation_count"] for r in verified),
                "under_segmentation_count_total": sum(r["instance_under_segmentation_count"] for r in verified),
                "macro_precision": distribution([r["instance_precision"] for r in verified if r["instance_precision"] is not None]),
                "macro_recall": distribution([r["instance_recall"] for r in verified if r["instance_recall"] is not None]),
            }
        summary["by_semantic_source"][source] = {
            "verified_instance_metrics": instance_summary,
            "plant_count": len(subset), "trait_reference_errors": traits,
            "volume_hull_minus_voxel_m3": distribution([r["volume_hull_minus_voxel_m3"] for r in subset if "volume_hull_minus_voxel_m3" in r]),
            "length_geodesic_minus_pca_mean_m": distribution([r["length_geodesic_minus_pca_mean_m"] for r in subset if r["length_geodesic_minus_pca_mean_m"] is not None]),
            "leaf_count_diagnostic": distribution([r["leaf_count"] for r in subset]),
            "timing_ms": {key: distribution([r[key] for r in subset_times if key in r])
                          for key in sorted({k for r in subset_times for k in r} - {"sample_id", "semantic_source"})}}
    if sampled_eval.confusion.sum():
        summary["semantic_metrics"] = {"sampled_10k": sampled_eval.compute(), "projected_full_cloud": projected_eval.compute()}
    write_csv(out / "per_plant.csv", rows)
    write_csv(out / "phenotypes.csv", rows)
    write_csv(out / "timings.csv", timings)
    write_json(out / "failures.json", failures)
    write_json(out / "summary.json", summary)
    return summary
