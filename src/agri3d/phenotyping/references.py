"""Explicit verified annotation adapters. No inference from color diversity."""
import csv
import json

import numpy as np
from scipy.optimize import linear_sum_assignment


def decode_verified_instances(cloud, palette_path):
    policy = json.loads(open(palette_path).read())
    if (policy.get("schema_version") != 1 or not isinstance(policy.get("evidence"), str)
            or not policy["evidence"].strip()):
        raise ValueError("Instance policy requires schema_version=1 and verification evidence")
    colors = cloud.attributes.get("colors")
    if colors is None:
        raise ValueError("Instance RGB annotations missing")
    mapping = policy.get("rgb_to_instance", {})
    if mapping.get("0,0,0") != 0:
        raise ValueError("Verified palette must map black stalk to instance 0")
    labels = np.full(len(colors), -1, dtype=int)
    seen = set()
    for key, value in mapping.items():
        rgb = [int(x) for x in key.split(",")]
        if len(rgb) != 3 or any(x < 0 or x > 255 for x in rgb):
            raise ValueError("Invalid palette RGB")
        if tuple(rgb) in seen:
            raise ValueError("Duplicate numeric RGB in palette")
        seen.add(tuple(rgb))
        if (type(value) is not int or value < 0
                or (tuple(rgb) != (0, 0, 0) and value == 0)
                or (tuple(rgb) == (0, 0, 0) and value != 0)):
            raise ValueError("Require positive leaf IDs and black-only stalk zero")
        labels[np.all(colors == rgb, axis=1)] = value
    if np.any(labels < 0):
        raise ValueError("Unmapped colors: instance GT remains unverified for this cloud")
    return labels, policy


def instance_metrics(predicted, target, threshold=0.5):
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Invalid instance IoU threshold")
    predicted, target = np.asarray(predicted), np.asarray(target)
    if (predicted.shape != target.shape or predicted.ndim != 1 or not len(target)
            or predicted.dtype.kind not in "iu" or target.dtype.kind not in "iu"
            or np.any(target < 0) or np.any(predicted < -1)):
        raise ValueError("Invalid aligned instance labels")
    pids, tids = np.unique(predicted[predicted > 0]), np.unique(target[target > 0])
    iou = np.zeros((len(pids), len(tids)))
    intersections = np.zeros_like(iou)
    for i, p in enumerate(pids):
        for j, t in enumerate(tids):
            intersection = np.count_nonzero((predicted == p) & (target == t))
            union = np.count_nonzero((predicted == p) | (target == t))
            iou[i, j] = intersection / union
            intersections[i, j] = intersection
    rows, cols = linear_sum_assignment(-iou)
    matches = [float(iou[i, j]) for i, j in zip(rows, cols) if iou[i, j] >= threshold]
    return {"iou_threshold": threshold, "matching": "Hungarian maximum summed point IoU",
            "predicted_leaf_count": len(pids), "reference_leaf_count": len(tids),
            "leaf_count_error": len(pids) - len(tids),
            "absolute_leaf_count_error": abs(len(pids) - len(tids)),
            "matched_leaves": len(matches), "matched_instance_iou": float(np.mean(matches)) if matches else None,
            "precision": len(matches) / len(pids) if len(pids) else None,
            "recall": len(matches) / len(tids) if len(tids) else None,
            "over_segmentation_count": int(np.maximum((intersections > 0).sum(0) - 1, 0).sum()),
            "under_segmentation_count": int(np.maximum((intersections > 0).sum(1) - 1, 0).sum()),
            "overlap_count_rule": "any nonzero point overlap; excludes stalk/unassigned"}


def load_trait_references(path, evidence):
    if not evidence:
        raise ValueError("Trait reference adapter requires verification evidence and units")
    result = {}
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"sample_id", "height_m", "leaf_count"} <= set(reader.fieldnames or []):
            raise ValueError("Reference CSV requires sample_id,height_m,leaf_count (blank optional)")
        for row in reader:
            key = row["sample_id"]
            if not key or key in result:
                raise ValueError("Missing/duplicate reference sample_id")
            record = {}
            for name in ("height_m", "leaf_count"):
                if row[name].strip():
                    value = float(row[name])
                    if not np.isfinite(value) or value < 0 or (name == "leaf_count" and value != int(value)):
                        raise ValueError("Invalid reference units/value")
                    record[name] = value
            result[key] = record
    return result
