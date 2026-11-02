"""Fixed descriptive fragmentation/merge-risk proxies, not instance accuracy."""
import numpy as np
from scipy.spatial import cKDTree

DIAGNOSTIC_POLICY = {
    "near_threshold_points_inclusive": [100, 200], "very_small_rejected_points_inclusive": 25,
    "extreme_component_to_accepted_median_ratio": 3., "tall_component_plant_height_fraction": .6,
    "wide_attachment_z_span_m": .15, "attachment_region_gap_m": .05,
    "attachment_region_min_unique_stalk_points": 5,
    "adjacent_count_collapse_fraction": .25, "adjacent_count_collapse_min_components": 3,
    "absorption_previous_component_coverage": .9,
}


def attachment_context(record):
    leaf = np.flatnonzero(record.semantic == 0)
    stalk = np.flatnonzero(record.semantic == 1)
    distance = np.full(len(record.xyz), np.inf)
    nearest = np.full(len(record.xyz), -1, dtype=int)
    if len(stalk) and len(leaf):
        d, ix = cKDTree(record.xyz[stalk]).query(record.xyz[leaf], workers=1)
        distance[leaf], nearest[leaf] = d, stalk[ix]
    return distance, nearest


def diagnose(record, structure, context, config, runtime_ms):
    xyz = record.xyz
    raw = np.asarray(structure.metadata["component_sizes"], dtype=int)
    accepted = np.asarray([len(leaf.point_indices) for leaf in structure.leaves], dtype=int)
    rejected = raw[raw < config.min_leaf_points]
    n_leaf = int((record.semantic == 0).sum())
    median = float(np.median(accepted)) if len(accepted) else None
    height = float(np.ptp(xyz[:, 2]))
    details = []
    distance, nearest = context
    for leaf in structure.leaves:
        ids = leaf.point_indices
        contacts = np.unique(nearest[ids][distance[ids] <= config.attachment_radius_m])
        z = np.sort(xyz[contacts, 2]) if len(contacts) else np.array([])
        regions = np.split(z, np.flatnonzero(np.diff(z) > .05) + 1) if len(z) else []
        region_count = sum(len(region) >= 5 for region in regions)
        span = float(np.ptp(z)) if len(z) else None
        z_span = float(np.ptp(xyz[ids, 2]))
        size_ratio = len(ids) / median if median else None
        flags = []
        if size_ratio is not None and size_ratio > 3.:
            flags.append("extreme_size")
        if height and z_span / height > .6:
            flags.append("large_vertical_span")
        if span is not None and span > .15:
            flags.append("wide_attachment_height_span")
        if region_count >= 2:
            flags.append("multiple_attachment_height_regions")
        details.append({"instance_id": leaf.instance_id, "points": len(ids),
            "vertical_span_m": z_span, "size_to_accepted_median": size_ratio,
            "attachment_z_span_m": span, "attachment_height_regions": int(region_count),
            "closest_stalk_distance_m": leaf.metadata["distance_to_stalk_m"],
            "heuristic_merge_flags": flags})
    distances = [leaf.metadata["distance_to_stalk_m"] for leaf in structure.leaves
                 if leaf.metadata["distance_to_stalk_m"] is not None]
    unassigned = len(structure.unassigned_indices)
    return {
        "raw_component_count": len(raw), "accepted_leaf_instances": len(accepted),
        "rejected_component_count": len(rejected), "rejected_point_count": int(rejected.sum()),
        "unassigned_leaf_points": unassigned, "unassigned_fraction": unassigned / n_leaf if n_leaf else 0.,
        "leaf_point_count": n_leaf, "accepted_instance_point_coverage": 1 - unassigned / n_leaf if n_leaf else None,
        "largest_component_points": int(raw.max()) if len(raw) else 0,
        "median_component_points": median,
        "median_raw_component_points": float(np.median(raw)) if len(raw) else None,
        "small_component_fraction": len(rejected) / len(raw) if len(raw) else None,
        "raw_to_accepted_component_ratio": len(raw) / len(accepted) if len(accepted) else None,
        "near_threshold_accepted_count": int(((accepted >= 100) & (accepted <= 200)).sum()),
        "near_threshold_accepted_fraction": float(((accepted >= 100) & (accepted <= 200)).mean()) if len(accepted) else None,
        "very_small_rejected_fraction": float((rejected <= 25).mean()) if len(rejected) else None,
        "attachment_consistent_fraction": float(np.mean(np.asarray(distances) <= config.attachment_radius_m)) if distances else None,
        "mean_attachment_distance_m": float(np.mean(distances)) if distances else None,
        "max_attachment_distance_m": float(np.max(distances)) if distances else None,
        "heuristic_merge_risk_components": sum(bool(d["heuristic_merge_flags"]) for d in details),
        "extreme_size_components": sum("extreme_size" in d["heuristic_merge_flags"] for d in details),
        "tall_components": sum("large_vertical_span" in d["heuristic_merge_flags"] for d in details),
        "wide_attachment_components": sum("wide_attachment_height_span" in d["heuristic_merge_flags"] for d in details),
        "multiple_attachment_components": sum("multiple_attachment_height_regions" in d["heuristic_merge_flags"] for d in details),
        "largest_to_accepted_median_ratio": float(raw.max() / median) if median else None,
        "graph_undirected_edges": structure.metadata["graph_undirected_edges"],
        "instance_runtime_ms": runtime_ms,
    }, {"raw_component_sizes": raw.tolist(), "accepted_components": details}


def partition_comparison(previous, current, leaf_mask):
    a, b = previous[leaf_mask], current[leaf_mask]
    if not len(a):
        return {"leaf_adjusted_rand_index": None, "absorbed_previous_components": 0,
                "current_components_absorbing_multiple": 0, "maximum_previous_components_combined": 0}
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    _, cell_counts = np.unique(np.column_stack([ai, bi]), axis=0, return_counts=True)
    ac, bc = np.bincount(ai), np.bincount(bi)
    pairs = lambda counts: float(np.sum(counts.astype(float) * (counts - 1) / 2))
    total = len(a) * (len(a) - 1) / 2
    expected = pairs(ac) * pairs(bc) / total if total else 0.
    maximum = (pairs(ac) + pairs(bc)) / 2
    ari = (pairs(cell_counts) - expected) / (maximum - expected) if maximum != expected else 1.
    absorbs = {}
    for old in np.unique(a[a > 0]):
        targets, counts = np.unique(b[a == old], return_counts=True)
        j = int(np.argmax(counts))
        if targets[j] > 0 and counts[j] / counts.sum() >= .9:
            absorbs[int(targets[j])] = absorbs.get(int(targets[j]), 0) + 1
    return {"leaf_adjusted_rand_index": float(ari),
            "absorbed_previous_components": sum(max(0, n - 1) for n in absorbs.values()),
            "current_components_absorbing_multiple": sum(n >= 2 for n in absorbs.values()),
            "maximum_previous_components_combined": max(absorbs.values(), default=0)}
