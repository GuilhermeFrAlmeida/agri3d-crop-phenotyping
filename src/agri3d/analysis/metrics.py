"""Binary per-plant diagnostics using the existing evaluator's conventions."""

import numpy as np

from agri3d.training.evaluate import SegmentationEvaluator


def sample_metrics(sample_id, predictions, targets, valid_mask):
    """Rates use the corresponding GT class count; undefined rates are None."""
    if np.asarray(targets).ndim != 1:
        raise ValueError("Per-sample targets must have shape (N,)")
    evaluator = SegmentationEvaluator(2)
    evaluator.update(predictions, targets, valid_mask)
    result = {"sample_id": sample_id, **evaluator.compute()}
    confusion = evaluator.confusion
    total = int(confusion.sum())
    for i, name in enumerate(("leaf", "stalk")):
        gt_count = int(confusion[i].sum())
        pred_count = int(confusion[:, i].sum())
        result[f"gt_{name}_count"] = gt_count
        result[f"gt_{name}_ratio"] = gt_count / total
        result[f"predicted_{name}_count"] = pred_count
        result[f"predicted_{name}_ratio"] = pred_count / total
    for gt, pred, name in ((0, 1, "leaf_as_stalk"), (1, 0, "stalk_as_leaf")):
        count = int(confusion[gt, pred])
        denominator = int(confusion[gt].sum())
        result[f"{name}_count"] = count
        result[f"{name}_rate"] = count / denominator if denominator else None
    return result, confusion


def distribution(values):
    """Population standard deviation across available plant metrics (ddof=0)."""
    array = np.asarray([v for v in values if v is not None], dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite diagnostic metric")
    result = {"count": len(array), "missing_count": len(values) - len(array)}
    result.update(
        {
            name: float(function(array)) if len(array) else None
            for name, function in (
                ("mean", np.mean),
                ("median", np.median),
                ("std", np.std),
                ("min", np.min),
                ("max", np.max),
            )
        }
    )
    return result


def ranked(rows, field, count, descending=False):
    available = [row for row in rows if row[field] is not None]
    return sorted(
        available,
        key=lambda row: (-row[field] if descending else row[field], row["sample_id"]),
    )[:count]


def select_visualization_cases(rows):
    """Fixed 3/5/5 case selection; sample ID resolves ties deterministically."""
    groups = {
        "best_mIoU": ranked(rows, "mIoU", 3, True),
        "worst_mIoU": ranked(rows, "mIoU", 5),
        "worst_stalk_iou": ranked(rows, "class_1_iou", 5),
    }
    reasons = {}
    for reason, selected in groups.items():
        for row in selected:
            reasons.setdefault(row["sample_id"], []).append(reason)
    return reasons


def metric_summary(rows, top_k):
    if not rows or top_k < 1:
        raise ValueError("Require samples and positive top_k")
    return {
        "sample_count": len(rows),
        "per_sample_statistics": {
            field: distribution([row[field] for row in rows])
            for field in ("mIoU", "class_0_iou", "class_1_iou")
        },
        "rankings": {
            "best_mIoU": ranked(rows, "mIoU", top_k, True),
            "worst_mIoU": ranked(rows, "mIoU", top_k),
            "lowest_stalk_recall": ranked(rows, "class_1_recall", top_k),
            "highest_stalk_false_negative_rate": ranked(
                rows, "stalk_as_leaf_rate", top_k, True
            ),
        },
        "visualization_cases": select_visualization_cases(rows),
    }
