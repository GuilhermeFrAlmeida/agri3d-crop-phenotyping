"""Exact-filename metadata joins and descriptive associations, never genotype inference."""

import math
from collections import defaultdict
from datetime import date, datetime

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .metrics import distribution

CATEGORICAL_FIELDS = (
    "IsConnected",
    "IsEasyToCountLeaves",
    "AreLeavesTouching",
    "HasEar",
    "HasTassel",
)
FIELDS = (
    "Tagname",
    "IsConnected",
    "IsEasyToCountLeaves",
    "AreLeavesTouching",
    "Comment",
    "HasColor",
    "HasEar",
    "Size",
    "HasTassel",
)


def read_metadata(path):
    """Read the first worksheet. Duplicate/invalid keys and schema errors fail fast."""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise ImportError(
            "Metadata.xlsx requires pip install 'agri3d[analysis]'"
        ) from exc
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        sheet = workbook.worksheets[0]
        iterator = sheet.iter_rows(values_only=True)
        header = next(iterator, ())
        if any(header.count(field) != 1 for field in FIELDS):
            raise ValueError(f"Metadata worksheet requires unique columns: {FIELDS}")
        indices = {field: header.index(field) for field in FIELDS}
        result = {}
        for number, values in enumerate(iterator, 2):
            if all(value is None for value in values):
                continue
            row = {
                field: values[index] if index < len(values) else None
                for field, index in indices.items()
            }
            tag = row["Tagname"]
            if (
                not isinstance(tag, str)
                or not tag
                or tag != tag.strip()
                or tag.startswith("=")
            ):
                raise ValueError(f"Invalid exact Tagname at worksheet row {number}")
            if tag in result:
                raise ValueError(f"Duplicate Tagname: {tag}")
            # Keep source values visible, including formulas; never silently evaluate them.
            for field, value in row.items():
                if isinstance(value, (date, datetime)):
                    row[field] = value.isoformat()
                elif isinstance(value, float) and not math.isfinite(value):
                    row[field] = repr(value)  # Explicit malformed value, valid JSON.
            result[tag] = row
        return result
    finally:
        workbook.close()


def join_metadata(rows, metadata):
    """Tagname is the exact manifest path basename, including extension and case."""
    if len({row["Tagname"] for row in rows}) != len(rows):
        raise ValueError("Validation filenames are ambiguous for exact Tagname join")
    joined, unmatched = [], []
    for row in rows:
        source = metadata.get(row["Tagname"])
        joined.append(
            {
                **row,
                "metadata_matched": source is not None,
                **{
                    field: source[field] if source else None
                    for field in FIELDS
                    if field != "Tagname"
                },
            }
        )
        if source is None:
            unmatched.append(row["Tagname"])
    return joined, {
        "matched_count": len(rows) - len(unmatched),
        "unmatched_validation_tags": unmatched,
        "unused_workbook_rows": len(metadata) - (len(rows) - len(unmatched)),
        "join_key": "exact manifest PLY basename == Tagname (case and extension preserved)",
    }


def boolean_value(value):
    if value is None or value == "":
        return None, "missing"
    if isinstance(value, bool):
        return str(value).lower(), None
    if isinstance(value, (int, float)) and value in (0, 1):
        return str(bool(value)).lower(), None
    if isinstance(value, str) and value.strip().lower() in (
        "yes",
        "no",
        "true",
        "false",
        "0",
        "1",
    ):
        return str(value.strip().lower() in ("yes", "true", "1")).lower(), None
    return None, "malformed: expected boolean, 0/1, yes/no or true/false"


def numeric_value(value):
    if value is None or value == "":
        return None, "missing"
    if isinstance(value, bool):
        return None, "malformed: boolean is not a continuous value"
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None, "malformed: expected finite numeric value"
    if not math.isfinite(number):
        return None, "malformed: expected finite numeric value"
    return number, None


def correlation(rows, values, metric):
    pairs = [
        (value, row[metric])
        for row, value in zip(rows, values)
        if value is not None and row[metric] is not None
    ]
    result = {
        "count": len(pairs),
        "excluded_count": len(rows) - len(pairs),
        "pearson": None,
        "spearman": None,
        "reason": None,
    }
    if len(pairs) < 3:
        result["reason"] = "At least 3 paired plants required"
        return result
    x, y = np.asarray(pairs, dtype=float).T
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        result["reason"] = "Constant variable; correlation undefined"
        return result
    result["pearson"] = float(pearsonr(x, y).statistic)
    result["spearman"] = float(spearmanr(x, y).statistic)
    return result


def conditioned_summary(rows, include_metadata=False):
    """Descriptive summaries; no causal claims or multiple-comparison significance tests."""
    result = {"categorical": {}, "continuous": {}, "value_issues": []}
    if include_metadata:
        for field in CATEGORICAL_FIELDS:
            groups = defaultdict(list)
            missing, malformed = 0, 0
            for row in rows:
                value, issue = boolean_value(row[field])
                if issue:
                    missing += issue == "missing"
                    malformed += issue != "missing"
                    result["value_issues"].append(
                        {
                            "sample_id": row["sample_id"],
                            "field": field,
                            "value": row[field],
                            "reason": issue,
                        }
                    )
                else:
                    groups[value].append(row)
            summaries = {}
            for value, group in sorted(groups.items()):
                summaries[value] = {"count": len(group), "small_group": len(group) < 5}
                for metric in ("mIoU", "class_1_iou"):
                    stats = distribution([row[metric] for row in group])
                    summaries[value][metric] = {
                        key: stats[key]
                        for key in ("count", "missing_count", "mean", "median")
                    }
            result["categorical"][field] = {
                "groups": summaries,
                "missing_count": missing,
                "malformed_count": malformed,
            }
    for field in ["Size", "gt_stalk_ratio"] if include_metadata else ["gt_stalk_ratio"]:
        values = []
        for row in rows:
            value, issue = numeric_value(row[field])
            values.append(value)
            if issue:
                result["value_issues"].append(
                    {
                        "sample_id": row["sample_id"],
                        "field": field,
                        "value": row[field],
                        "reason": issue,
                    }
                )
        result["continuous"][field] = {
            metric: correlation(rows, values, metric)
            for metric in ("mIoU", "class_1_iou")
        }
    return result
