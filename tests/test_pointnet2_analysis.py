"""Diagnostics on synthetic validation clouds; no real data or training required."""

import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest
import torch
from openpyxl import Workbook
from plyfile import PlyData, PlyElement

from agri3d.analysis.metadata import (
    FIELDS,
    conditioned_summary,
    join_metadata,
    read_metadata,
)
from agri3d.analysis.metrics import (
    distribution,
    metric_summary,
    sample_metrics,
    select_visualization_cases,
)
from agri3d.analysis.pointnet2 import analyze_checkpoint, evaluate_samples
from agri3d.analysis.visualization import error_categories, save_views
from agri3d.models import PointNet2Config, PointNet2Segmentation
from agri3d.training.checkpoints import save_checkpoint, seed_everything
from agri3d.training.pointnet2_runner import (
    RunConfig,
    experiment_contract,
    make_dataset,
    read_manifest,
    validate_checkpoint,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def diagnostic_row(name="plant", tag="plant.ply"):
    row, _ = sample_metrics(
        name,
        np.array([0, 1, 1, 0, 1]),
        np.array([0, 0, 1, 1, 1]),
        np.ones(5, dtype=bool),
    )
    row["Tagname"] = tag
    return row


def workbook(path, records):
    book = Workbook()
    sheet = book.active
    sheet.append(list(FIELDS))
    for record in records:
        sheet.append([record.get(field) for field in FIELDS])
    book.save(path)


def metadata_record(tag, **kwargs):
    return {
        "Tagname": tag,
        **{field: True for field in FIELDS if field != "Tagname"},
        "Comment": "test",
        "Size": 2,
        **kwargs,
    }


def test_confusion_errors_ratios_and_padding():
    row, confusion = sample_metrics(
        "x",
        np.array([0, 1, 1, 0, 1, 99]),
        np.array([0, 0, 1, 1, 1, -1]),
        np.array([True] * 5 + [False]),
    )
    np.testing.assert_array_equal(confusion, [[1, 1], [1, 2]])
    assert row["evaluated_points"] == 5
    assert row["accuracy"] == pytest.approx(3 / 5)
    assert row["class_0_iou"] == pytest.approx(1 / 3)
    assert row["class_1_iou"] == pytest.approx(1 / 2)
    assert row["mIoU"] == pytest.approx(5 / 12)
    assert row["class_0_precision"] == row["class_0_recall"] == 0.5
    assert row["class_1_f1"] == pytest.approx(2 / 3)
    assert row["gt_leaf_count"] == row["predicted_leaf_count"] == 2
    assert row["gt_stalk_ratio"] == row["predicted_stalk_ratio"] == 0.6
    assert row["leaf_as_stalk_count"] == row["stalk_as_leaf_count"] == 1
    assert row["leaf_as_stalk_rate"] == 0.5
    assert row["stalk_as_leaf_rate"] == pytest.approx(1 / 3)


def test_absent_classes_and_invalid_shapes():
    row, _ = sample_metrics(
        "leaf", np.zeros(3, dtype=int), np.zeros(3, dtype=int), np.ones(3, dtype=bool)
    )
    assert row["class_1_iou"] is None and row["stalk_as_leaf_rate"] is None
    assert row["mIoU"] == 1
    assert distribution([None, 0.5, 1]) == {
        "count": 2,
        "missing_count": 1,
        "mean": 0.75,
        "median": 0.75,
        "std": 0.25,
        "min": 0.5,
        "max": 1.0,
    }
    with pytest.raises(ValueError, match="No evaluated"):
        sample_metrics(
            "x", np.zeros(3, dtype=int), np.zeros(3, dtype=int), np.zeros(3, dtype=bool)
        )
    with pytest.raises(ValueError, match="shape"):
        sample_metrics(
            "x",
            np.zeros((1, 3), dtype=int),
            np.zeros((1, 3), dtype=int),
            np.ones((1, 3), dtype=bool),
        )


def test_rankings_deduplicate_ties_and_nulls():
    rows = [
        {**diagnostic_row(str(i)), "mIoU": i / 20, "class_1_iou": (20 - i) / 20}
        for i in range(20)
    ]
    selected = select_visualization_cases(rows)
    assert len(selected) == 10  # top3 and bottom stalk5 overlap
    assert selected["19"] == ["best_mIoU", "worst_stalk_iou"]
    summary = metric_summary(rows, 2)
    assert [r["sample_id"] for r in summary["rankings"]["best_mIoU"]] == ["19", "18"]
    tied = [diagnostic_row("b"), diagnostic_row("a")]
    assert metric_summary(tied, 1)["rankings"]["worst_mIoU"][0]["sample_id"] == "a"
    tied[0]["class_1_iou"] = None
    assert "worst_stalk_iou" not in select_visualization_cases(tied)["b"]


def test_exact_metadata_join(tmp_path):
    path = tmp_path / "Metadata.xlsx"
    workbook(path, [metadata_record("plant.ply"), metadata_record("other.ply")])
    rows = [diagnostic_row(), diagnostic_row("other", "Plant.ply")]
    joined, info = join_metadata(rows, read_metadata(path))
    assert joined[0]["metadata_matched"] and joined[0]["IsConnected"] is True
    assert not joined[1]["metadata_matched"]
    assert info["unmatched_validation_tags"] == ["Plant.ply"]
    assert info["unused_workbook_rows"] == 1
    with pytest.raises(ValueError, match="ambiguous"):
        join_metadata([diagnostic_row("a"), diagnostic_row("b")], read_metadata(path))


@pytest.mark.parametrize(
    "tags, message", [(["x.ply", "x.ply"], "Duplicate"), ([" x.ply"], "Invalid")]
)
def test_invalid_metadata_keys(tmp_path, tags, message):
    path = tmp_path / "Metadata.xlsx"
    workbook(path, [metadata_record(tag) for tag in tags])
    with pytest.raises(ValueError, match=message):
        read_metadata(path)


def test_metadata_schema(tmp_path):
    book = Workbook()
    book.active.append(["Tagname"])
    path = tmp_path / "bad.xlsx"
    book.save(path)
    with pytest.raises(ValueError, match="columns"):
        read_metadata(path)


def test_conditioned_stats_missing_malformed_and_constant():
    rows = []
    metadata = {}
    for i in range(5):
        row = {
            **diagnostic_row(str(i), f"{i}.ply"),
            "mIoU": 0.5 + i / 10,
            "class_1_iou": 0.3 + i / 10,
            "gt_stalk_ratio": 0.1 + i / 10,
        }
        rows.append(row)
        metadata[f"{i}.ply"] = metadata_record(
            f"{i}.ply",
            Size=[1, "2", 3, "large", None][i],
            IsConnected=[True, "yes", False, "unknown", None][i],
        )
    joined, _ = join_metadata(rows, metadata)
    summary = conditioned_summary(joined, True)
    connected = summary["categorical"]["IsConnected"]
    assert connected["groups"]["true"]["count"] == 2
    assert connected["groups"]["true"]["mIoU"]["mean"] == pytest.approx(0.55)
    assert connected["missing_count"] == connected["malformed_count"] == 1
    assert summary["continuous"]["Size"]["mIoU"]["count"] == 3
    assert summary["continuous"]["Size"]["mIoU"]["pearson"] == pytest.approx(1.0)
    assert summary["continuous"]["gt_stalk_ratio"]["class_1_iou"][
        "spearman"
    ] == pytest.approx(1.0)
    assert len(summary["value_issues"]) == 4
    constant = conditioned_summary(
        [diagnostic_row("a"), diagnostic_row("b"), diagnostic_row("c")]
    )
    assert "Constant" in constant["continuous"]["gt_stalk_ratio"]["mIoU"]["reason"]
    short = conditioned_summary([diagnostic_row()])
    assert short["continuous"]["gt_stalk_ratio"]["mIoU"]["count"] == 1
    assert short["continuous"]["gt_stalk_ratio"]["mIoU"]["pearson"] is None


def fixture_checkpoint(tmp_path):
    records = []
    for index, split in enumerate(["train", "val", "val", "val", "test"]):
        tag = f"plant{index}.ply"
        records.append(
            {
                "sample_id": f"id{index}",
                "path": tag,
                "split": split,
                "group_id": f"group{index}",
            }
        )
        if split != "val":
            continue  # Neither train nor test files exist.
        data = np.zeros(
            30,
            dtype=[(key, "f4") for key in ("x", "y", "z")]
            + [(key, "u1") for key in ("red", "green", "blue")],
        )
        xyz = np.random.default_rng(index).normal(size=(30, 3)).astype("float32")
        for column, key in enumerate(("x", "y", "z")):
            data[key] = xyz[:, column]
        data["green"] = np.where(xyz[:, 2] > 0, 0, 140)
        PlyData([PlyElement.describe(data, "vertex")], text=True).write(
            str(tmp_path / tag)
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records))
    _, digest = read_manifest(manifest)
    config = RunConfig(
        points=36,
        device="cpu",
        batch_size=2,
        model={"npoints": [8, 4, 2, 2], "nsamples": [4] * 4, "chunk_size": 4},
    )
    seed_everything(config.seed)
    model = PointNet2Segmentation(PointNet2Config(**config.model))
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint,
        {
            "mode": "regular",
            "config": asdict(config),
            "epoch": 76,
            "contract": experiment_contract(config, digest),
            "model_state": model.state_dict(),
        },
    )
    return manifest, checkpoint, records, config


def test_deterministic_analysis_matches_validation_and_preserves_contract(tmp_path):
    manifest, checkpoint, records, config = fixture_checkpoint(tmp_path)
    original = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    meta = tmp_path / "Metadata.xlsx"
    workbook(
        meta, [metadata_record("plant1.ply"), metadata_record("plant2.ply", Size="bad")]
    )
    results = []
    for name in ("a", "b"):
        results.append(
            analyze_checkpoint(
                str(manifest),
                str(tmp_path),
                str(checkpoint),
                str(tmp_path / name),
                str(meta),
                "cpu",
                save_predictions=True,
                visualize=True,
            )
        )
    a, b = results
    assert a["sample_count"] == 3 and a["checkpoint_epoch"] == 76
    assert a["split"] == "val" and a["contract"] == experiment_contract(
        config, read_manifest(manifest)[1]
    )
    assert (tmp_path / "a/per_sample_metrics.csv").read_bytes() == (
        tmp_path / "b/per_sample_metrics.csv"
    ).read_bytes()
    assert a["rankings"] == b["rankings"]
    validation = validate_checkpoint(
        str(manifest),
        str(tmp_path),
        str(checkpoint),
        str(tmp_path / "validation.json"),
        "cpu",
    )
    assert {k: v for k, v in a["metrics"].items() if k != "seconds"} == {
        k: v for k, v in validation["metrics"].items() if k != "seconds"
    }
    rows = list(csv.DictReader((tmp_path / "a/per_sample_metrics.csv").open()))
    assert (
        sum(int(row["evaluated_points"]) for row in rows)
        == a["metrics"]["evaluated_points"]
    )
    dataset = make_dataset(records, str(tmp_path), "val", config)
    for index in range(len(dataset)):
        sample = dataset[index]
        artifacts = a["artifacts"][sample["sample_id"]]
        payload = np.load(tmp_path / "a" / artifacts["predictions"], allow_pickle=False)
        for key in (
            "points",
            "labels",
            "valid_mask",
            "source_index",
            "centroid",
            "scale",
        ):
            np.testing.assert_array_equal(payload[key], sample[key].numpy())
        assert (tmp_path / "a" / artifacts["visualization"]).exists()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original
    assert (
        not (tmp_path / "plant0.ply").exists()
        and not (tmp_path / "plant4.ply").exists()
    )
    assert a["metadata_join"]["unmatched_validation_tags"] == ["plant3.ply"]


def test_validation_only_and_contract_failures(tmp_path):
    for split in ("train", "test"):
        with pytest.raises(ValueError, match="validation only"):
            analyze_checkpoint(
                "missing", "missing", "missing", tmp_path / "out", split=split
            )
    manifest, checkpoint, _, _ = fixture_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        analyze_checkpoint(manifest, tmp_path, checkpoint, tmp_path / "out", top_k=0)
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="contract"):
        analyze_checkpoint(manifest, tmp_path, checkpoint, tmp_path / "out")
    command = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_pointnet2.py",
            "--manifest",
            "none",
            "--root",
            "none",
            "--checkpoint",
            "none",
            "--output-dir",
            "none",
            "--split",
            "test",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert command.returncode == 2 and "invalid choice" in command.stderr


def test_visualization_error_categories_and_html(tmp_path):
    target = np.array([0, 1, 0, 1])
    prediction = np.array([0, 1, 1, 0])
    np.testing.assert_array_equal(error_categories(target, prediction), [0, 1, 2, 3])
    path = tmp_path / "view.html"
    save_views(path, "<plant>", np.arange(12).reshape(4, 3), target, prediction)
    content = path.read_text()
    for text in (
        "Ground truth",
        "Prediction",
        "Error map",
        "Correct leaf",
        "Correct stalk",
        "false positive",
        "false negative",
        "plotly_relayout",
    ):
        assert text in content
    assert "cdn.plot.ly" not in content[:1000]  # Embedded JS, no network requirement.


def test_streaming_visualization_retention_and_aggregate_confusion():
    class Echo(torch.nn.Module):
        def forward(self, points):
            return torch.stack([1 - points[..., 0], points[..., 0]], -1)

    batches = []
    for index in range(20):
        labels = torch.tensor([[0] * 10 + [1] * 10])
        pred = labels.clone()
        pred[0, :index] = 1 - pred[0, :index]
        points = torch.zeros(1, 20, 3)
        points[..., 0] = pred
        batches.append(
            {
                "sample_id": [str(index)],
                "points": points,
                "labels": labels,
                "valid_mask": torch.ones(1, 20, dtype=torch.bool),
                "source_index": torch.arange(20)[None],
                "centroid": torch.zeros(1, 3),
                "scale": torch.ones(1),
            }
        )
    rows, metrics, confusion, cases = evaluate_samples(
        Echo(), batches, torch.device("cpu"), True
    )
    assert set(cases) == set(select_visualization_cases(rows)) and len(cases) <= 13
    assert metrics["evaluated_points"] == 400 == confusion.sum()
