"""Read-only, validation-only diagnostics for a saved regular-run checkpoint."""

import csv
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from agri3d.models import PointNet2Config, PointNet2Segmentation
from agri3d.training.checkpoints import load_checkpoint, seed_everything
from agri3d.training.evaluate import SegmentationEvaluator
from agri3d.training.losses import masked_cross_entropy
from agri3d.training.pointnet2_runner import (
    RunConfig,
    environment,
    experiment_contract,
    make_dataset,
    make_loader,
    read_manifest,
    select_device,
    verify_checkpoint_contract,
    write_json,
)

from .metadata import conditioned_summary, join_metadata, read_metadata
from .metrics import metric_summary, sample_metrics, select_visualization_cases


def evaluate_samples(model, loader, device, keep_cases=False):
    """One inference pass; aggregate confusion reproduces dataset-wide metrics.

    Optionally retain only the current 3/5/5 ranking candidates (at most 13 plants),
    so visualization never requires resampling or a second inference pass.
    """
    model.eval()
    evaluator = SegmentationEvaluator(2)
    rows, cases = [], {}
    ids = set()
    loss_sum, count, batches = 0.0, 0, 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            targets = batch["labels"].to(device)
            mask = batch["valid_mask"].to(device)
            logits = model(points)
            loss = masked_cross_entropy(logits, targets, mask)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite validation logits/loss")
            valid = int(mask.sum().item())
            loss_sum += float(loss.item()) * valid
            count += valid
            batches += 1
            predictions = logits.argmax(-1).cpu().numpy()
            labels, masks = targets.cpu().numpy(), mask.cpu().numpy()
            evaluator.update(predictions, labels, masks)
            for index, sample_id in enumerate(batch["sample_id"]):
                if sample_id in ids:
                    raise ValueError(f"Duplicate evaluated sample_id: {sample_id}")
                ids.add(sample_id)
                row, _ = sample_metrics(
                    sample_id, predictions[index], labels[index], masks[index]
                )
                rows.append(row)
                if keep_cases:
                    cases[sample_id] = {
                        "points": batch["points"][index].cpu().numpy().copy(),
                        "labels": labels[index].copy(),
                        "predictions": predictions[index].copy(),
                        "valid_mask": masks[index].copy(),
                        "source_index": batch["source_index"][index]
                        .cpu()
                        .numpy()
                        .copy(),
                        "centroid": batch["centroid"][index].cpu().numpy().copy(),
                        "scale": batch["scale"][index].cpu().numpy().copy(),
                    }
                    chosen = select_visualization_cases(rows)
                    cases = {
                        key: value for key, value in cases.items() if key in chosen
                    }
    if not count:
        raise ValueError("No evaluated validation points")
    metrics = {
        "loss": loss_sum / count,
        **evaluator.compute(),
        "batches": batches,
        "seconds": time.perf_counter() - start,
    }
    return rows, metrics, evaluator.confusion, cases


def artifact_name(sample_id):
    """Stable filesystem-safe key; exact IDs are retained in summary.json."""
    return hashlib.sha256(sample_id.encode()).hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_checkpoint(
    manifest,
    root,
    checkpoint_path,
    output_dir,
    metadata_path=None,
    device_name="auto",
    num_workers=0,
    top_k=5,
    visualize=False,
    save_predictions=False,
    split="val",
):
    """Use the checkpoint's original contract; no CLI override of model/data choices."""
    if split != "val":
        raise ValueError(
            "Analysis permits frozen validation only; train/test evaluation is rejected"
        )
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Analysis output directory must be new or empty")
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("mode") != "regular":
        raise ValueError("Analysis requires a regular-run checkpoint")
    config = RunConfig(**checkpoint["config"])
    config.device, config.num_workers = device_name, num_workers
    config.validate()
    seed_everything(config.seed)
    records, manifest_hash = read_manifest(manifest)
    verify_checkpoint_contract(
        checkpoint,
        experiment_contract(config, manifest_hash),
        "Analysis manifest/config differs from checkpoint contract",
    )
    device = select_device(device_name)
    dataset = make_dataset(records, root, "val", config)
    # Preflight exact join/dependencies before inference. Never check train/test paths.
    filenames = [
        {"sample_id": r["sample_id"], "Tagname": Path(r["path"]).name}
        for r in dataset.records
    ]
    metadata = read_metadata(metadata_path) if metadata_path else None
    if metadata is not None:
        join_metadata(filenames, metadata)
    if visualize:
        try:
            import plotly  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Visualization requires pip install 'agri3d[analysis]'"
            ) from exc
    loader = make_loader(
        dataset, config, False, torch.Generator().manual_seed(config.seed + 1)
    )
    model = PointNet2Segmentation(PointNet2Config(**config.model)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    rows, metrics, confusion, cases = evaluate_samples(
        model, loader, device, visualize or save_predictions
    )
    tags = {r["sample_id"]: r["Tagname"] for r in filenames}
    for row in rows:
        row["Tagname"] = tags[row["sample_id"]]
    join_info = None
    if metadata is not None:
        rows, join_info = join_metadata(rows, metadata)
    associations = conditioned_summary(rows, metadata is not None)
    summary = {
        "split": "val",
        "scope": "fixed deterministic sampled points; not full-cloud projection",
        "checkpoint_sha256": hashlib.sha256(
            Path(checkpoint_path).read_bytes()
        ).hexdigest(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "contract": checkpoint["contract"],
        "environment": environment(device),
        "metrics": metrics,
        "confusion_matrix": confusion.tolist(),
        "confusion_convention": "rows=GT, columns=prediction; 0=leaf, 1=stalk",
        **metric_summary(rows, top_k),
        "metadata_join": join_info,
        "metadata_sha256": hashlib.sha256(Path(metadata_path).read_bytes()).hexdigest()
        if metadata_path
        else None,
        "conditioned_summaries": associations,
        "artifacts": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_sample_metrics.csv", rows)
    for sample_id, reasons in summary["visualization_cases"].items():
        if not (visualize or save_predictions):
            continue
        payload = cases[sample_id]
        name = artifact_name(sample_id)
        artifacts = {"selection_reasons": reasons}
        if save_predictions:
            destination = output / "predictions" / f"{name}.npz"
            destination.parent.mkdir(exist_ok=True)
            np.savez_compressed(destination, sample_id=np.asarray(sample_id), **payload)
            artifacts["predictions"] = str(destination.relative_to(output))
        if visualize:
            from .visualization import save_views

            destination = output / "visualizations" / f"{name}.html"
            mask = payload["valid_mask"]
            save_views(
                destination,
                sample_id,
                payload["points"][mask],
                payload["labels"][mask],
                payload["predictions"][mask],
            )
            artifacts["visualization"] = str(destination.relative_to(output))
        summary["artifacts"][sample_id] = artifacts
    write_json(output / "summary.json", summary)
    return summary
