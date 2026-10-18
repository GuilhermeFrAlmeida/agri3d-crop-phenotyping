"""CPU-only checkpoint inspection: no dataset/model imports or training."""

import hashlib
import json
from pathlib import Path

from agri3d.training.checkpoints import load_checkpoint


def inspect_checkpoint(path, recent=10):
    if recent < 1:
        raise ValueError("recent must be positive")
    checkpoint = load_checkpoint(path)
    mode = checkpoint.get("mode")
    field = "val" if mode == "regular" else "train_eval"
    history = checkpoint.get("history", [])
    trajectory = [
        {"epoch": row.get("epoch"), **row[field]} for row in history if field in row
    ]
    best, best_value, best_loss = None, -1.0, float("inf")
    for row in trajectory:
        value, loss = row.get("mIoU"), row.get("loss")
        if value is None or loss is None:
            continue
        if value > best_value or (value == best_value and loss < best_loss):
            best, best_value, best_loss = row.get("epoch"), value, loss
    saved_best = checkpoint.get("best_miou")
    if (saved_best is not None and best_value != saved_best) or (
        checkpoint.get("best_loss") is not None and best_loss != checkpoint["best_loss"]
    ):
        best = None  # Incomplete history cannot establish the actual selected epoch.
    contract = checkpoint.get("contract", {})
    computed = hashlib.sha256(
        json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    saved_hash = checkpoint.get("contract_sha256")
    epoch = checkpoint.get("epoch")
    config = checkpoint.get("config", {})
    groups = checkpoint.get("optimizer_state", {}).get("param_groups", [])
    return {
        "checkpoint_path": str(Path(path).resolve()),
        "mode": mode,
        "completed_epoch": epoch,
        "configured_total_epochs": config.get("epochs"),
        "next_completed_epoch": epoch + 1 if isinstance(epoch, int) else None,
        "best_validation_mIoU": saved_best if mode == "regular" else None,
        "best_validation_loss": checkpoint.get("best_loss")
        if mode == "regular"
        else None,
        "best_loss_note": "Loss at selected mIoU checkpoint, not necessarily minimum loss across epochs",
        "selected_epoch": best,
        "best_epoch": best,
        "checkpoint_weights_epoch": epoch,
        "last_epoch": trajectory[-1]["epoch"] if trajectory else None,
        "epochs_since_last_improvement": epoch - best
        if isinstance(epoch, int) and best is not None
        else None,
        "latest_epoch_metrics": history[-1] if history else None,
        "recent_validation_trajectory": trajectory[-recent:]
        if mode == "regular"
        else [],
        "scheduler_contract": contract.get("scheduler"),
        "scheduler_state": checkpoint.get("scheduler_state"),
        "lr_reduction_epochs": [
            row["epoch"] for row in history if row.get("lr_changed")
        ],
        "lr_at_best_epoch": next(
            (row.get("lr") for row in history if row.get("epoch") == best), None
        ),
        "augmentation_contract": contract.get("augmentation"),
        "manifest_sha256": contract.get("manifest_sha256"),
        "model_configuration": contract.get("model", config.get("model")),
        "optimizer": contract.get("optimizer"),
        "optimizer_group_lrs": [group.get("lr") for group in groups],
        "batch_size": contract.get("batch_size", config.get("batch_size")),
        "training_environment": checkpoint.get("environment"),
        "runtime": checkpoint.get("runtime"),
        "contract_sha256": saved_hash,
        "computed_contract_sha256": computed,
        "contract_hash_valid": saved_hash == computed
        if saved_hash is not None
        else None,
        "recovery_state_present": {
            key: key in checkpoint
            for key in (
                "model_state",
                "optimizer_state",
                "rng",
                "loader_rng",
                "history",
                "contract",
                "best_miou",
                "best_loss",
            )
        },
    }
