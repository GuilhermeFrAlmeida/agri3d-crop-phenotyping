"""Canonical .pt backend. Only this adapter imports the PointNet++ runner."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from agri3d.pipeline.provenance import sha256_file
from agri3d.training.checkpoints import load_checkpoint, seed_everything
from agri3d.training.pointnet2_runner import (
    RunConfig,
    build_model,
    experiment_contract,
    select_device,
    verify_checkpoint_contract,
)

from .predictor import SemanticPrediction, validate_xyz


class PyTorchPointNet2Predictor:
    def __init__(self, checkpoint, device="cpu", require_frozen_run1=True,
                 supplied_metadata=None):
        self.path = Path(checkpoint)
        checkpoint_hash = sha256_file(self.path)
        payload = load_checkpoint(self.path)
        if checkpoint_hash != sha256_file(self.path):
            raise ValueError("Checkpoint changed during load")
        if payload.get("mode") != "regular":
            raise ValueError("Semantic inference requires a regular-run checkpoint")
        self.config = RunConfig(**payload["config"])
        self.config.validate()
        contract = payload.get("contract", {})
        expected = experiment_contract(self.config, contract.get("manifest_sha256"))
        if not expected["manifest_sha256"]:
            raise ValueError("Checkpoint manifest provenance is missing")
        verify_checkpoint_contract(payload, expected, "Checkpoint contract mismatch")
        if contract["features"] != "xyz":
            raise ValueError("Frozen downstream backend requires XYZ-only Run 1")
        if require_frozen_run1:
            frozen = experiment_contract(RunConfig(), contract["manifest_sha256"])
            if contract != frozen or self.config.overfit_samples or self.config.epochs != 100:
                raise ValueError("Checkpoint differs from frozen Run 1 contract")
        if payload.get("selection_metric") not in (None, "val_mIoU"):
            raise ValueError("Checkpoint was not selected by validation mIoU")
        self.device = select_device(device)
        seed_everything(self.config.seed)
        self.model = build_model(self.config).to(self.device).eval()
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.input_points = self.config.points
        epoch = payload.get("epoch")
        record = next((r for r in payload.get("history", [])
                       if r.get("epoch") == epoch), {})
        # A last.pt best_miou can describe earlier weights: never attach it to current weights.
        val = record.get("val", {}).get("mIoU")
        if (require_frozen_run1 and val is not None and payload.get("best_miou") is not None
                and val != payload["best_miou"]):
            raise ValueError("Current weights are not the recorded selected checkpoint; use best.pt")
        self.metadata = {
            "model_family": "pointnet2_ssg", "feature_mode": "xyz",
            "experiment": payload.get("experiment"),
            "checkpoint_sha256": checkpoint_hash,
            "input_points": self.input_points, "classes": contract["classes"],
            "manifest_sha256": contract["manifest_sha256"],
            "selection_metric": payload.get("selection_metric"),
            "selected_weights_verified": val is not None and val == payload.get("best_miou"),
            "validation_mIoU": val, "weights_epoch": epoch,
            "checkpoint_contract": contract,
            "checkpoint_metadata": {"environment": payload.get("environment"),
                                    "config": payload["config"]},
            "repository_metadata": {"frozen_run1_policy_enforced": require_frozen_run1,
                                    "compatible_experiment_policy": "run1_xyz" if require_frozen_run1 else None},
            "user_supplied_metadata": supplied_metadata or {},
            "backend": "pytorch", "device": str(self.device),
        }
        self.saved_config = asdict(self.config)

    def predict(self, xyz, features=None):
        xyz = validate_xyz(xyz)
        if xyz.shape != (self.input_points, 3) or xyz.dtype != np.float32:
            raise ValueError("Input must match checkpoint points and float32 XYZ")
        if features is not None:
            raise ValueError("RGB/extra features forbidden by frozen XYZ backend")
        device_timing = {}
        with torch.inference_mode():
            tensor = torch.from_numpy(xyz.copy())[None].to(self.device)
            if self.device.type == "cuda":
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
            logits = self.model(tensor)[0]
            probabilities = logits.softmax(-1)
            if self.device.type == "cuda":
                end.record()
                end.synchronize()
                device_timing["cuda_forward_softmax"] = float(begin.elapsed_time(end))
            probabilities = probabilities.cpu().numpy()
            logits = logits.cpu().numpy()
        return SemanticPrediction(logits.argmax(1), logits, probabilities,
                                  np.arange(len(xyz)), self.metadata, device_timing).validate(len(xyz))
