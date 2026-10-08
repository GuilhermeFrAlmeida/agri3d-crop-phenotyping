"""One train/validation epoch, independent of datasets and checkpoint policy."""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluate import SegmentationEvaluator
from .losses import masked_cross_entropy


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    evaluator = SegmentationEvaluator(num_classes=2)
    loss_sum, point_count, batches = 0.0, 0, 0
    start = time.perf_counter()
    with torch.set_grad_enabled(training):
        for batch in loader:
            points = batch["points"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["valid_mask"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(points)
            loss = masked_cross_entropy(logits, labels, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite segmentation loss")
            if training:
                loss.backward()
                # Detect a broken backward pass instead of saving corrupt weights.
                finite_gradients = torch.stack(
                    [
                        torch.isfinite(p.grad).all()
                        for p in model.parameters()
                        if p.grad is not None
                    ]
                ).all()
                if not finite_gradients:
                    raise FloatingPointError("Nonfinite model gradients")
                optimizer.step()
            n_valid = int(mask.sum().item())
            loss_sum += float(loss.detach().item()) * n_valid
            point_count += n_valid
            batches += 1
            evaluator.update(
                logits.detach().argmax(-1).cpu().numpy(),
                labels.cpu().numpy(),
                mask.cpu().numpy(),
            )
    if not point_count:
        raise ValueError("Epoch has no valid points")
    return {
        "loss": loss_sum / point_count,
        **evaluator.compute(),
        "batches": batches,
        "seconds": time.perf_counter() - start,
    }
