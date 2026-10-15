"""One train/validation epoch, independent of datasets and checkpoint policy."""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from agri3d.performance.timing import BatchTimings

from .evaluate import SegmentationEvaluator
from .losses import masked_cross_entropy


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    non_blocking: bool = False,
    detailed_timing: bool = False,
    max_batches: int | None = None,
    step_callback=None,
) -> dict:
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError("max_batches must be positive")
    training = optimizer is not None
    model.train(training)
    evaluator = SegmentationEvaluator(num_classes=2)
    loss_sum, point_count, batches = 0.0, 0, 0
    timer = BatchTimings(device, detailed_timing)
    start = time.perf_counter()
    iterator = None
    with torch.set_grad_enabled(training):
        while max_batches is None or batches < max_batches:
            timer.begin()
            try:
                with timer.stage("data_wait"):
                    if iterator is None:
                        iterator = iter(loader)
                    batch = next(iterator)
            except StopIteration:
                break
            with timer.stage("h2d"):
                points = batch["points"].to(device, non_blocking=non_blocking)
                labels = batch["labels"].to(device, non_blocking=non_blocking)
                mask = batch["valid_mask"].to(device, non_blocking=non_blocking)
            with timer.stage("batch_preparation"):
                if training:
                    optimizer.zero_grad(set_to_none=True)
            with timer.stage("forward"):
                logits = model(points)
            with timer.stage("loss"):
                loss = masked_cross_entropy(logits, labels, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite segmentation loss")
            if training:
                with timer.stage("backward"):
                    loss.backward()
                with timer.stage("gradient_check"):
                    finite_gradients = torch.stack(
                        [
                            torch.isfinite(p.grad).all()
                            for p in model.parameters()
                            if p.grad is not None
                        ]
                    ).all()
                    if not finite_gradients:
                        raise FloatingPointError("Nonfinite model gradients")
                with timer.stage("optimizer"):
                    optimizer.step()
            with timer.stage("metrics"):
                n_valid = int(mask.sum().item())
                loss_sum += float(loss.detach().item()) * n_valid
                point_count += n_valid
                batches += 1
                evaluator.update(
                    logits.detach().argmax(-1).cpu().numpy(),
                    labels.cpu().numpy(),
                    mask.cpu().numpy(),
                )
            timer.finish(batch)
            if step_callback is not None:
                step_callback()
    if not point_count:
        raise ValueError("Epoch has no valid points")
    result = {
        "loss": loss_sum / point_count,
        **evaluator.compute(),
        "batches": batches,
        "seconds": time.perf_counter() - start,
    }
    if detailed_timing:
        result["timing"] = timer.summary()
    return result
