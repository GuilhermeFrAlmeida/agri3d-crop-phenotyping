"""Unweighted semantic cross-entropy on real (non-padding) points only."""

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_cross_entropy(logits: Tensor, targets: Tensor, valid_mask: Tensor) -> Tensor:
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError("logits must have shape (B,N,2)")
    if targets.shape != logits.shape[:2] or valid_mask.shape != targets.shape:
        raise ValueError("targets/mask must match logits shape (B,N)")
    if targets.dtype != torch.long or valid_mask.dtype != torch.bool:
        raise TypeError("targets must be int64 and valid_mask must be bool")
    if targets.device != logits.device or valid_mask.device != logits.device:
        raise ValueError("logits, targets and valid_mask must be on the same device")
    if not valid_mask.any():
        raise ValueError("No valid points in batch")
    labels = targets[valid_mask]
    if torch.any((labels < 0) | (labels > 1)):
        raise ValueError("Valid labels must be 0=leaf or 1=stalk")
    # Slice before CE so arbitrary sentinel values on padding are never evaluated.
    return F.cross_entropy(logits[valid_mask], labels)
