"""Semantic backend boundary; no network-specific types cross this interface."""
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from agri3d.preprocessing.labels import validate_labels


def validate_xyz(xyz):
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError("Expected nonempty (N,3) XYZ")
    if not np.isfinite(xyz).all():
        raise ValueError("XYZ must be finite")
    return xyz


@dataclass
class SemanticPrediction:
    labels: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    source_indices: np.ndarray
    model_metadata: dict
    timing_ms: dict = field(default_factory=dict)

    def validate(self, count):
        self.labels = validate_labels(self.labels)
        if self.labels.shape != (count,):
            raise ValueError("Prediction label count differs from input")
        for name in ("logits", "probabilities"):
            a = np.asarray(getattr(self, name))
            setattr(self, name, a)
            if a.shape != (count, 2) or not np.isfinite(a).all():
                raise ValueError(f"Invalid {name}")
        if (np.any(self.probabilities < 0) or np.any(self.probabilities > 1)
                or not np.allclose(self.probabilities.sum(1), 1, atol=1e-6)):
            raise ValueError("Invalid probabilities")
        if not np.array_equal(self.labels, self.logits.argmax(1)):
            raise ValueError("Labels must be logit argmax")
        if not np.array_equal(self.probabilities.argmax(1), self.labels):
            raise ValueError("Probability labels differ")
        indices = np.asarray(self.source_indices)
        if indices.shape != (count,) or indices.dtype.kind not in "iu":
            raise ValueError("Invalid source indices")
        # Backend indices address the input supplied to predict, not a PLY file.
        if not np.array_equal(indices, np.arange(count)):
            raise ValueError("Backend must preserve input order")
        return self


class SemanticPredictor(Protocol):
    metadata: dict
    input_points: int

    def predict(self, xyz: np.ndarray, features=None) -> SemanticPrediction:
        """Predict normalized XYZ, preserving the supplied input order."""
        ...
