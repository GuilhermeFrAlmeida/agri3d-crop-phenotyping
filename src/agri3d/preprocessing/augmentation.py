"""Mild XYZ augmentation in normalized space, after fixed-size sampling.

Rotation is about physical Z only. Point order/attributes are never changed.
An explicit Generator lets the Dataset reuse its seed/epoch/sample RNG without
consuming randomness before sampling. No global RNG or torch RNG is used.
"""

import math
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from agri3d.io.ply import PointCloudData


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = False
    rotate_z: bool = True
    scale_min: float = 0.95
    scale_max: float = 1.05
    jitter_sigma: float = 0.005
    jitter_clip: float = 0.02

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.rotate_z) is not bool:
            raise TypeError("enabled and rotate_z must be booleans")
        values = (self.scale_min, self.scale_max, self.jitter_sigma, self.jitter_clip)
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in values
        ):
            raise ValueError("Augmentation parameters must be finite numbers")
        if not 0 < self.scale_min <= self.scale_max:
            raise ValueError("Require 0 < scale_min <= scale_max")
        if self.jitter_sigma < 0 or self.jitter_clip < 0:
            raise ValueError("Jitter sigma/clip must be nonnegative")

    @classmethod
    def from_settings(cls, settings=None):
        if isinstance(settings, cls):
            return settings
        if settings is None:
            return cls()
        if not isinstance(settings, dict):
            raise TypeError("augmentation must be a mapping")
        return cls(**settings)

    def contract(self):
        # Preserve old Run 1 contracts exactly; disabled latent parameters are inactive.
        if not self.enabled:
            return None
        return {
            **asdict(self),
            **{
                name: float(getattr(self, name))
                for name in ("scale_min", "scale_max", "jitter_sigma", "jitter_clip")
            },
            "policy_version": 1,
            "rotation_z_range_radians": [0.0, 2 * math.pi] if self.rotate_z else None,
            "placement": "after_normalization_and_sampling; training_only",
            "jitter_duplicates": "shared_by_source_index",
        }


def augment_xyz(xyz, rng, config, source_index=None):
    """Return new float32 XYZ and applied transform metadata; never mutate inputs.

    Repeated padded points share the jitter of their source point, preserving
    repeated coordinates. Jitter is clipped per coordinate, not by vector norm.
    Centroid/scale normalization metadata still describes the original cloud.
    """
    config = AugmentationConfig.from_settings(config)
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError("Expected nonempty (N,3) XYZ")
    if xyz.dtype != np.float32:
        raise TypeError("Augmentation expects float32 XYZ")
    if not np.isfinite(xyz).all():
        raise ValueError("Augmentation requires finite XYZ")
    result = xyz.copy()
    if not config.enabled:
        return result, None
    if not isinstance(rng, np.random.Generator):
        raise TypeError("Augmentation requires an explicit numpy Generator")
    angle = float(rng.uniform(0, 2 * math.pi)) if config.rotate_z else 0.0
    if config.rotate_z:
        c, s = np.float32(math.cos(angle)), np.float32(math.sin(angle))
        result[:, :2] = result[:, :2] @ np.array([[c, s], [-s, c]], dtype=np.float32)
    factor = float(rng.uniform(config.scale_min, config.scale_max))
    result *= np.float32(factor)
    if config.jitter_sigma and config.jitter_clip:
        inverse, count = None, len(result)
        if source_index is not None:
            source_index = np.asarray(source_index)
            if (
                source_index.shape != (len(result),)
                or source_index.dtype.kind not in "iu"
            ):
                raise ValueError("source_index must be aligned integer identities")
            unique, inverse = np.unique(source_index, return_inverse=True)
            count = len(unique)
        noise = np.clip(
            rng.normal(0, config.jitter_sigma, (count, 3)),
            -config.jitter_clip,
            config.jitter_clip,
        ).astype(np.float32)
        result += noise if inverse is None else noise[inverse]
    if not np.isfinite(result).all():
        raise ValueError("Augmentation produced nonfinite XYZ")
    return result, {
        "rotation_z_radians": angle,
        "isotropic_scale": factor,
        **config.contract(),
    }


def augment_cloud(pcd: PointCloudData, rng, config) -> PointCloudData:
    """Deep-copy attributes/metadata and augment XYZ only (including padding)."""
    pcd.validate(finite=True)
    xyz, applied = augment_xyz(pcd.xyz, rng, config, pcd.attributes.get("source_index"))
    result = PointCloudData(
        xyz, {k: v.copy() for k, v in pcd.attributes.items()}, deepcopy(pcd.metadata)
    )
    if applied is not None:
        result.metadata["training_augmentation"] = applied
    return result.validate(finite=True)
