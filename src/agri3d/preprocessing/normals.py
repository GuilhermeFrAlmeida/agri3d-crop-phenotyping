"""Full-cloud kNN PCA normals: geometry only, explicit ties/sign/degeneracy."""

import hashlib
import inspect
from dataclasses import asdict, dataclass, replace

import numpy as np
import scipy
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class NormalConfig:
    source: str = "estimated"
    method: str = "knn_pca"
    k: int = 32
    orientation: str = "largest_abs_component_positive"
    policy_version: int = 1

    @classmethod
    def from_settings(cls, settings=None):
        if settings is None:
            settings = {}
        if not isinstance(settings, dict):
            raise TypeError("normals must be a mapping")
        cfg = cls(**settings)
        if cfg.orientation == "deterministic":
            cfg = replace(cfg, orientation="largest_abs_component_positive")
        if (cfg.source, cfg.method, cfg.orientation) != (
            "estimated",
            "knn_pca",
            "largest_abs_component_positive",
        ):
            raise ValueError(
                "Require estimated knn_pca normals with canonical deterministic orientation"
            )
        if type(cfg.k) is not int or cfg.k < 3:
            raise ValueError("Normal k must be an integer >=3")
        if type(cfg.policy_version) is not int or cfg.policy_version != 1:
            raise ValueError("Unsupported normal policy_version")
        return cfg

    def contract(self):
        return {
            **asdict(self),
            "neighbors": "include_self_then_distance_source_row_ties",
            "covariance": "centered_float64_divide_by_effective_k",
            "eigen_gap_rtol": 1e-8,
            "degenerate_fallback": [0.0, 0.0, 1.0],
        }


def canonical_orientation(normals):
    """Dominant absolute component positive; exact magnitude ties favor X/Y/Z."""
    result = np.array(normals, dtype=np.float64, copy=True)
    axis = np.argmax(np.abs(result), axis=1)
    sign = np.where(result[np.arange(len(result)), axis] < 0, -1.0, 1.0)
    return result * sign[:, None]


def estimate_normals(xyz, config=None, chunk_size=2048):
    """k includes self, capped by N. Row identity is the cleaned source order.

    cKDTree uses eps=0 and one worker. Sort neighbors by exact float64 squared
    distance, then source row; expand cutoff ties before selecting neighbors.
    An explicit self point is always included, even among duplicate coordinates.
    No color, label, global fit, sampling RNG, or epoch is accepted by this API.
    """
    config = NormalConfig.from_settings() if config is None else config
    if not isinstance(config, NormalConfig):
        config = NormalConfig.from_settings(config)
    config = NormalConfig.from_settings(asdict(config))
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Normal estimation requires finite nonempty (N,3) XYZ")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    n, k = len(xyz), min(config.k, len(xyz))
    tree = cKDTree(xyz)
    normals = np.empty((n, 3), dtype=np.float32)
    degenerate = np.empty(n, dtype=bool)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        rows = np.arange(start, stop)
        if k == 1:
            neighbors = rows[:, None]
        else:
            _, candidates = tree.query(xyz[rows], k=min(n, k + 1), eps=0, workers=1)
            d2 = ((xyz[candidates] - xyz[rows, None]) ** 2).sum(-1)
            d2[candidates == rows[:, None]] = np.inf
            order = np.lexsort((candidates, d2), axis=1)
            candidates = np.take_along_axis(candidates, order, axis=1)
            d2 = np.take_along_axis(d2, order, axis=1)
            other = candidates[:, : k - 1].copy()
            # Only ambiguous cutoff rows require a radius expansion, usually rare.
            if k < n:
                tied = np.isclose(d2[:, k - 2], d2[:, k - 1], rtol=1e-12, atol=0)
                for offset in np.flatnonzero(tied):
                    radius = np.nextafter(
                        np.sqrt(d2[offset, k - 2]) * (1 + 1e-12), np.inf
                    )
                    all_indices = np.asarray(
                        tree.query_ball_point(
                            xyz[rows[offset]], radius, eps=0, workers=1
                        ),
                        dtype=np.int64,
                    )
                    all_indices = all_indices[all_indices != rows[offset]]
                    distances = ((xyz[all_indices] - xyz[rows[offset]]) ** 2).sum(-1)
                    sorted_indices = all_indices[np.lexsort((all_indices, distances))]
                    if len(sorted_indices) < k - 1:
                        raise ValueError(
                            "Insufficient neighbors after deterministic tie expansion"
                        )
                    other[offset] = sorted_indices[: k - 1]
            neighbors = np.concatenate([rows[:, None], other], axis=1)
        local = xyz[neighbors]
        centered = local - local.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / k
        values, vectors = np.linalg.eigh(covariance)
        largest = np.maximum(values[:, 2], 0)
        bad = (largest <= 0) | ((values[:, 1] - values[:, 0]) <= 1e-8 * largest)
        oriented = canonical_orientation(vectors[:, :, 0])
        oriented /= np.linalg.norm(oriented, axis=1, keepdims=True)
        oriented[bad] = (0, 0, 1)
        normals[start:stop] = oriented.astype(np.float32)
        degenerate[start:stop] = bad
    validate_normals(normals, n)
    return normals, degenerate


def validate_normals(normals, point_count):
    normals = np.asarray(normals)
    if normals.shape != (point_count, 3) or not np.isfinite(normals).all():
        raise ValueError("Normals must have finite aligned shape (N,3)")
    if not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Normals must be unit direction vectors")


def normal_implementation_fingerprint():
    return hashlib.sha256(
        inspect.getsource(inspect.getmodule(estimate_normals)).encode()
    ).hexdigest()


def normal_provenance(config):
    return {
        "policy": config.contract(),
        "implementation_sha256": normal_implementation_fingerprint(),
        "scipy_version": scipy.__version__,
    }
