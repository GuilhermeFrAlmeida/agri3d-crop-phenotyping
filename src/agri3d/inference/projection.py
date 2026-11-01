"""One frozen projection policy: exact metric-space nearest sampled neighbor."""
import numpy as np
from scipy.spatial import cKDTree

from agri3d.preprocessing.labels import validate_labels

from .predictor import validate_xyz


def nearest_indices(reference, queries):
    """Nearest row with exact squared-distance/source-row tie breaking."""
    reference, queries = validate_xyz(reference), validate_xyz(queries)
    reference = reference.astype(np.float64)
    queries = queries.astype(np.float64)
    queries, inverse = np.unique(queries, axis=0, return_inverse=True)
    tree = cKDTree(reference)
    distance, indices = tree.query(queries, k=min(2, len(reference)), workers=1)
    if len(reference) == 1:
        return np.zeros(len(inverse), dtype=np.int64)
    result = indices[:, 0].copy()
    tied = np.flatnonzero(np.isclose(distance[:, 0], distance[:, 1], rtol=1e-12, atol=0))
    for i in tied:
        radius = np.nextafter(distance[i, 1] * (1 + 1e-12), np.inf)
        candidates = np.asarray(tree.query_ball_point(queries[i], radius), dtype=np.int64)
        d2 = ((reference[candidates] - queries[i]) ** 2).sum(1)
        result[i] = candidates[np.lexsort((candidates, d2))[0]]
    return result[inverse].astype(np.int64)


def project_nearest(full_xyz, sample_xyz, labels, clean_indices, probabilities=None):
    full_xyz, sample_xyz = validate_xyz(full_xyz), validate_xyz(sample_xyz)
    labels = validate_labels(labels)
    ids = np.asarray(clean_indices)
    if ids.shape != labels.shape or len(ids) != len(sample_xyz) or ids.dtype.kind not in "iu":
        raise ValueError("Sample indices/labels do not align")
    if np.any(ids < 0) or np.any(ids >= len(full_xyz)) or len(np.unique(ids)) != len(ids):
        raise ValueError("Require unique valid cleaned-cloud sample indices")
    if not np.array_equal(full_xyz[ids], sample_xyz):
        raise ValueError("Sample XYZ/source-index alignment mismatch")
    nearest = nearest_indices(sample_xyz, full_xyz)
    # Coincident source rows can have different predictions; index identity takes precedence.
    nearest[ids] = np.arange(len(ids))
    projected = labels[nearest]
    full_probabilities = None
    if probabilities is not None:
        p = np.asarray(probabilities)
        if (p.shape != (len(ids), 2) or not np.isfinite(p).all()
                or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(1), 1)):
            raise ValueError("Invalid sampled probabilities")
        full_probabilities = p[nearest]
    return projected, full_probabilities
