"""Pure PyTorch geometry. Public tensors use (batch, points, channels).

Searches are non-differentiable; feature gathering/interpolation propagates
feature gradients. No custom extensions or full input-resolution N x N search.
"""

import math

import torch
from torch import Tensor

from agri3d.performance.ranges import profile_range


def check_xyz(xyz: Tensor, name: str = "xyz") -> None:
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or min(xyz.shape[:2]) < 1:
        raise ValueError(f"{name} must have nonempty shape (B, N, 3)")
    if not xyz.is_floating_point():
        raise TypeError(f"{name} must be floating point")


def check_pair(source: Tensor, target: Tensor) -> None:
    check_xyz(source, "source")
    check_xyz(target, "target")
    if source.shape[0] != target.shape[0]:
        raise ValueError("Point sets must have the same batch size")
    if source.device != target.device or source.dtype != target.dtype:
        raise ValueError("Point sets must have the same device and dtype")


@profile_range("agri3d::square_distance")
def square_distance(source: Tensor, target: Tensor) -> Tensor:
    """Squared Euclidean distances, (B, S, 3) x (B, N, 3) -> (B, S, N).

    Callers bound S with a query chunk. clamp removes small negative roundoff.
    """
    check_pair(source, target)
    return (
        source.square().sum(-1, keepdim=True)
        + target.square().sum(-1).unsqueeze(1)
        - 2 * torch.bmm(source, target.transpose(1, 2))
    ).clamp_min(0)


@profile_range("agri3d::index_points")
def index_points(points: Tensor, indices: Tensor) -> Tensor:
    """Gather (B, N, C) with (B, ...) int64 indices -> (B, ..., C)."""
    if points.ndim != 3 or min(points.shape) < 1:
        raise ValueError("points must have nonempty shape (B, N, C)")
    if indices.ndim < 2 or indices.shape[0] != points.shape[0]:
        raise ValueError("indices must have shape (B, ...) matching points")
    if indices.dtype != torch.long or indices.device != points.device:
        raise ValueError("indices must be int64 on the points device")
    if (
        indices.numel() == 0
        or torch.any(indices < 0)
        or torch.any(indices >= points.shape[1])
    ):
        raise ValueError("Point indices must be nonempty and within [0, N)")
    shape = indices.shape
    gathered = points.gather(
        1, indices.reshape(shape[0], -1, 1).expand(-1, -1, points.shape[2])
    )
    return gathered.reshape(*shape, points.shape[2])


@torch.no_grad()
@profile_range("agri3d::fps")
def farthest_point_sample(xyz: Tensor, npoint: int) -> Tensor:
    """Deterministic unique-index FPS, starting farthest from the cloud centroid.

    Returns (B, min(npoint, N)). Ties select the first remaining index, even
    for coincident points. O(B*N*npoint) compute and O(B*N) working memory.
    """
    check_xyz(xyz)
    if not isinstance(npoint, int) or npoint < 1:
        raise ValueError("npoint must be a positive integer")
    batch, count, _ = xyz.shape
    npoint = min(npoint, count)
    distances = torch.full(
        (batch, count), float("inf"), device=xyz.device, dtype=xyz.dtype
    )
    selected = torch.zeros((batch, count), device=xyz.device, dtype=torch.bool)
    indices = torch.empty((batch, npoint), device=xyz.device, dtype=torch.long)
    farthest = (xyz - xyz.mean(1, keepdim=True)).square().sum(-1).argmax(-1)
    rows = torch.arange(batch, device=xyz.device)
    for step in range(npoint):
        indices[:, step] = farthest
        selected[rows, farthest] = True
        center = xyz[rows, farthest].unsqueeze(1)
        distances = torch.minimum(distances, (xyz - center).square().sum(-1))
        distances.masked_fill_(selected, -1)
        farthest = distances.argmax(-1)
    return indices


@torch.no_grad()
@profile_range("agri3d::radius_query")
def query_ball_point(
    radius: float, nsample: int, xyz: Tensor, centers: Tensor, chunk_size: int = 128
) -> Tensor:
    """Nearest K points within radius, (B, S, K); repeat nearest for short balls.

    If an external query has an empty ball, use its nearest point. SA queries
    are actual centroids and include themselves. Radius is in normalized units.
    """
    check_pair(xyz, centers)
    if not math.isfinite(radius) or radius <= 0 or nsample < 1 or chunk_size < 1:
        raise ValueError("radius, nsample and chunk_size must be positive")
    k = min(nsample, xyz.shape[1])
    chunks = []
    for start in range(0, centers.shape[1], chunk_size):
        distance = square_distance(centers[:, start : start + chunk_size], xyz)
        values, indices = distance.topk(k, dim=-1, largest=False, sorted=True)
        indices = torch.where(values <= radius * radius, indices, indices[..., :1])
        if k < nsample:
            indices = torch.cat(
                [indices, indices[..., :1].expand(-1, -1, nsample - k)], -1
            )
        chunks.append(indices)
    return torch.cat(chunks, dim=1)


@profile_range("agri3d::grouping")
def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: Tensor,
    features: Tensor | None = None,
    chunk_size: int = 128,
) -> tuple[Tensor, Tensor]:
    """Return centroids (B,S,3) and relative XYZ + features (B,S,K,3+C)."""
    if features is not None and (
        features.ndim != 3
        or features.shape[:2] != xyz.shape[:2]
        or features.device != xyz.device
        or features.dtype != xyz.dtype
    ):
        raise ValueError("features must align with XYZ: (B,N,C), same dtype/device")
    centers = index_points(xyz, farthest_point_sample(xyz, npoint))
    indices = query_ball_point(radius, nsample, xyz, centers, chunk_size)
    grouped = index_points(xyz, indices) - centers.unsqueeze(2)
    if features is not None:
        grouped = torch.cat([grouped, index_points(features, indices)], dim=-1)
    return centers, grouped


@profile_range("agri3d::interpolation")
def interpolate_features(
    dense_xyz: Tensor,
    sparse_xyz: Tensor,
    sparse_features: Tensor,
    chunk_size: int = 128,
) -> Tensor:
    """3-NN inverse-squared-distance interpolation -> (B,N,C).

    One/two sparse points are supported. Coincident neighbors share all weight
    equally among the coincident neighbors selected by 3-NN. Feature gradients
    flow through gather; geometry weights/indices are not differentiated.
    """
    check_pair(dense_xyz, sparse_xyz)
    if (
        sparse_features.ndim != 3
        or sparse_features.shape[:2] != sparse_xyz.shape[:2]
        or sparse_features.device != sparse_xyz.device
        or sparse_features.dtype != sparse_xyz.dtype
    ):
        raise ValueError("Sparse features must align with sparse XYZ")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    chunks = []
    for start in range(0, dense_xyz.shape[1], chunk_size):
        with torch.no_grad():
            queries = dense_xyz[:, start : start + chunk_size]
            _, indices = square_distance(queries, sparse_xyz).topk(
                min(3, sparse_xyz.shape[1]), dim=-1, largest=False, sorted=True
            )
            # Recompute selected distances directly to avoid cancellation near exact matches.
            distances = (
                (queries.unsqueeze(2) - index_points(sparse_xyz, indices))
                .square()
                .sum(-1)
            )
            exact = distances == 0
            inverse = distances.clamp_min(torch.finfo(distances.dtype).eps).reciprocal()
            weights = inverse / inverse.sum(-1, keepdim=True)
            exact_weights = exact.to(weights.dtype) / exact.sum(
                -1, keepdim=True
            ).clamp_min(1)
            weights = torch.where(exact.any(-1, keepdim=True), exact_weights, weights)
        chunks.append(
            (index_points(sparse_features, indices) * weights.unsqueeze(-1)).sum(2)
        )
    return torch.cat(chunks, dim=1)
