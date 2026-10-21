"""PointNet++ SSG semantic baseline. Public tensors are always (B, N, C).

Architecture follows hierarchical local PointNets and skip-connected feature
propagation from Qi et al. (2017). Implemented independently in pure PyTorch.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .pointnet2_ops import check_xyz, interpolate_features, sample_and_group


@dataclass(frozen=True)
class PointNet2Config:
    npoints: tuple[int, ...] = (1024, 256, 64, 16)
    radii: tuple[float, ...] = (0.1, 0.2, 0.4, 0.8)
    nsamples: tuple[int, ...] = (32, 32, 32, 32)
    chunk_size: int = 128

    def __post_init__(self):
        if not all(len(x) == 4 for x in (self.npoints, self.radii, self.nsamples)):
            raise ValueError("Exactly four abstraction stages required")
        if any(
            not isinstance(n, int) or n < 1 for n in (*self.npoints, *self.nsamples)
        ):
            raise ValueError("npoints/nsamples must be positive integers")
        if any(n < 2 for n in self.npoints[:3]) or any(k < 2 for k in self.nsamples):
            raise ValueError(
                "First three stages and neighborhood sizes must be >=2 for batch-size-one BatchNorm"
            )
        if any(b > a for a, b in zip(self.npoints, self.npoints[1:])):
            raise ValueError("npoints must be nonincreasing")
        if (
            any(not math.isfinite(r) or r <= 0 for r in self.radii)
            or self.chunk_size < 1
        ):
            raise ValueError("Positive finite radii and chunk_size required")

    def to_dict(self):
        return asdict(self)


def shared_mlp(
    in_channels: int, widths: tuple[int, ...], dimensions: int
) -> nn.Sequential:
    conv, norm = (
        (nn.Conv2d, nn.BatchNorm2d) if dimensions == 2 else (nn.Conv1d, nn.BatchNorm1d)
    )
    layers = []
    for width in widths:
        layers.extend([conv(in_channels, width, 1, bias=False), norm(width), nn.ReLU()])
        in_channels = width
    return nn.Sequential(*layers)


class SetAbstraction(nn.Module):
    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channels: int,
        widths: tuple[int, ...],
        chunk_size: int = 128,
    ):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        self.chunk_size = chunk_size
        self.mlp = shared_mlp(in_channels + 3, widths, 2)

    def forward(
        self, xyz: Tensor, features: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        centers, grouped = sample_and_group(
            self.npoint, self.radius, self.nsample, xyz, features, self.chunk_size
        )
        # A local shared MLP acts on (B,C,S,K), then symmetric max over K.
        local = self.mlp(grouped.permute(0, 3, 1, 2).contiguous())
        return centers, local.max(-1).values.transpose(1, 2).contiguous()


class FeaturePropagation(nn.Module):
    def __init__(
        self, in_channels: int, widths: tuple[int, ...], chunk_size: int = 128
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.mlp = shared_mlp(in_channels, widths, 1)

    def forward(
        self,
        dense_xyz: Tensor,
        sparse_xyz: Tensor,
        dense_features: Tensor | None,
        sparse_features: Tensor,
    ) -> Tensor:
        interpolated = interpolate_features(
            dense_xyz, sparse_xyz, sparse_features, self.chunk_size
        )
        if dense_features is not None:
            if (
                dense_features.ndim != 3
                or dense_features.shape[:2] != dense_xyz.shape[:2]
                or dense_features.device != dense_xyz.device
                or dense_features.dtype != dense_xyz.dtype
            ):
                raise ValueError("Dense skip features must align with dense XYZ")
            interpolated = torch.cat([dense_features, interpolated], -1)
        return (
            self.mlp(interpolated.transpose(1, 2).contiguous())
            .transpose(1, 2)
            .contiguous()
        )


class PointNet2Segmentation(nn.Module):
    """Float32 normalized XYZ and optional separate features -> (B,N,2) logits.

    XYZ-only construction preserves legacy weights; Run 4 supplies estimated normals. Requires >=2 points per cloud for
    BatchNorm's final propagation stage during batch-size-one training.
    Padding consists of duplicate XYZ; it participates in model context but
    must be excluded from the objective/metrics using the Dataset valid_mask.
    """

    def __init__(
        self, config: PointNet2Config | None = None, feature_channels: int = 0
    ):
        super().__init__()
        self.config = config or PointNet2Config()
        if type(feature_channels) is not int or feature_channels < 0:
            raise ValueError("feature_channels must be a nonnegative integer")
        self.feature_channels = feature_channels
        channels = (feature_channels, 64, 128, 256)
        widths = ((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512))
        self.abstraction = nn.ModuleList(
            [
                SetAbstraction(n, r, k, c, w, self.config.chunk_size)
                for n, r, k, c, w in zip(
                    self.config.npoints,
                    self.config.radii,
                    self.config.nsamples,
                    channels,
                    widths,
                )
            ]
        )
        self.propagation = nn.ModuleList(
            [
                FeaturePropagation(768, (256, 256), self.config.chunk_size),
                FeaturePropagation(384, (256, 256), self.config.chunk_size),
                FeaturePropagation(320, (256, 128), self.config.chunk_size),
                FeaturePropagation(
                    131 + feature_channels, (128, 128, 128), self.config.chunk_size
                ),
            ]
        )
        self.classifier = nn.Sequential(
            shared_mlp(128, (128,), 1), nn.Conv1d(128, 2, 1)
        )

    def forward(self, xyz: Tensor, features: Tensor | None = None) -> Tensor:
        point_features = features
        check_xyz(xyz)
        if xyz.dtype != torch.float32:
            raise TypeError("PointNet++ expects float32 XYZ")
        if xyz.shape[1] < 2:
            raise ValueError("PointNet++ requires at least two input points")
        if not torch.isfinite(xyz).all():
            raise ValueError("PointNet++ input must be finite")
        if self.feature_channels:
            if point_features is None or point_features.shape != (
                *xyz.shape[:2],
                self.feature_channels,
            ):
                raise ValueError(
                    "Point features must align with XYZ and configured feature channels"
                )
            if point_features.dtype != xyz.dtype or point_features.device != xyz.device:
                raise ValueError("Point features require same dtype/device as XYZ")
            if not torch.isfinite(point_features).all():
                raise ValueError("Point features must be finite")
        elif point_features is not None:
            raise ValueError("XYZ-only model does not accept point features")
        coords, features = [xyz], [point_features]
        for stage in self.abstraction:
            next_xyz, next_features = stage(coords[-1], features[-1])
            coords.append(next_xyz)
            features.append(next_features)
        propagated = features[-1]
        for i, stage in enumerate(self.propagation):
            level = 3 - i
            skip = (
                (
                    xyz
                    if point_features is None
                    else torch.cat([xyz, point_features], -1)
                )
                if level == 0
                else features[level]
            )
            propagated = stage(coords[level], coords[level + 1], skip, propagated)
        return self.classifier(propagated.transpose(1, 2)).transpose(1, 2).contiguous()
