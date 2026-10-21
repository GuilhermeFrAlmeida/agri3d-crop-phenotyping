import hashlib
import inspect
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from agri3d.io.ply import PLYReader
from agri3d.preprocessing.augmentation import AugmentationConfig, augment_cloud
from agri3d.preprocessing.cache import NormalizedCloudCache
from agri3d.preprocessing.features import FeatureConfig
from agri3d.preprocessing.filtering import CloudFilter
from agri3d.preprocessing.labels import decode_labels, decode_maizefield3d_labels
from agri3d.preprocessing.normalization import CloudNormalizer
from agri3d.preprocessing.normals import (
    NormalConfig,
    estimate_normals,
    normal_provenance,
    validate_normals,
)
from agri3d.preprocessing.sampling import FixedSizeSampler
from agri3d.training.splits import validate_manifest


class SegmentationDataset(Dataset):
    """XYZ-only segmentation samples; opt into MaizeField3D RGB supervision.

    ``annotation_mode="maizefield3d"`` applies only to segmented exports.
    Generic explicit mappings remain the default for other datasets.
    Optional training augmentation follows sampling and uses the same per-sample RNG.
    Centroid/scale metadata describes pre-augmentation normalization. Validation
    ignores augmentation and retains the original fixed-point contract.
    """

    def __init__(
        self,
        manifest,
        root,
        split,
        num_points=10000,
        seed=42,
        color_map=None,
        label_map=None,
        sampling="voxel",
        voxel_size=0.005,
        sor=False,
        annotation_mode="explicit",
        augmentation=None,
        cache_dir=None,
        profile_data=False,
        features=None,
    ):
        if annotation_mode not in ("explicit", "maizefield3d"):
            raise ValueError("Unknown annotation_mode")
        if annotation_mode == "maizefield3d" and (
            color_map is not None or label_map is not None
        ):
            raise ValueError(
                "MaizeField3D mode cannot be combined with explicit mappings"
            )
        self.augmentation = AugmentationConfig.from_settings(augmentation)
        if split == "test" and self.augmentation.enabled:
            raise ValueError("Training augmentation is not permitted for test datasets")
        self.feature_policy = FeatureConfig.from_settings(features)
        if self.feature_policy.channels:
            if (
                split not in ("train", "val")
                or annotation_mode != "maizefield3d"
                or sor
            ):
                raise ValueError(
                    "Normal features permit train/val MaizeField3D without SOR only; test is forbidden"
                )
            if self.augmentation.enabled:
                raise ValueError("Run 4 normals require disabled augmentation")
        self.annotation_mode = annotation_mode
        records = (
            json.loads(Path(manifest).read_text())
            if isinstance(manifest, (str, Path))
            else manifest
        )
        validate_manifest(records)
        self.manifest_hash = hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        self.records = [r for r in records if r["split"] == split]
        if not self.records:
            raise ValueError("Requested split is empty")
        self.root = Path(root).resolve()
        for r in self.records:
            path = (self.root / r["path"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError("Manifest path missing or outside data root")
        if cache_dir and (split == "test" or annotation_mode != "maizefield3d" or sor):
            raise ValueError("Cache supports train/val MaizeField3D without SOR only")
        self.cache = (
            NormalizedCloudCache(
                cache_dir,
                hashlib.sha256(
                    inspect.getsource(type(self)._normalized_cloud).encode()
                ).hexdigest(),
                feature_provenance=(
                    normal_provenance(
                        NormalConfig.from_settings(self.feature_policy.normals)
                    )
                    if self.feature_policy.channels
                    else None
                ),
                feature_contract=self.feature_policy.contract(),
            )
            if cache_dir
            else None
        )
        self.profile_data = profile_data
        self._shared_epoch = None
        self.split, self.seed, self.epoch = split, seed, 0
        self.color_map, self.label_map, self.sor = color_map, label_map, sor
        self.sampler = FixedSizeSampler(num_points, sampling, voxel_size)

    @property
    def epoch(self):
        return (
            self._shared_epoch.value if self._shared_epoch is not None else self._epoch
        )

    @epoch.setter
    def epoch(self, value):
        self._epoch = value
        if self._shared_epoch is not None:
            self._shared_epoch.value = value

    def enable_shared_epoch(self):
        if self._shared_epoch is None:
            self._shared_epoch = multiprocessing.get_context("spawn").Value(
                "q", self.epoch
            )

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        epoch = self.epoch if self.split == "train" else 0
        digest = hashlib.sha256(
            f"{self.seed}:{epoch}:{r['sample_id']}".encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        started = time.perf_counter() if self.profile_data else 0
        path = self.root / r["path"]
        pcd = (
            self.cache.get(
                path,
                lambda: self._normalized_cloud(path),
                {
                    "manifest_sha256": self.manifest_hash,
                    "sample_id": r["sample_id"],
                    "path": r["path"],
                    "split": self.split,
                },
            )
            if self.cache
            else self._normalized_cloud(path)
        )
        if self.feature_policy.channels:
            validate_normals(pcd.attributes["normals"], len(pcd.xyz))
        normalized_ms = (
            (time.perf_counter() - started) * 1000 if self.profile_data else 0
        )
        started = time.perf_counter() if self.profile_data else 0
        pcd = self.sampler.sample(pcd, rng)
        sampling_ms = (time.perf_counter() - started) * 1000 if self.profile_data else 0
        started = time.perf_counter() if self.profile_data else 0
        if self.split == "train" and self.augmentation.enabled:
            pcd = augment_cloud(pcd, rng, self.augmentation)
        augmentation_ms = (
            (time.perf_counter() - started) * 1000 if self.profile_data else 0
        )
        result = {
            "points": torch.from_numpy(pcd.xyz),
            "labels": torch.from_numpy(pcd.attributes["label"]),
            "valid_mask": torch.from_numpy(pcd.attributes["valid_mask"]),
            "source_index": torch.from_numpy(pcd.attributes["source_index"]),
            "sample_id": r["sample_id"],
            "centroid": torch.tensor(
                pcd.metadata["centroid_shift"], dtype=torch.float64
            ),
            "scale": torch.tensor(pcd.metadata["scale_factor"], dtype=torch.float64),
        }

        if self.feature_policy.channels:
            validate_normals(pcd.attributes["normals"], len(pcd.xyz))
            result["features"] = torch.from_numpy(pcd.attributes["normals"])
        if self.profile_data:
            result["_cache_hit"] = int(self.cache.last_hit) if self.cache else -1
            result["_data_timing_ms"] = {
                "normalized_cloud": normalized_ms,
                "sampling": sampling_ms,
                "augmentation": augmentation_ms,
            }
        return result

    def _normalized_cloud(self, path):
        pcd = PLYReader(path).load_point_cloud()
        pcd = CloudFilter().remove_nan_inf(pcd)
        pcd = (
            decode_maizefield3d_labels(pcd)
            if self.annotation_mode == "maizefield3d"
            else decode_labels(pcd, self.color_map, self.label_map)
        )
        if self.sor:
            pcd = CloudFilter().statistical_outlier_removal(pcd)
        if self.feature_policy.channels:
            cfg = NormalConfig.from_settings(self.feature_policy.normals)
            normals, degenerate = estimate_normals(pcd.xyz, cfg)
            pcd.attributes["normals"] = (
                normals  # Always estimated; ignore any PLY normal fields.
            )
            pcd.attributes["normal_degenerate"] = degenerate
            pcd.metadata["normal_estimation"] = {
                "policy": cfg.contract(),
                "full_cloud_points": len(pcd.xyz),
                "effective_k": min(cfg.k, len(pcd.xyz)),
                "degenerate_count": int(degenerate.sum()),
            }
        pcd = CloudNormalizer(scale_to_unit_sphere=True).process(pcd)
        if self.feature_policy.channels:
            validate_normals(pcd.attributes["normals"], len(pcd.xyz))
        return pcd
