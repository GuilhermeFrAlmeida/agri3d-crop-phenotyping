import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from agri3d.io.ply import PLYReader
from agri3d.preprocessing.labels import decode_labels, decode_maizefield3d_labels
from agri3d.preprocessing.filtering import CloudFilter
from agri3d.preprocessing.normalization import CloudNormalizer
from agri3d.preprocessing.sampling import FixedSizeSampler
from agri3d.training.splits import validate_manifest

class SegmentationDataset(Dataset):
    """XYZ-only segmentation samples; opt into MaizeField3D RGB supervision.

    ``annotation_mode="maizefield3d"`` applies only to segmented exports.
    Generic explicit mappings remain the default for other datasets.
    """
    def __init__(self, manifest, root, split, num_points=10000, seed=42,
                 color_map=None, label_map=None, sampling='voxel', voxel_size=0.005,
                 sor=False, annotation_mode="explicit"):
        if annotation_mode not in ("explicit", "maizefield3d"):
            raise ValueError("Unknown annotation_mode")
        if annotation_mode == "maizefield3d" and (color_map is not None or label_map is not None):
            raise ValueError("MaizeField3D mode cannot be combined with explicit mappings")
        self.annotation_mode = annotation_mode
        records = json.loads(Path(manifest).read_text()) if isinstance(manifest, (str, Path)) else manifest
        validate_manifest(records)
        if split not in ('train','val','test'):
            raise ValueError("Unknown split")
        self.records = [r for r in records if r['split'] == split]
        if not self.records:
            raise ValueError("Requested split is empty")
        self.root = Path(root).resolve()
        for r in self.records:
            path = (self.root / r['path']).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError("Manifest path missing or outside data root")
        self.split, self.seed, self.epoch = split, seed, 0
        self.color_map, self.label_map, self.sor = color_map, label_map, sor
        self.sampler = FixedSizeSampler(num_points, sampling, voxel_size)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        epoch = self.epoch if self.split == 'train' else 0
        digest = hashlib.sha256(f"{self.seed}:{epoch}:{r['sample_id']}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        pcd = PLYReader(self.root / r['path']).load_point_cloud()
        pcd = CloudFilter().remove_nan_inf(pcd)
        pcd = (decode_maizefield3d_labels(pcd) if self.annotation_mode == "maizefield3d"
               else decode_labels(pcd, self.color_map, self.label_map))
        if self.sor:
            pcd = CloudFilter().statistical_outlier_removal(pcd)
        pcd = CloudNormalizer(scale_to_unit_sphere=True).process(pcd)
        pcd = self.sampler.sample(pcd, rng)
        return {'points': torch.from_numpy(pcd.xyz),
                'labels': torch.from_numpy(pcd.attributes['label']),
                'valid_mask': torch.from_numpy(pcd.attributes['valid_mask']),
                'source_index': torch.from_numpy(pcd.attributes['source_index']),
                'sample_id': r['sample_id'],
                'centroid': torch.tensor(pcd.metadata['centroid_shift'], dtype=torch.float64),
                'scale': torch.tensor(pcd.metadata['scale_factor'], dtype=torch.float64)}
