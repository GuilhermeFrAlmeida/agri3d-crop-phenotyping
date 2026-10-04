from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np
from plyfile import PlyData

@dataclass
class PointCloudData:
    xyz: np.ndarray
    attributes: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def validate(self, finite=False):
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or len(self.xyz) == 0:
            raise ValueError("Expected nonempty (N, 3) XYZ")
        if self.xyz.dtype != np.float32:
            raise TypeError("XYZ must be float32")
        if any(np.asarray(v).ndim == 0 or len(v) != len(self.xyz) for v in self.attributes.values()):
            raise ValueError("Point attributes must align with XYZ")
        if finite and not np.isfinite(self.xyz).all():
            raise ValueError("Nonfinite XYZ")
        return self

    def select(self, indices):
        return PointCloudData(self.xyz[indices].copy(),
            {k: v[indices].copy() for k, v in self.attributes.items()},
            deepcopy(self.metadata)).validate()

class PLYReader:
    """Read geometry and explicit properties. RGB is never inferred to be ground truth."""
    def __init__(self, file_path):
        self.file_path = str(file_path)

    def load_point_cloud(self):
        vertex = PlyData.read(self.file_path)['vertex']
        names = vertex.data.dtype.names
        if not all(k in names for k in ('x', 'y', 'z')):
            raise ValueError("PLY requires x, y, z")
        xyz = np.column_stack([vertex[k] for k in ('x','y','z')]).astype(np.float32)
        attrs = {}
        for fields, key in [(('red','green','blue'),'colors'), (('nx','ny','nz'),'normals')]:
            if all(k in names for k in fields):
                attrs[key] = np.column_stack([vertex[k] for k in fields])
        for key in ('label', 'instance_id', 'intensity'):
            if key in names:
                attrs[key] = np.asarray(vertex[key]).copy()
        attrs['source_index'] = np.arange(len(xyz), dtype=np.int64)
        return PointCloudData(xyz, attrs, {'source_file': self.file_path,
            'point_count': len(xyz)}).validate()

    def validate_point_cloud(self, pcd):
        pcd.validate()
        return True
