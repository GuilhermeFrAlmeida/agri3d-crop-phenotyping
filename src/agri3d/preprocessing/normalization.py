import numpy as np

class CloudNormalizer:
    def __init__(self, center=True, scale_to_unit_sphere=False):
        self.center = center
        self.scale_to_unit_sphere = scale_to_unit_sphere

    def process(self, pcd):
        pcd.validate(finite=True)
        if 'scale_factor' in pcd.metadata or 'centroid_shift' in pcd.metadata:
            raise ValueError("Cloud already normalized")
        result = pcd.select(np.arange(len(pcd.xyz)))
        xyz = result.xyz.astype(np.float64)
        shift = xyz.mean(axis=0) if self.center else np.zeros(3)
        xyz -= shift
        scale = float(np.linalg.norm(xyz, axis=1).max()) if self.scale_to_unit_sphere else 1.0
        scale = scale if scale > 0 else 1.0
        result.xyz = (xyz / scale).astype(np.float32)
        result.metadata.update(centroid_shift=shift.tolist(), scale_factor=scale)
        return result.validate(finite=True)

def restore_xyz(pcd):
    return pcd.xyz.astype(np.float64) * pcd.metadata.get('scale_factor', 1.0) + np.asarray(pcd.metadata.get('centroid_shift', [0,0,0]))
