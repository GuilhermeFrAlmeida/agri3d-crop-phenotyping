import numpy as np
from agri3d.io.ply import PointCloudData

class PhenotypeExtractor:
    def extract_traits(self, pcd: PointCloudData, labels: np.ndarray) -> dict:
        # Reverse normalization scaling to compute real-world metrics in meters
        from agri3d.preprocessing.normalization import restore_xyz
        from agri3d.preprocessing.labels import validate_labels
        pcd.validate(finite=True)
        labels = validate_labels(labels)
        if len(labels) != len(pcd.xyz):
            raise ValueError("Labels do not align with geometry")
        original_xyz = restore_xyz(pcd)

        height = np.max(original_xyz[:, 2]) - np.min(original_xyz[:, 2])
        leaf_points = original_xyz[labels == 0]
        stalk_points = original_xyz[labels == 1]

        canopy_volume = 0.0
        if len(leaf_points) >= 4:
            from scipy.spatial import ConvexHull, QhullError
            try:
                hull = ConvexHull(leaf_points)
                canopy_volume = hull.volume
            except QhullError:
                pass

        stalk_height = 0.0
        if len(stalk_points) > 0:
            stalk_height = np.max(stalk_points[:, 2]) - np.min(stalk_points[:, 2])

        return {
            "plant_height_m": float(height),
            "stalk_height_m": float(stalk_height),
            "canopy_volume_m3": float(canopy_volume),
            "leaf_points": int(len(leaf_points)),
            "stalk_points": int(len(stalk_points)),
            "canopy_bounding_box_m": {
                "dx": float(np.max(original_xyz[:, 0]) - np.min(original_xyz[:, 0])),
                "dy": float(np.max(original_xyz[:, 1]) - np.min(original_xyz[:, 1])),
                "dz": float(height)
            }
        }