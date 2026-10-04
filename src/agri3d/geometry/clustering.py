import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from agri3d.io.ply import PointCloudData

class GeometricBaseline:
    """Phase 2: Non-DL baseline for semantic segmentation (0: Leaf, 1: Stalk)."""

    def __init__(self, search_radius: float = 0.05, max_nn: int = 30, vertical_thresh: float = 0.35, eps: float = 0.04, min_samples: int = 20):
        self.search_radius = search_radius
        self.max_nn = max_nn
        self.vertical_thresh = vertical_thresh
        self.eps = eps
        self.min_samples = min_samples

    def segment(self, pcd: PointCloudData) -> np.ndarray:
        """Executes the normal estimation, filtering, and geometric clustering."""
        # Initialize labels array (default to 0: Leaf)
        pcd.validate(finite=True)
        num_points = pcd.xyz.shape[0]
        labels = np.zeros(num_points, dtype=np.int32)

        # 1. Normal Estimation
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(pcd.xyz)
        o3d_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.search_radius, max_nn=self.max_nn
            )
        )
        # Normal signs are irrelevant because the classifier uses absolute Z.
        normals = np.asarray(o3d_cloud.normals)

        # 2. Heuristic Filter: Stalk candidates have horizontal normals (Z close to 0)
        # and sit relatively close to the spatial center (X,Y near 0 assuming Stage 03 Normalization)
        z_normals_abs = np.abs(normals[:, 2])
        xy_distance = np.sqrt(pcd.xyz[:, 0]**2 + pcd.xyz[:, 1]**2)

        # Candidate mask: Low vertical normal component AND close to the central vertical axis
        candidate_mask = (z_normals_abs < self.vertical_thresh) & (xy_distance < np.percentile(xy_distance, 30))
        candidate_indices = np.where(candidate_mask)[0]

        if len(candidate_indices) == 0:
            return labels # Failed to find stalk, return all leaves

        candidate_points = pcd.xyz[candidate_indices]

        # 3. Geometric Clustering: Use DBSCAN to find the contiguous main stalk
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(candidate_points)

        # Find the largest cluster (assumed to be the main continuous stalk)
        unique_clusters, counts = np.unique(clustering.labels_[clustering.labels_ >= 0], return_counts=True)

        if len(unique_clusters) > 0:
            largest_cluster_id = unique_clusters[np.argmax(counts)]
            main_stalk_mask = (clustering.labels_ == largest_cluster_id)

            # Map candidate indices back to original point cloud indices
            stalk_indices = candidate_indices[main_stalk_mask]

            # 4. Stalk/Leaf Candidate Separation: Assign 1 to the identified stalk points
            labels[stalk_indices] = 1

        return labels