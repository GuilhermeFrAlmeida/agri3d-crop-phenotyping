import numpy as np

class CloudFilter:
    def __init__(self, sor_nb_neighbors=20, sor_std_ratio=2.0, voxel_size=None):
        if sor_nb_neighbors < 2 or sor_std_ratio <= 0:
            raise ValueError("Invalid SOR parameters")
        self.sor_nb_neighbors = sor_nb_neighbors
        self.sor_std_ratio = sor_std_ratio
        self.voxel_size = voxel_size

    def remove_nan_inf(self, pcd):
        pcd.validate()
        return pcd.select(np.flatnonzero(np.isfinite(pcd.xyz).all(axis=1)))

    def statistical_outlier_removal(self, pcd):
        pcd.validate(finite=True)
        if len(pcd.xyz) <= self.sor_nb_neighbors:
            return pcd.select(np.arange(len(pcd.xyz)))
        import open3d as o3d
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pcd.xyz)
        _, indices = cloud.remove_statistical_outlier(
            nb_neighbors=self.sor_nb_neighbors, std_ratio=self.sor_std_ratio)
        return pcd.select(np.asarray(indices, dtype=np.int64))

    def voxel_downsample(self, pcd):
        from agri3d.preprocessing.sampling import voxel_indices
        return pcd.select(voxel_indices(pcd.xyz, self.voxel_size))

    def process(self, pcd):
        result = self.statistical_outlier_removal(self.remove_nan_inf(pcd))
        return self.voxel_downsample(result) if self.voxel_size is not None else result
