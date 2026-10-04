import open3d as o3d
import numpy as np
import os
from agri3d.io.ply import PointCloudData

class MeshReconstructor:
    """Generates 3D surface meshes from segmented point clouds using Alpha Shapes or Ball Pivoting."""

    def __init__(self, method: str = "alpha", alpha: float = 0.03, depth: int = 8):
        self.method = method
        self.alpha = alpha
        self.depth = depth

    def reconstruct(self, pcd: PointCloudData, labels: np.ndarray = None) -> o3d.geometry.TriangleMesh:
        """Reconstructs 3D mesh and transfers class colors to vertices."""
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(pcd.xyz)

        # Attach visualization colors if labels are present
        if labels is not None:
            colors = np.zeros((pcd.xyz.shape[0], 3))
            colors[labels == 0] = [0.1, 0.8, 0.2]  # Leaf: Green
            colors[labels == 1] = [0.6, 0.4, 0.2]  # Stalk: Brown
            o3d_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Estimate normals required for mesh reconstruction algorithms
        o3d_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        o3d_cloud.orient_normals_consistent_tangent_plane(100)

        if self.method == "alpha":
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d_cloud, self.alpha)
        elif self.method == "poisson":
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_cloud, depth=self.depth)
            # Crop Poisson mesh to bounding box of point cloud to remove outer hull artifacts
            bbox = o3d_cloud.get_axis_aligned_bounding_box()
            mesh = mesh.crop(bbox)
        else:
            raise ValueError(f"Unsupported reconstruction method: {self.method}")

        mesh.compute_vertex_normals()
        return mesh

    def export_mesh(self, mesh: o3d.geometry.TriangleMesh, output_path: str):
        """Saves the mesh to .ply or .obj format."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        o3d.io.write_triangle_mesh(output_path, mesh)