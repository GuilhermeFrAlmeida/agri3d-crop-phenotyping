"""Metric full-cloud measurements; these proxies are not validated biological traits."""
import numpy as np
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import ConvexHull, QhullError

from agri3d.inference.predictor import validate_xyz
from agri3d.pipeline.config import TraitConfig
from agri3d.preprocessing.sampling import voxel_indices

from .instances import neighborhood_graph, principal_axis
from .schemas import PhenotypeResult


def canopy_volume(xyz, voxel_size_m):
    xyz = np.asarray(xyz, dtype=np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
            or not np.isfinite(voxel_size_m) or voxel_size_m <= 0):
        raise ValueError("Invalid canopy geometry/voxel size")
    hull, reason = None, "insufficient points" if len(xyz) < 4 else None
    if len(xyz) >= 4:
        try:
            hull = float(ConvexHull(xyz).volume)
        except QhullError:
            reason = "degenerate hull (no jitter/approximation applied)"
    occupied = len(np.unique(np.floor(xyz / voxel_size_m).astype(np.int64), axis=0)) if len(xyz) else 0
    return {"convex_hull_m3": hull, "convex_hull_unavailable_reason": reason,
            "voxel_occupancy_m3": occupied * voxel_size_m ** 3,
            "occupied_voxels": occupied, "voxel_size_m": voxel_size_m,
            "voxel_origin_m": [0., 0., 0.]}


def leaf_lengths(xyz, config):
    xyz = np.asarray(xyz, dtype=np.float64)
    result = {"length_pca_m": None, "length_geodesic_m": None,
              "geodesic_unavailable_reason": None}
    if len(xyz) < 2 or not np.any(xyz - xyz[0]):
        result["geodesic_unavailable_reason"] = "insufficient distinct points"
        return result
    axis = np.asarray(principal_axis(xyz))
    result["length_pca_m"] = float(np.ptp((xyz - xyz.mean(0)) @ axis))
    if not config.geodesic:
        result["geodesic_unavailable_reason"] = "disabled"
        return result
    nodes = xyz[voxel_indices(xyz, config.geodesic_voxel_size_m)]
    result["geodesic_nodes"] = len(nodes)
    if not 2 <= len(nodes) <= config.geodesic_max_nodes:
        result["geodesic_unavailable_reason"] = "node budget/insufficient nodes"
        return result
    graph = neighborhood_graph(nodes, config.geodesic_radius_m, config.geodesic_neighbors)
    count, _ = connected_components(graph, directed=False)
    if count != 1:
        result["geodesic_unavailable_reason"] = "disconnected graph"
        return result
    first = dijkstra(graph, directed=False, indices=0)
    second = dijkstra(graph, directed=False, indices=int(np.argmax(first)))
    result["length_geodesic_m"] = float(second.max())
    return result


class PhenotypeExtractor:
    def __init__(self, config=None):
        self.config = config or TraitConfig()

    def extract(self, metric_xyz, structure):
        xyz = validate_xyz(metric_xyz).astype(np.float64)
        structure.instance_labels(len(xyz))
        leaf_ids = np.concatenate([leaf.point_indices for leaf in structure.leaves] +
                                  [structure.unassigned_indices])
        leaves = [{"leaf_id": leaf.instance_id, "point_count": len(leaf.point_indices),
                   "attachment_point_m": leaf.attachment_point,
                   **leaf_lengths(xyz[leaf.point_indices], self.config)} for leaf in structure.leaves]
        stalk = xyz[structure.stalk_indices]
        return PhenotypeResult({
            "height_m": float(np.ptp(xyz[:, 2])), "z_min": float(xyz[:, 2].min()),
            "z_max": float(xyz[:, 2].max()), "point_count_used": len(xyz),
            "leaf_count": len(leaves), "leaf_points": len(leaf_ids), "stalk_points": len(stalk),
            "canopy_volume": canopy_volume(xyz[leaf_ids], self.config.voxel_size_m),
        }, leaves, {
            "point_count": len(stalk),
            "vertical_extent_m": float(np.ptp(stalk[:, 2])) if len(stalk) else None,
            "centroid_m": stalk.mean(0).tolist() if len(stalk) else None,
            "principal_axis": principal_axis(stalk),
        }, {
            "height": "full cleaned metric cloud max(z)-min(z); vertical Z assumption",
            "leaf_count": "accepted geometric components, not validated leaf identities",
            "length_pca": "extent on first principal axis; curved-leaf underestimate",
            "length_geodesic": "two Dijkstra sweeps on voxel-representative radius graph; diameter lower bound, not midrib",
            "volume": "leaf-point convex hull and origin-anchored surface occupancy; neither is tissue volume",
        })
