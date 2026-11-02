"""Bounded-degree metric radius graph baseline; touching leaves may merge."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from agri3d.inference.predictor import validate_xyz
from agri3d.pipeline.config import InstanceConfig
from agri3d.preprocessing.labels import validate_labels

from .schemas import LeafInstance, PlantStructure


def neighborhood_graph(xyz, radius, neighbors):
    """Symmetric capped kNN radius graph; exact SciPy single-worker search."""
    xyz = validate_xyz(xyz).astype(np.float64)
    tree = cKDTree(xyz)
    k = min(neighbors + 1, len(xyz))
    d, ix = tree.query(xyz, k=k, workers=1)
    d, ix = np.asarray(d).reshape(len(xyz), k), np.asarray(ix).reshape(len(xyz), k)
    rows, cols, values = [], [], []
    for i in range(len(xyz)):
        candidates = ix[i]
        candidates = candidates[candidates != i]
        distance = np.linalg.norm(xyz[candidates] - xyz[i], axis=1)
        order = np.lexsort((candidates, distance))
        candidates, distance = candidates[order][:neighbors], distance[order][:neighbors]
        good = distance <= radius
        rows.extend([i] * int(good.sum()))
        cols.extend(candidates[good].tolist())
        # Positive duplicate-edge weights preserve zero-distance connectivity for Dijkstra.
        values.extend(np.maximum(distance[good], 1e-15).tolist())
    graph = coo_matrix((values, (rows, cols)), shape=(len(xyz), len(xyz))).tocsr()
    return graph.maximum(graph.T)


def principal_axis(xyz):
    if len(xyz) < 2:
        return None
    centered = np.asarray(xyz, dtype=np.float64) - np.mean(xyz, axis=0)
    if not np.any(centered):
        return None
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    return axis.tolist()


class InstanceExtractor:
    def __init__(self, config=None):
        self.config = config or InstanceConfig()

    def extract(self, xyz, semantic_labels):
        xyz = validate_xyz(xyz)
        semantic_labels = validate_labels(semantic_labels)
        if len(semantic_labels) != len(xyz):
            raise ValueError("Semantic labels must align with metric cloud")
        stalk = np.flatnonzero(semantic_labels == 1)
        leaf_ids = np.flatnonzero(semantic_labels == 0)
        accepted, rejected, sizes = [], [], []
        raw_components, graph_edges = 0, 0
        stalk_tree = cKDTree(xyz[stalk]) if len(stalk) else None
        if len(leaf_ids):
            graph = neighborhood_graph(xyz[leaf_ids], self.config.neighborhood_radius_m,
                                       self.config.max_neighbors)
            graph_edges = int(graph.nnz // 2)
            raw_components, component = connected_components(graph, directed=False)
            # Label IDs follow the smallest cleaned source row; never rely on library order.
            groups = [leaf_ids[component == i] for i in range(raw_components)]
            groups.sort(key=lambda ids: int(ids.min()))
            for ids in groups:
                sizes.append(len(ids))
                if len(ids) < self.config.min_leaf_points:
                    rejected.append(ids)
                    continue
                attachment, distance = None, None
                if stalk_tree is not None:
                    distances, nearest = stalk_tree.query(xyz[ids], workers=1)
                    j = int(np.argmin(distances))
                    attachment = xyz[stalk[int(nearest[j])]].astype(float).tolist()
                    distance = float(distances[j])
                accepted.append(LeafInstance(len(accepted) + 1, ids, attachment, {
                    "distance_to_stalk_m": distance,
                    "attachment_consistent": distance <= self.config.attachment_radius_m
                    if distance is not None else None,
                    "principal_axis": principal_axis(xyz[ids]),
                }))
        unassigned = np.sort(np.concatenate(rejected)) if rejected else np.array([], dtype=int)
        structure = PlantStructure(stalk, accepted, unassigned, {
            "method": self.config.method, "raw_components": int(raw_components),
            "graph_undirected_edges": graph_edges,
            "accepted_leaf_instances": len(accepted), "rejected_small_components": len(rejected),
            "component_sizes": sizes, "unassigned_leaf_points": len(unassigned),
            "unassigned_fraction_of_leaf": len(unassigned) / len(leaf_ids) if len(leaf_ids) else 0.,
            "stalk_axis": principal_axis(xyz[stalk]) if len(stalk) else None,
            "attachment_rule": "closest component/stalk pair; diagnostic only; no merging/splitting",
        })
        structure.instance_labels(len(xyz))
        return structure
