"""Strict public pipeline configuration; all distance thresholds are meters."""
import math
from dataclasses import asdict, dataclass, field

import yaml


@dataclass(frozen=True)
class InstanceConfig:
    """Phase 4A selection frozen for val78; historical configs remain explicit."""
    method: str = "geometric_graph_v1"
    neighborhood_radius_m: float = 0.020
    max_neighbors: int = 32
    min_leaf_points: int = 100
    attachment_radius_m: float = 0.03

    def __post_init__(self):
        if self.method != "geometric_graph_v1":
            raise ValueError("Unsupported instance method")
        for x in (self.neighborhood_radius_m, self.attachment_radius_m):
            if not math.isfinite(x) or x <= 0:
                raise ValueError("Instance radii must be positive meters")
        for x in (self.max_neighbors, self.min_leaf_points):
            if type(x) is not int or x < 1:
                raise ValueError("Instance counts must be positive integers")


@dataclass(frozen=True)
class TraitConfig:
    voxel_size_m: float = 0.01
    geodesic: bool = True
    geodesic_voxel_size_m: float = 0.01
    geodesic_radius_m: float = 0.03
    geodesic_neighbors: int = 8
    geodesic_max_nodes: int = 2048

    def __post_init__(self):
        for x in (self.voxel_size_m, self.geodesic_voxel_size_m, self.geodesic_radius_m):
            if not math.isfinite(x) or x <= 0:
                raise ValueError("Trait voxel/radius values must be positive meters")
        if type(self.geodesic) is not bool:
            raise ValueError("geodesic must be boolean")
        for x in (self.geodesic_neighbors, self.geodesic_max_nodes):
            if type(x) is not int or x < 2:
                raise ValueError("Geodesic budgets must be integers >=2")


@dataclass(frozen=True)
class PipelineConfig:
    coordinate_units: str = "m"
    vertical_axis: str = "z"
    seed: int = 42
    projection: str = "nearest_neighbor"
    instances: InstanceConfig = field(default_factory=InstanceConfig)
    phenotypes: TraitConfig = field(default_factory=TraitConfig)

    def __post_init__(self):
        if not isinstance(self.instances, InstanceConfig) or not isinstance(self.phenotypes, TraitConfig):
            raise ValueError("Pipeline requires typed instance/phenotype settings")
        if self.coordinate_units != "m" or self.vertical_axis != "z":
            raise ValueError("Current dataset contract requires meters and vertical Z")
        if self.seed != 42 or self.projection != "nearest_neighbor":
            raise ValueError("Frozen sampling seed/projection policy mismatch")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        with open(path) as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a mapping")
        data = dict(data)
        data["instances"] = InstanceConfig(**data.get("instances", {}))
        data["phenotypes"] = TraitConfig(**data.get("phenotypes", {}))
        return cls(**data)
