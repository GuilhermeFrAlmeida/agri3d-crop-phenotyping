"""Typed structure indices address the cleaned, full metric cloud."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LeafInstance:
    instance_id: int
    point_indices: np.ndarray
    attachment_point: list | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PlantStructure:
    stalk_indices: np.ndarray
    leaves: list[LeafInstance]
    unassigned_indices: np.ndarray
    metadata: dict = field(default_factory=dict)

    def instance_labels(self, count):
        result = np.full(count, -1, dtype=np.int32)
        result[self.stalk_indices] = 0
        all_ids = [self.stalk_indices, self.unassigned_indices]
        for leaf in self.leaves:
            if leaf.instance_id < 1:
                raise ValueError("Leaf IDs must be positive")
            result[leaf.point_indices] = leaf.instance_id
            all_ids.append(leaf.point_indices)
        flat = np.concatenate(all_ids)
        if (len(flat) != count or len(np.unique(flat)) != count
                or not np.array_equal(np.sort(flat), np.arange(count))):
            raise ValueError("Structure must account for every point exactly once")
        if len({leaf.instance_id for leaf in self.leaves}) != len(self.leaves):
            raise ValueError("Duplicate instance IDs")
        return result


@dataclass
class PhenotypeResult:
    phenotypes: dict
    leaves: list[dict]
    stalk: dict
    methods: dict

    def to_dict(self):
        return {"phenotypes": self.phenotypes, "leaves": self.leaves,
                "stalk": self.stalk, "methods": self.methods}
