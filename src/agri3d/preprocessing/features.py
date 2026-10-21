"""Feature policy: RGB is supervision only, never a model input."""

from dataclasses import dataclass, field

from .normals import NormalConfig


@dataclass(frozen=True)
class FeatureConfig:
    mode: str = "xyz"
    normals: dict = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings=None):
        if settings is None:
            settings = {}
        if not isinstance(settings, dict):
            raise TypeError("features must be a mapping")
        cfg = cls(**settings)
        if cfg.mode not in ("xyz", "xyz_normals"):
            raise ValueError(
                "Features permit xyz or xyz_normals only; RGB is forbidden"
            )
        if not isinstance(cfg.normals, dict):
            raise TypeError("features.normals must be a mapping")
        if cfg.mode == "xyz" and cfg.normals:
            raise ValueError("Normal policy requires xyz_normals mode")
        if cfg.mode == "xyz_normals":
            NormalConfig.from_settings(cfg.normals)
        return cfg

    @property
    def channels(self):
        return 3 if self.mode == "xyz_normals" else 0

    def contract(self):
        if self.mode == "xyz":
            return "xyz"  # Exact legacy experiment-contract representation.
        return {
            "geometry": "xyz",
            "point_features": "estimated_normals",
            "normal_estimation": NormalConfig.from_settings(self.normals).contract(),
        }
