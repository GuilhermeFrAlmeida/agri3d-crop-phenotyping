"""Validation-mIoU plateau scheduling; disabled policies preserve legacy contracts."""

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class SchedulerConfig:
    enabled: bool = False
    name: str = "reduce_on_plateau"
    mode: str = "max"
    factor: float = 0.5
    patience: int = 8
    threshold: float = 0.0005
    threshold_mode: str = "abs"
    min_lr: float = 0.00001
    cooldown: int = 0
    eps: float = 1e-8

    @classmethod
    def from_settings(cls, settings):
        if not isinstance(settings, dict):
            raise TypeError("scheduler must be a mapping")
        config = cls(**settings)
        if type(config.enabled) is not bool:
            raise TypeError("scheduler.enabled must be boolean")
        if (config.name, config.mode, config.threshold_mode) != (
            "reduce_on_plateau",
            "max",
            "abs",
        ):
            raise ValueError(
                "Scheduler requires reduce_on_plateau, mode=max, threshold_mode=abs"
            )
        for key in ("patience", "cooldown"):
            if type(getattr(config, key)) is not int or getattr(config, key) < 0:
                raise ValueError(f"scheduler.{key} must be a nonnegative integer")
        for key in ("factor", "threshold", "min_lr", "eps"):
            value = getattr(config, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"scheduler.{key} must be finite numeric")
            if value < 0:
                raise ValueError(f"scheduler.{key} must be nonnegative")
        if not 0 < config.factor < 1:
            raise ValueError("scheduler.factor must be in (0, 1)")
        return config

    def contract(self):
        return {**asdict(self), "monitor": "val_mIoU", "step": "after_validation"}

    def build(self, optimizer):
        if not self.enabled:
            return None
        settings = asdict(self)
        for key in ("enabled", "name"):
            settings.pop(key)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **settings)


def scheduler_summary(scheduler):
    if scheduler is None:
        return None
    return {
        "monitor": "val_mIoU",
        "best": scheduler.best,
        "num_bad_epochs": scheduler.num_bad_epochs,
        "cooldown_counter": scheduler.cooldown_counter,
        "last_epoch": scheduler.last_epoch,
    }
