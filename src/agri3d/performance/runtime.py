"""Runtime controls are separate from RunConfig's scientific experiment contract."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


@dataclass(frozen=True)
class RuntimeOptions:
    pin_memory: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    non_blocking: bool = False
    cache_dir: str | None = None
    detailed_timing: bool = False
    telemetry: bool = False

    def __post_init__(self):
        if isinstance(self.cache_dir, Path):
            object.__setattr__(self, "cache_dir", str(self.cache_dir))
        if self.cache_dir is not None and not isinstance(self.cache_dir, str):
            raise TypeError("cache_dir must be a path string or Path")

    def validate(self, num_workers):
        for name in (
            "pin_memory",
            "persistent_workers",
            "non_blocking",
            "detailed_timing",
            "telemetry",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        if not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if self.prefetch_factor is not None and (
            type(self.prefetch_factor) is not int or self.prefetch_factor < 1
        ):
            raise ValueError("prefetch_factor must be a positive integer or null")
        if num_workers == 0 and (
            self.persistent_workers or self.prefetch_factor is not None
        ):
            raise ValueError(
                "prefetch_factor/persistent_workers require num_workers > 0"
            )


class EpochDataLoader(DataLoader):
    """Persistent workers preserve the legacy per-iterator base-seed RNG draw.

    Local per-sample RNGs do not depend on worker RNG state. Consuming the same
    int64 draw as a recreated iterator keeps the RandomSampler/loader generator
    state equal at epoch boundaries, including across checkpoint resume.
    """

    def __iter__(self):
        previous = getattr(self, "_phase3p_iterations", 0)
        if previous and self.persistent_workers:
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        self._phase3p_iterations = previous + 1
        return super().__iter__()


def build_loader(dataset, config, shuffle, generator, runtime=None):
    runtime = runtime or RuntimeOptions()
    runtime.validate(config.num_workers)
    if runtime.persistent_workers:
        base = dataset
        while isinstance(base, Subset):
            base = base.dataset
        base.enable_shared_epoch()
    kwargs = {}
    if config.num_workers:
        kwargs["multiprocessing_context"] = "spawn"
        if runtime.prefetch_factor is not None:
            kwargs["prefetch_factor"] = runtime.prefetch_factor
    loader_type = EpochDataLoader if runtime.persistent_workers else DataLoader
    return loader_type(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        persistent_workers=runtime.persistent_workers,
        pin_memory=runtime.pin_memory,
        generator=generator,
        drop_last=False,
        **kwargs,
    )


def add_runtime_arguments(parser):
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--non-blocking", action="store_true")
    parser.add_argument("--cache-dir", help="Opt-in local normalized-cloud cache")
    parser.add_argument(
        "--detailed-timing",
        action="store_true",
        help="Synchronizing debug timings; slows CUDA",
    )
    parser.add_argument("--telemetry", action="store_true")


def runtime_from_args(args):
    return RuntimeOptions(
        **{name: getattr(args, name) for name in RuntimeOptions.__dataclass_fields__}
    )
