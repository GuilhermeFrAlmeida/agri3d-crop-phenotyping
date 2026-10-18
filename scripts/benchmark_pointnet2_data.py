"""Benchmark loader/transfer and optional normalized cache on train/val only."""

import argparse
import json
from pathlib import Path

import yaml

from agri3d.performance.benchmark import benchmark_loaders
from agri3d.training.pointnet2_runner import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("--pin-memory", choices=["on", "off", "both"], default="both")
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir")
    parser.add_argument("--compare-cache", action="store_true")
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--non-blocking", action="store_true")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Compare transfer off/on and persistent workers off/on",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cfg = RunConfig(**yaml.safe_load(Path(args.config).read_text()))
    pins = (False, True) if args.pin_memory == "both" else (args.pin_memory == "on",)
    result = benchmark_loaders(
        args.manifest,
        args.root,
        args.output_dir,
        cfg,
        args.split,
        args.workers,
        pins,
        args.warmup_batches,
        args.batches,
        args.device,
        args.cache_dir,
        args.compare_cache,
        args.prefetch_factor,
        args.persistent_workers,
        args.non_blocking,
        args.matrix,
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "statuses": [r["status"] for r in result["results"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
