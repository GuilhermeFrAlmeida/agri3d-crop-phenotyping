"""Profile a short disposable PointNet++ train/val workload; never test."""

import argparse
import json
from pathlib import Path

import yaml

from agri3d.performance.profiling import profile_workload
from agri3d.performance.runtime import add_runtime_arguments, runtime_from_args
from agri3d.training.pointnet2_runner import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--profile-batches", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument(
        "--nvtx-only",
        action="store_true",
        help="Nsight workload: disable torch profiler to avoid CUPTI subscriber conflict",
    )
    add_runtime_arguments(parser)
    args = parser.parse_args()
    try:
        config = (
            RunConfig(**yaml.safe_load(Path(args.config).read_text()))
            if args.config
            else None
        )
        result = profile_workload(
            args.manifest,
            args.root,
            args.output_dir,
            config,
            args.checkpoint,
            args.split,
            args.device,
            args.num_workers,
            args.warmup_batches,
            args.profile_batches,
            runtime_from_args(args),
            args.nvtx or args.nvtx_only,
            not args.nvtx_only,
        )
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "profile_batches": result["profile_batches"],
                "telemetry": result["telemetry"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
