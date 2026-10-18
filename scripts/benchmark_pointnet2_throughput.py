"""Bounded disposable training throughput at candidate batch sizes; no test access."""

import argparse
import json
from pathlib import Path

import yaml

from agri3d.performance.runtime import add_runtime_arguments, runtime_from_args
from agri3d.performance.throughput import benchmark_batch_sizes
from agri3d.training.pointnet2_runner import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text())
    settings["num_workers"] = args.num_workers
    result = benchmark_batch_sizes(
        args.manifest,
        args.root,
        args.output_dir,
        RunConfig(**settings),
        args.batch_sizes,
        args.warmup_batches,
        args.batches,
        args.device,
        runtime_from_args(args),
        args.checkpoint,
    )
    print(
        json.dumps(
            [
                {
                    "batch_size": r["batch_size"],
                    "status": r["status"],
                    "samples_per_second": r.get("throughput", {}).get(
                        "samples_per_second"
                    ),
                }
                for r in result["results"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
