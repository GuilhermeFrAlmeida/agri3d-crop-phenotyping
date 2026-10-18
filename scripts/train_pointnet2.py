"""Train a pure-PyTorch PointNet++ baseline using the frozen manifest."""

import argparse
from dataclasses import fields
from pathlib import Path

import yaml

from agri3d.performance.runtime import add_runtime_arguments, runtime_from_args
from agri3d.training.pointnet2_runner import RunConfig, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", help="Optional YAML RunConfig; CLI values override it"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    for name in (
        "points",
        "epochs",
        "batch-size",
        "seed",
        "num-workers",
        "overfit-samples",
    ):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device")
    parser.add_argument(
        "--resume", help="Resume output-dir/last.pt; --epochs is the total target"
    )
    parser.add_argument(
        "--overfit-checkpoint",
        help="Passing tiny-overfit checkpoint required for a new regular run",
    )
    add_runtime_arguments(parser)
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    if not isinstance(settings, dict):
        parser.error("Configuration must be a YAML mapping")
    allowed = {f.name for f in fields(RunConfig)}
    if set(settings) - allowed:
        parser.error(f"Unknown configuration fields: {sorted(set(settings) - allowed)}")
    for name in allowed - {"model", "augmentation", "scheduler"}:
        value = getattr(args, name, None)
        if value is not None:
            settings[name] = value
    try:
        train(
            args.manifest,
            args.root,
            args.output_dir,
            RunConfig(**settings),
            resume=args.resume,
            overfit_checkpoint=args.overfit_checkpoint,
            runtime=runtime_from_args(args),
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
