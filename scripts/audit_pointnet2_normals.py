"""Audit full-cloud estimated normals before Run 4 training; train/val only."""

import argparse
import json
from pathlib import Path

import yaml

from agri3d.analysis.normal_audit import audit_normals
from agri3d.performance.runtime import RuntimeOptions
from agri3d.training.pointnet2_runner import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--max-plants", type=int, default=5)
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        config = RunConfig(**yaml.safe_load(Path(args.config).read_text()))
        result = audit_normals(
            args.manifest,
            args.root,
            args.output,
            config,
            args.split,
            args.max_plants,
            RuntimeOptions(cache_dir=args.cache_dir),
        )
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "plants_checked",
                    "points_checked",
                    "finite_normal_ratio",
                    "degenerate_neighborhood_count",
                    "passed",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
