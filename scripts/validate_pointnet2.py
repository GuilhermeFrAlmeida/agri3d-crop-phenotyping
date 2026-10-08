"""Evaluate a selected regular-run checkpoint on the frozen validation split only."""

import argparse
import json

from agri3d.training.pointnet2_runner import validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    try:
        result = validate_checkpoint(
            args.manifest,
            args.root,
            args.checkpoint,
            args.output,
            args.device,
            args.num_workers,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
