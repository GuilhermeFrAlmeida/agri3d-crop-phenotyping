"""Inspect .pt recovery metadata on CPU without loading plants or training."""

import argparse
import json
from pathlib import Path

from agri3d.performance.inspection import inspect_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--recent", type=int, default=10)
    parser.add_argument(
        "--output", help="Optional JSON report; checkpoint is read-only"
    )
    args = parser.parse_args()
    if args.output and Path(args.output).resolve() == Path(args.checkpoint).resolve():
        parser.error("Inspection output must not overwrite checkpoint")
    result = inspect_checkpoint(args.checkpoint, args.recent)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
