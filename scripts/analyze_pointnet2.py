"""Analyze a regular PointNet++ checkpoint on frozen validation plants only."""

import argparse
import json

from agri3d.analysis.pointnet2 import analyze_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata", help="Optional Metadata.xlsx; exact Tagname join")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--split",
        choices=["val"],
        default="val",
        help="Validation only; test explicitly rejected",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save standalone three-panel Plotly HTML",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save selected sampled-point arrays as NPZ",
    )
    args = parser.parse_args()
    try:
        result = analyze_checkpoint(
            args.manifest,
            args.root,
            args.checkpoint,
            args.output_dir,
            args.metadata,
            args.device,
            args.num_workers,
            args.top_k,
            args.visualize,
            args.save_predictions,
            args.split,
        )
    except (ValueError, TypeError, ImportError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "sample_count": result["sample_count"],
                "metrics": result["metrics"],
                "output_dir": args.output_dir,
                "metadata_join": result["metadata_join"],
                "metadata_value_issue_count": len(
                    result["conditioned_summaries"]["value_issues"]
                ),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
