"""Phase 4A: reuse the fixed val5 artifacts for a geometry-only 12-cell grid."""
import argparse
import json

from agri3d.analysis.instance_sweep import SweepConfig, finalize_review, run_sweep
from agri3d.pipeline.config import PipelineConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-artifacts", required=True)
    p.add_argument("--config", default="configs/phenotyping_pipeline_phase4a_initial.yaml")
    p.add_argument("--sweep-config", default="configs/phase4a_instance_sweep.yaml")
    p.add_argument("--mode", choices=["both", "oracle", "predicted"], default="both")
    p.add_argument("--split", choices=["val"], default="val")
    p.add_argument("--plants", nargs="+")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--review-json", help="Explicit multi-criteria review bound to this input fingerprint")
    p.add_argument("--finalize-review", action="store_true", help="Attach review to completed output without repeating geometry/inference")
    p.add_argument("--no-visualizations", action="store_true")
    p.add_argument("--code-commit")
    args = p.parse_args()
    if args.finalize_review:
        if not args.review_json:
            p.error("--finalize-review requires --review-json")
        summary, selected = finalize_review(args.input_artifacts, args.output_dir, args.review_json, not args.no_visualizations)
    else:
        summary, selected = run_sweep(args.input_artifacts, args.output_dir,
            PipelineConfig.load(args.config), SweepConfig.load(args.sweep_config), args.mode,
            args.plants, args.split, args.review_json, not args.no_visualizations, args.code_commit)
    print(json.dumps({"rows": summary["rows"], "decision": selected["decision"],
                      "phase4_readiness": selected["phase4_readiness"]}, indent=2))


if __name__ == "__main__":
    main()
