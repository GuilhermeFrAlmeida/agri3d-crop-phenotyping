"""Bounded or full validation-only structural experiments A–D, no model training."""
import argparse
import json

from agri3d.pipeline.__main__ import build_predictor
from agri3d.pipeline.batch import evaluate_pipeline
from agri3d.pipeline.config import PipelineConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--split", choices=["val"], default="val")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config")
    p.add_argument("--max-plants", type=int)
    p.add_argument("--mode", choices=["both", "oracle", "predicted"], default="both")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch")
    p.add_argument("--qualification")
    p.add_argument("--instance-palette")
    p.add_argument("--reference-traits")
    p.add_argument("--reference-evidence")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--code-commit")
    args = p.parse_args()
    if args.mode != "oracle" and not args.checkpoint:
        p.error("Predicted semantic modes require --checkpoint")
    predictor = build_predictor(args) if args.mode != "oracle" else None
    summary = evaluate_pipeline(args.manifest, args.root, predictor, args.output_dir,
        PipelineConfig.load(args.config), args.split, args.max_plants, args.mode,
        args.fail_fast, args.visualize, args.instance_palette, args.reference_traits,
        args.reference_evidence, args.code_commit)
    print(json.dumps(summary, indent=2))
    if summary["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
