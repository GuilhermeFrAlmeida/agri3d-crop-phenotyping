"""Public CLI: python -m agri3d.pipeline analyze / analyze-batch."""
import argparse
import json

from .analyze import analyze_plant
from .artifacts import write_plant
from .batch import evaluate_pipeline
from .config import PipelineConfig


def build_predictor(args):
    if args.backend == "onnx":
        from agri3d.inference.onnx_backend import OnnxSemanticPredictor
        if not args.qualification:
            raise ValueError("ONNX requires --qualification from successful parity checks")
        return OnnxSemanticPredictor(args.checkpoint, args.qualification)
    from agri3d.inference.pointnet2_backend import PyTorchPointNet2Predictor
    return PyTorchPointNet2Predictor(args.checkpoint, args.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "analyze-batch"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--config")
        p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        p.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch")
        p.add_argument("--qualification")
        p.add_argument("--visualize", action="store_true")
        p.add_argument("--code-commit")
        if name == "analyze":
            p.add_argument("--input", required=True)
            p.add_argument("--sample-id", help="Use manifest sample ID to reproduce its epoch-0 sample")
            p.add_argument("--output", required=True)
        else:
            p.add_argument("--manifest", required=True)
            p.add_argument("--root", required=True)
            p.add_argument("--split", choices=["val"], default="val")
            p.add_argument("--max-plants", type=int)
            p.add_argument("--output-dir", required=True)
            p.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    config = PipelineConfig.load(args.config)
    predictor = build_predictor(args)
    if args.command == "analyze":
        result = analyze_plant(args.input, predictor, config, args.sample_id, code_commit=args.code_commit)
        write_plant(args.output, result, args.visualize)
        print(json.dumps({"plant_id": result.report["plant_id"], "output": args.output}))
    else:
        summary = evaluate_pipeline(args.manifest, args.root, predictor, args.output_dir,
            config, args.split, args.max_plants, "predicted", args.fail_fast,
            args.visualize, code_commit=args.code_commit)
        print(json.dumps(summary, indent=2))
        if summary["failures"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
