"""Requalify an ONNX artifact against its canonical .pt source model."""
import argparse
from pathlib import Path

from agri3d.inference.export import qualify_model
from agri3d.inference.pointnet2_backend import PyTorchPointNet2Predictor
from agri3d.pipeline.artifacts import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--fixtures")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        p.error("Output must not exist")
    predictor = PyTorchPointNet2Predictor(args.checkpoint)
    try:
        report = qualify_model(predictor, args.model, args.fixtures)
    except Exception as exc:
        report = {"schema_version": 1, "status": "NOT QUALIFIED",
                  "error_type": type(exc).__name__, "blocker": str(exc),
                  "semantic_model": predictor.metadata}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    if report["status"] != "QUALIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
