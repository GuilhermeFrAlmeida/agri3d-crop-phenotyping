"""Export the frozen .pt model and qualify fixed-batch ONNX numerical behavior."""
import argparse
import json

from agri3d.inference.export import export_pointnet2
from agri3d.inference.pointnet2_backend import PyTorchPointNet2Predictor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fixtures", help="Optional NPZ with normalized float32 (10000,3) arrays")
    args = p.parse_args()
    report = export_pointnet2(PyTorchPointNet2Predictor(args.checkpoint), args.output_dir, args.fixtures)
    print(json.dumps({"status": report["status"], "report": args.output_dir + "/qualification.json"}))
    if report["status"] != "QUALIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
