"""Fixed-shape ONNX export and numerical qualification; never approximate geometry."""
import json
from pathlib import Path

import numpy as np

from agri3d.pipeline.artifacts import write_json
from agri3d.pipeline.provenance import provenance, sha256_file


class ExportNotQualified(RuntimeError):
    pass


def qualification_inputs(points, fixture_path=None):
    rng = np.random.default_rng(42)
    arrays = []
    for mode in ("volume", "plane", "line"):
        xyz = rng.uniform(-0.8, 0.8, (points, 3)).astype(np.float32)
        if mode == "plane":
            xyz[:, 2] = 0
        elif mode == "line":
            xyz[:, 1:] = 0
        arrays.append((mode, xyz))
    if fixture_path:
        with np.load(fixture_path, allow_pickle=False) as data:
            for name in data.files:
                xyz = data[name]
                if xyz.shape != (points, 3) or xyz.dtype != np.float32 or not np.isfinite(xyz).all():
                    raise ValueError("Export fixtures must be finite normalized float32 (N,3)")
                arrays.append((name, xyz.copy()))
    return arrays


def qualify_model(predictor, model_path, fixture_path=None, atol=1e-5, rtol=1e-4):
    import onnx
    import onnxruntime as ort

    if atol != 1e-5 or rtol != 1e-4:
        raise ValueError("Frozen qualification tolerances are atol=1e-5, rtol=1e-4")
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp, out = session.get_inputs(), session.get_outputs()
    if len(inp) != 1 or len(out) != 1 or inp[0].name != "xyz" or out[0].name != "logits":
        raise ValueError("Unexpected ONNX I/O")
    if inp[0].shape != [1, predictor.input_points, 3] or inp[0].type != "tensor(float)":
        raise ValueError("Only fixed batch 1, checkpoint point count and float32 are supported")
    cases = []
    for name, xyz in qualification_inputs(predictor.input_points, fixture_path):
        expected = predictor.predict(xyz).logits
        actual = session.run(["logits"], {"xyz": xyz[None]})[0]
        if actual.shape != (1, len(xyz), 2) or actual.dtype != np.float32 or not np.isfinite(actual).all():
            raise ValueError("Invalid exported logits")
        difference = np.abs(expected.astype(float) - actual[0].astype(float))
        agreement = float(np.mean(expected.argmax(1) == actual[0].argmax(1)))
        cases.append({"name": name, "input_sha256": __import__("hashlib").sha256(xyz.tobytes()).hexdigest(),
                      "max_abs_logit_difference": float(difference.max()),
                      "mean_abs_logit_difference": float(difference.mean()),
                      "label_agreement": agreement,
                      "logits_within_tolerance": bool(np.allclose(actual[0], expected, atol=atol, rtol=rtol))})
    passed = all(c["label_agreement"] == 1. and c["logits_within_tolerance"] for c in cases)
    return {"schema_version": 1, "status": "QUALIFIED" if passed else "NOT QUALIFIED",
            "scope": "fixed B=1; synthetic cases plus explicitly supplied normalized fixtures; not semantic accuracy qualification",
            "shape": [1, predictor.input_points, 3], "opsets": {x.domain: x.version for x in model.opset_import},
            "atol": atol, "rtol": rtol, "cases": cases,
            "model_sha256": sha256_file(model_path), "semantic_model": predictor.metadata,
            "fixture_sha256": sha256_file(fixture_path) if fixture_path else None,
            "provenance": provenance()}


def export_pointnet2(predictor, output_dir, fixture_path=None):
    import torch

    out = Path(output_dir)
    if out.exists():
        raise ValueError("Export output directory must not exist")
    out.mkdir(parents=True)
    path = out / "model.onnx"
    report = {"schema_version": 1, "status": "NOT QUALIFIED", "requested_opset": 18,
              "shape": [1, predictor.input_points, 3], "semantic_model": predictor.metadata,
              "provenance": provenance()}
    try:
        xyz = qualification_inputs(predictor.input_points)[0][1]
        torch.onnx.export(predictor.model, (torch.from_numpy(xyz)[None].to(predictor.device),),
                          str(path), input_names=["xyz"], output_names=["logits"],
                          opset_version=18, dynamo=True, external_data=False)
        report = qualify_model(predictor, path, fixture_path)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, blocker=str(exc),
                      policy="No validation removal, tracing approximation, custom op, or geometry rewrite is applied")
    write_json(out / "qualification.json", report)
    return report


def load_qualification(model_path, report_path):
    report = json.loads(Path(report_path).read_text())
    cases = report.get("cases", [])
    if (report.get("schema_version") != 1 or report.get("status") != "QUALIFIED"
            or len(cases) < 3 or report.get("atol") != 1e-5 or report.get("rtol") != 1e-4
            or any(c.get("label_agreement") != 1. or not c.get("logits_within_tolerance") for c in cases)
            or report.get("model_sha256") != sha256_file(model_path)):
        raise ExportNotQualified("ONNX backend requires a matching, passing numerical qualification report")
    return report
