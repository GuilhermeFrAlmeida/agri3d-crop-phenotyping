# Model interoperability

**Canonical validated format: PyTorch regular-run `.pt`. Current ONNX status: NOT QUALIFIED (blocked at capture).** No trained model/export is committed. This feature does not retrain, approximate geometry or change PointNet++.

## Observed blocker

An actual modern export attempt at fixed `[1,10000,3]`, with the official Run 1 architecture and synthetic weights/input, failed with `TorchExportError` during `torch.export` capture. The existing forward validation `if not torch.isfinite(xyz).all()` requires a data-dependent tensor boolean that symbolic capture cannot guard. This occurred before ONNX model generation. FPS's Python loop and query/interpolation control flow are further potential export constraints, not additional blockers demonstrated by this attempt. No legacy tracing fallback, validation removal, custom operator or approximate FPS is silently substituted.

Observed environment: PyTorch 2.15.0+cpu, ONNX 1.23.2, ONNX Runtime 1.31.1, onnxscript 0.7.2. Requested opset 18, fixed batch size 1 and 10k XYZ-only points. Export libraries are optional (`.[interoperability]`). Future library versions may behave differently; every artifact must qualify independently.

## Export and qualification contract

The interface remains backend-neutral. `scripts/export_pointnet2.py` attempts modern `torch.onnx.export(..., dynamo=True)` and always records a status/error report after a valid checkpoint load. The output directory must be new. Export success alone is insufficient: qualification runs `onnx.checker` with full checking, creates an ONNX Runtime CPU session, checks fixed I/O names/shapes/dtypes and compares logits with canonical PyTorch.

Three deterministic synthetic normalized input types (volume, plane and line) are required. An optional NPZ supplies additional normalized float32 `(10000,3)` arrays from representative development plants; no RGB input is supported. A qualifying artifact must produce **100% label agreement in every case** and `np.allclose` logit agreement with fixed `atol=1e-5, rtol=1e-4`. Each case records max/mean absolute logit difference, label agreement and input hash. The report records output model hash, checkpoint provenance, I/O shape, actual opset imports, library versions and supplied fixture hash. A real trained checkpoint still requires its own attempt/qualification; random synthetic export tests do not qualify it.

`OnnxSemanticPredictor` only accepts a report marked QUALIFIED with passing cases/tolerances and a matching ONNX SHA256. CPU-only ONNX is exposed; dynamic batch/point counts are not qualified. A report is provenance supplied by the artifact owner, not a cryptographic attestation. PyTorch remains the default even if a future artifact passes.

## Exact commands

```bash
python -m pip install -e '.[interoperability]'
CHECKPOINT="<RUN1_BEST_PT>"
OUT="<NEW_EXPORT_DIR>"
python scripts/export_pointnet2.py --checkpoint "$CHECKPOINT" --output-dir "$OUT"
```

Current expected outcome: exit code 2 and `$OUT/qualification.json` with `NOT QUALIFIED` and the capture error. Do not attempt ONNX inference without a qualified artifact. If a future export succeeds, requalify it explicitly:

```bash
python scripts/validate_model_export.py \
  --checkpoint "$CHECKPOINT" --model "$OUT/model.onnx" \
  --fixtures "<OPTIONAL_NORMALIZED_DEVELOPMENT_FIXTURES_NPZ>" \
  --output "<NEW_REQUALIFICATION_JSON>"
```

Omit `--fixtures` for the three built-in synthetic cases. Keep representative fixture provenance. No model or fixture bytes are added to Git.
