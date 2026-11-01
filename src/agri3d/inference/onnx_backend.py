"""ONNX runtime adapter gated by an explicit parity qualification artifact."""
import numpy as np

from .export import load_qualification
from .predictor import SemanticPrediction, validate_xyz


class OnnxSemanticPredictor:
    def __init__(self, model_path, qualification_path):
        import onnxruntime as ort
        report = load_qualification(model_path, qualification_path)
        self.input_points = report["shape"][1]
        self.metadata = {**report["semantic_model"], "backend": "onnxruntime",
                         "device": "cpu", "onnx_sha256": report["model_sha256"]}
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        if self.session.get_inputs()[0].shape != [1, self.input_points, 3]:
            raise ValueError("ONNX input shape differs from qualified shape")

    def predict(self, xyz, features=None):
        xyz = validate_xyz(xyz)
        if features is not None or xyz.dtype != np.float32 or xyz.shape != (self.input_points, 3):
            raise ValueError("ONNX frozen backend requires fixed-shape float32 XYZ only")
        logits = self.session.run(["logits"], {"xyz": xyz[None]})[0][0]
        exp = np.exp(logits - logits.max(1, keepdims=True))
        p = exp / exp.sum(1, keepdims=True)
        return SemanticPrediction(logits.argmax(1), logits, p, np.arange(len(xyz)), self.metadata).validate(len(xyz))
