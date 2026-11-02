"""Execute public notebook wrappers with synthetic inputs, never real data/weights."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from agri3d.inference.predictor import SemanticPrediction

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("0[123]_*.ipynb"))


class SyntheticPredictor:
    input_points = 10000
    metadata = {"backend": "synthetic-notebook-smoke-test", "accuracy_qualified": False}

    def predict(self, xyz, features=None):
        assert features is None
        labels = (np.abs(xyz[:, 0]) < .15).astype(np.int64)
        logits = np.eye(2, dtype=np.float32)[labels] * 4
        probabilities = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        return SemanticPrediction(labels, logits, probabilities, np.arange(len(xyz)), self.metadata)


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
def test_public_notebook_synthetic_execution(notebook, tmp_path, monkeypatch):
    nb = json.loads(notebook.read_text())
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] == 5
    assert len({c["id"] for c in nb["cells"]}) == len(nb["cells"])
    rng = np.random.default_rng(42)
    xyz = rng.uniform([-.2, -.05, 0], [.2, .05, .6], (600, 3)).astype("float32")
    vertex = np.empty(len(xyz), dtype=[(a, "f4") for a in "xyz"] +
                      [(a, "u1") for a in ("red", "green", "blue")])
    for i, a in enumerate("xyz"):
        vertex[a] = xyz[:, i]
    for a in ("red", "green", "blue"):
        vertex[a] = np.where(np.abs(xyz[:, 0]) < .02, 0, 128)
    path = tmp_path / "synthetic.ply"
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))
    monkeypatch.chdir(ROOT)
    namespace = {"_AGRI3D_NOTEBOOK_CONTEXT": {"ply_path": path, "annotated": True,
        "predictor": SyntheticPredictor(), "sample_id": "synthetic",
        "output_dir": tmp_path / "output", "visualize": False}}
    imports = set()
    for cell in nb["cells"]:
        assert not cell.get("attachments")
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [] and cell["execution_count"] is None
        source = "".join(cell["source"])
        tree = ast.parse(source)
        imports.update(n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom))
        exec(compile(tree, str(notebook), "exec"), namespace)
    assert any(name.startswith("agri3d.") for name in imports)
    if notebook.name.startswith("03"):
        report = json.loads((tmp_path / "output" / "phenotype.json").read_text())
        assert report["provenance"]["configuration"]["instances"]["max_neighbors"] == 32
        assert report["semantic_model"]["backend"] == "synthetic-notebook-smoke-test"


def test_exact_curated_notebook_inventory():
    assert len(NOTEBOOKS) == 3
