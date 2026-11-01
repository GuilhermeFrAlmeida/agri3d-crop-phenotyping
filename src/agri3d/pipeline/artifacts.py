"""Atomic plant outputs; one full geometry copy with machine-readable scalar labels."""
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from .provenance import sha256_file


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_structure_ply(path, result):
    count = len(result.cloud.xyz)
    vertices = np.empty(count, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("semantic_label", "i4"), ("instance_id", "i4"), ("source_index", "i4")])
    for i, name in enumerate(("x", "y", "z")):
        vertices[name] = result.cloud.xyz[:, i]
    vertices["semantic_label"] = result.semantic_labels
    vertices["instance_id"] = result.instance_labels
    vertices["source_index"] = result.cloud.attributes["source_index"]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def save_visualization(path, result, max_points=5000):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ids = np.linspace(0, len(result.cloud.xyz) - 1, min(max_points, len(result.cloud.xyz)), dtype=int)
    xyz = result.cloud.xyz[ids]
    panels = [("Input geometry", np.full(len(ids), "#777777")),
              ("Semantic prediction" if result.prediction else "Oracle semantics",
               np.where(result.semantic_labels[ids] == 0, "#2e9d50", "#315bd1")),
              ("Geometric instances", ["#777777" if j < 0 else "#333333" if j == 0
               else f"hsl({(int(j)*137)%360},65%,50%)" for j in result.instance_labels[ids]])]
    if result.target_labels is not None and result.prediction is not None:
        panels.append(("Semantic error", np.where(
            result.semantic_labels[ids] == result.target_labels[ids], "#bbbbbb", "#e44a36")))
    fig = make_subplots(rows=1, cols=len(panels), specs=[[{"type": "scene"}] * len(panels)],
                        subplot_titles=[x[0] for x in panels])
    for i, (_, colors) in enumerate(panels):
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                                  marker={"size": 1.5, "color": colors}, showlegend=False), row=1, col=i+1)
        fig.update_scenes(aspectmode="data", row=1, col=i+1)
    fig.write_html(str(path), include_plotlyjs=True)


def write_plant(output, result, visualize=False):
    output = Path(output)
    if output.exists():
        raise ValueError("Output directory must not exist; artifacts are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp-" + uuid.uuid4().hex)
    temporary.mkdir()
    start = time.perf_counter()
    try:
        np.save(temporary / "semantic_labels.npy", result.semantic_labels, allow_pickle=False)
        np.save(temporary / "instance_labels.npy", result.instance_labels, allow_pickle=False)
        write_structure_ply(temporary / "structure.ply", result)
        if result.prediction is not None:
            np.savez(temporary / "sample_predictions.npz", points=result.sampled.xyz,
                     labels=result.prediction.labels, logits=result.prediction.logits,
                     probabilities=result.prediction.probabilities,
                     source_index=result.sampled.attributes["source_index"],
                     valid_mask=result.sampled.attributes["valid_mask"])
        if visualize:
            save_visualization(temporary / "views.html", result)
        result.report["timing_ms"]["serialization"] = (time.perf_counter() - start) * 1000
        result.report["timing_ms"]["total"] = (time.perf_counter() - result.started) * 1000
        result.report["timing_scope"] = "wall time through payload serialization; excludes final JSON bookkeeping"
        write_json(temporary / "phenotype.json", result.report)
        write_json(temporary / "timing.json", result.report["timing_ms"])
        write_json(temporary / "provenance.json", result.report["provenance"])
        inventory = {p.name: {"size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
                     for p in sorted(temporary.iterdir())}
        write_json(temporary / "artifacts.json", {"schema_version": 1, "files": inventory})
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
