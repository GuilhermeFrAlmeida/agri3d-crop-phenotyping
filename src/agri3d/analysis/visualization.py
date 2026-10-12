"""Optional standalone Plotly diagnostics using identical points and scene bounds."""

import html
from pathlib import Path

import numpy as np

SEMANTIC = ((0, "Leaf", "#34a853"), (1, "Stalk", "#3367d6"))
ERRORS = (
    (0, "Correct leaf", "#34a853"),
    (1, "Correct stalk", "#3367d6"),
    (2, "Leaf → stalk (false positive)", "#ff9800"),
    (3, "Stalk → leaf (false negative)", "#d93025"),
)


def error_categories(targets, predictions):
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if targets.shape != predictions.shape or targets.ndim != 1:
        raise ValueError("Expected matching one-dimensional labels")
    if not np.isin(targets, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Expected binary labels")
    return np.where(targets == predictions, targets, np.where(targets == 0, 2, 3))


def save_views(path, sample_id, points, targets, predictions):
    """Three side-by-side interactive scenes, shared camera and fixed XYZ ranges."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "Visualization requires pip install 'agri3d[analysis]'"
        ) from exc
    points = np.asarray(points)
    categories = error_categories(targets, predictions)
    if (
        points.shape != (len(categories), 3)
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("Expected finite nonempty (N,3) points aligned with labels")
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}] * 3],
        subplot_titles=("Ground truth", "Prediction", "Error map"),
    )
    for column, labels, palette in (
        (1, targets, SEMANTIC),
        (2, predictions, SEMANTIC),
        (3, categories, ERRORS),
    ):
        for value, name, color in palette:
            xyz = points[np.asarray(labels) == value]
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    mode="markers",
                    marker={"size": 2, "color": color},
                    name=name,
                    legendgroup=name,
                    showlegend=column != 2,
                ),
                row=1,
                col=column,
            )
    axes = {}
    for index, axis in enumerate(("xaxis", "yaxis", "zaxis")):
        low, high = points[:, index].min(), points[:, index].max()
        margin = max(float(high - low) * 0.03, 0.001)
        axes[axis] = {
            "range": [float(low - margin), float(high + margin)],
            "title": axis[0].upper(),
        }
    scene = {
        **axes,
        "aspectmode": "data",
        "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1}},
    }
    fig.update_layout(
        title=html.escape(str(sample_id)),
        scene=scene,
        scene2=scene,
        scene3=scene,
        height=650,
        margin={"l": 0, "r": 0, "b": 0, "t": 70},
    )
    # Keep all cameras synchronized as the reader rotates/zooms any panel.
    synchronize = """
    const plot = document.getElementById('{plot_id}');
    let syncing = false;
    plot.on('plotly_relayout', function(event) {
      if (syncing) return;
      const key = Object.keys(event).find(k => /^scene[23]?\\.camera$/.test(k));
      if (!key) return;
      const updates = {};
      ['scene', 'scene2', 'scene3'].forEach(s => { updates[s + '.camera'] = event[key]; });
      syncing = true;
      Plotly.relayout(plot, updates).then(() => { syncing = false; });
    });
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=True, full_html=True, post_script=synchronize)
