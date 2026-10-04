import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from agri3d.io.ply import PointCloudData

class PointCloudVisualizer:
    """Creates headless, interactive side-by-side HTML comparisons for Ground Truth vs Predictions."""

    @staticmethod
    def render_side_by_side(pcd: PointCloudData, pred_labels: np.ndarray, gt_labels: np.ndarray = None, save_path: str = None):
        """Generates a 3D Plotly side-by-side rendering comparing Ground Truth and Predicted Segmentation."""

        # Sub-sample cloud for smooth browser-based 3D HTML rendering if cloud exceeds 20k points
        num_points = pcd.xyz.shape[0]
        if num_points > 20000:
            sample_idx = np.random.choice(num_points, 20000, replace=False)
            xyz = pcd.xyz[sample_idx]
            preds = pred_labels[sample_idx]
            gts = gt_labels[sample_idx] if gt_labels is not None else None
        else:
            xyz = pcd.xyz
            preds = pred_labels
            gts = gt_labels

        has_gt = gts is not None
        cols = 2 if has_gt else 1
        subplot_titles = ("Ground Truth (Leaf vs Stalk)", "Prediction (Leaf vs Stalk)") if has_gt else ("Prediction",)

        fig = make_subplots(
            rows=1, cols=cols,
            specs=[[{'type': 'scene'}] * cols],
            subplot_titles=subplot_titles
        )

        color_map = {0: '#2ca02c', 1: '#8c564b'} # 0: Green (Leaf), 1: Brown (Stalk)

        # Helper to attach 3D scatter traces
        def add_cloud_trace(fig_obj, coords, labels, col_idx):
            for cls_val, cls_name in zip([0, 1], ["Leaf", "Stalk"]):
                mask = (labels == cls_val)
                if np.sum(mask) == 0:
                    continue
                fig_obj.add_trace(
                    go.Scatter3d(
                        x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
                        mode='markers',
                        marker=dict(size=1.8, color=color_map[cls_val], opacity=0.8),
                        name=cls_name,
                        showlegend=(col_idx == 1)
                    ),
                    row=1, col=col_idx
                )

        if has_gt:
            add_cloud_trace(fig, xyz, gts, col_idx=1)
            add_cloud_trace(fig, xyz, preds, col_idx=2)
        else:
            add_cloud_trace(fig, xyz, preds, col_idx=1)

        fig.update_layout(
            title_text="Agri3D — Baseline Segmentation Evaluation",
            height=700,
            width=1300 if has_gt else 700,
            template="plotly_dark"
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)

        return fig