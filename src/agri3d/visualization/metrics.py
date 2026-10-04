import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class EvaluationPlotter:
    """Generates analytical charts for segmentation performance and phenotypic trait metrics."""

    def __init__(self, style: str = "darkgrid"):
        sns.set_theme(style=style)

    def plot_performance_metrics(self, df_metrics: pd.DataFrame, save_path: str):
        """Plots mIoU, Stalk IoU, Leaf IoU, and Processing Latency across evaluated samples."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Chart 1: Segmentation IoU Metrics Per Sample
        iou_cols = [c for c in ['mIoU', 'class_0_iou', 'class_1_iou'] if c in df_metrics.columns]
        rename_dict = {'mIoU': 'mIoU', 'class_0_iou': 'Leaf IoU', 'class_1_iou': 'Stalk IoU'}

        df_iou = df_metrics.set_index('plant_id')[iou_cols].rename(columns=rename_dict)
        df_iou.plot(kind='bar', ax=axes[0], colormap='viridis', width=0.8)
        axes[0].set_title('Segmentation Performance (IoU per Sample)')
        axes[0].set_ylabel('Score')
        axes[0].set_ylim(0, 1.0)
        axes[0].tick_params(axis='x', rotation=45)

        # Chart 2: Metric Distribution (Boxplot)
        sns.boxplot(data=df_iou, ax=axes[1], palette='Set2')
        axes[1].set_title('Segmentation Score Distributions')
        axes[1].set_ylabel('Score')
        axes[1].set_ylim(0, 1.0)

        # Chart 3: Processing Latency Target Breakdown
        if 'processing_time_ms' in df_metrics.columns:
            sns.barplot(data=df_metrics, x='plant_id', y='processing_time_ms', ax=axes[2], palette='magma')
            axes[2].axhline(250, color='red', linestyle='--', label='Latency Target (250ms)')
            axes[2].set_title('Inference & Phenotyping Latency')
            axes[2].set_ylabel('Time (ms)')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].legend()

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_phenotypic_metrics(self, df_pheno: pd.DataFrame, save_path: str):
        """Plots physical crop trait distributions: Height vs Canopy Volume and Stalk vs Leaf Ratios."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Chart 1: Plant Height vs Canopy Volume Scatter
        sns.scatterplot(
            data=df_pheno, x='plant_height_m', y='canopy_volume_m3',
            hue='stalk_ratio', size='processed_point_count', sizes=(40, 200), ax=axes[0], palette='crest'
        )
        axes[0].set_title('Plant Height vs. Canopy Volume')
        axes[0].set_xlabel('Plant Height (m)')
        axes[0].set_ylabel('Canopy Volume (m³)')

        # Chart 2: Point Distribution (Leaf vs Stalk Point Counts)
        df_counts = df_pheno.set_index('plant_id')[['leaf_points', 'stalk_points']]
        df_counts.plot(kind='bar', stacked=True, ax=axes[1], color=['#2ca02c', '#8c564b'])
        axes[1].set_title('Point Breakdown per Plant Component')
        axes[1].set_ylabel('Point Count')
        axes[1].tick_params(axis='x', rotation=45)

        # Chart 3: Stalk Height Distribution Across Batch
        sns.histplot(data=df_pheno, x='stalk_height_m', kde=True, ax=axes[2], color='teal')
        axes[2].set_title('Stalk Height Distribution')
        axes[2].set_xlabel('Stalk Height (m)')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()