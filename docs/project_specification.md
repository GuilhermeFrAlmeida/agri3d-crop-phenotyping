> **Historical target specification:** this initial planning document describes proposed capabilities, not the current implementation. The current status is in the [roadmap](roadmap.md), and current implemented architecture is in [architecture](architecture.md). REST API deployment, full instance reconstruction, and production latency in this file are not completed or measured.

# Agri3D System Specification & Engineering Contract

## 1. System Objective
Provide an industrial-grade, cloud-deployable 3D perception pipeline that ingests raw agricultural LiDAR point clouds, performs 3D deep-learning segmentation (stalk vs. leaf), reconstructs individual plant instances, and outputs quantifiable phenotypic metrics.

## 2. Input Data Contract

### Primary Input: Point Cloud Files (.ply)
- **Coordinate System:** Right-handed 3D Cartesian coordinates $(X, Y, Z)$ measured in meters ($m$).
- **Required Fields:**
  - `x` (`float32`): Radial offset along $X$-axis.
  - `y` (`float32`): Depth offset along $Y$-axis.
  - `z` (`float32`): Height offset along $Z$-axis.
- **Optional/Ignored Fields (Ablation Candidates):**
  - `red`, `green`, `blue` (`uint8`): Color channels.
  - `nx`, `ny`, `nz` (`float32`): Surface normals.
  - `intensity` (`float32`): LiDAR return intensity.
- **Data Standard:** Model training strictly defaults to $N \times 3$ input shape (geometric features only) to ensure model generalization independent of lighting or surface color.


## 3. Data Flow & Processing Pipeline Contract

```
   [ Input: .ply ]
          │
          ▼
[ Quality Processing ] ──► (Filter NaN/Inf, Outlier Removal, Centroid Shift)
          │
          ▼
[ Fixed-Size Sampling ] ──► (N = 10k / 50k / 100k points via FPS/Voxel Grid)
          │
          ▼
[ Deep Learning Model ] ──► (PointNet++ / Point Transformer)
          │
          ▼
[ Semantic Labels Array ] ──► (0: Leaf, 1: Stalk)
          │
          ▼
[ Instance Reconstruction ] ──► (Leaf Clustering / Connected Components)
          │
          ▼
[ Phenotypic Measurement ] ──► (Height, Leaf Count, Canopy Volume)
          │
          ▼
[ Output JSON/CSV ]

```

## 4. Output Specification Contract

### REST API / CLI JSON Schema Response
All predictions produce a structured response conforming to this schema:

```json
{
  "status": "success",
  "metadata": {
    "plant_id": "MaizeField3D_001042",
    "raw_point_count": 100000,
    "processed_point_count": 10240,
    "processing_time_ms": 142.8
  },
  "phenotypic_metrics": {
    "plant_height_m": 2.145,
    "leaf_count": 11,
    "stalk_height_m": 1.920,
    "canopy_volume_m3": 0.387,
    "canopy_bounding_box_m": {
      "dx": 0.85,
      "dy": 0.92,
      "dz": 2.145
    }
  },
  "segmentation_summary": {
    "stalk_points": 1420,
    "leaf_points": 8820,
    "stalk_ratio": 0.1387
  }
}

```

## 5. Performance Targets & Validation

| Metric | Target | Description |
| --- | --- | --- |
| **Segmentation mIoU** | $\ge 0.82$ | Mean Intersection over Union across stalk and leaf classes |
| **Leaf Class IoU** | $\ge 0.85$ | IoU specifically for canopy leaf structures |
| **Height Precision** | $\pm 0.03\text{ m}$ | Difference relative to manual annotation bounds |
| **Leaf Count Accuracy** | $\pm 1\text{ leaf}$ | Instance segmentation error margin on test set |
| **Latency Target** | $< 250\text{ ms}$ | Inference + phenotyping runtime on single GPU (e.g., T4/V100) |

---

## 6. Container & Cloud Deployment Strategy

* **Containerized Execution:** Fully encapsulated build via multi-stage `Dockerfile`.
* **Runtime Environment:** Python 3.10+, PyTorch CUDA acceleration, Open3D headless execution.
* **Config Management:** Centralized configuration using YAML files (`configs/*.yaml`).
* **Data Persistence:** Modular separation between source code (`/src`) and local mount volumes (`/data`, `/artifacts`).