# Architecture

Agri3D currently implements the data-to-semantic-segmentation path. Instance reconstruction and phenotypic measurements are downstream directions, not a completed end-to-end feature.

## High-level system

```mermaid
flowchart TD
    A[MaizeField3D segmented PLY] --> B[PLY ingestion]
    B --> C[Finite filtering and aligned cleaning]
    C --> D[RGB semantic decode: supervision only]
    D --> E[Full-cloud normalization]
    E --> F[Deterministic 10k sampling]
    F --> G[Optional estimated normals: Run 4]
    G --> H[PointNet++ SSG]
    H --> I[Leaf / stalk semantic labels]
    I --> J[Downstream phenotyping: planned]
```

For Run 4, normals are estimated on cleaned full-resolution XYZ before normalization/sampling. RGB is never passed as model input. See [dataset](dataset.md) and [Phase 3D](phase3d_run4_normals.md).

## Training and checkpoint selection

```mermaid
flowchart LR
    A[Seeded frozen manifest] --> B[Train/val-only Dataset]
    B --> C[Clean, decode, normalize, sample]
    C --> D[DataLoader]
    D --> E[PointNet++ and masked unweighted loss]
    E --> F[Optimizer update]
    F --> G[Validation metrics]
    G --> H[Best checkpoint by validation mIoU]
    H --> I[Checkpoint and JSON artifact persistence]
    G --> D
```

The manifest records plant-level train/validation/test assignments, but training and validation entry points load only train and validation records. The test split is not evaluated. The optimizer is Adam; regular checkpoints preserve model/optimizer/RNG/loader/history state for recovery. Runs 1–4 use 100 epochs and batch size 2; Run 3 alone uses validation-mIoU ReduceLROnPlateau, while Run 4 uses estimated normals and no scheduler.

## PointNet++ SSG

```mermaid
flowchart TD
    A[10,000 points: XYZ geometry] --> B[SA1: 1,024 centroids, r=0.1, k=32]
    B --> C[SA2: 256 centroids, r=0.2, k=32]
    C --> D[SA3: 64 centroids, r=0.4, k=32]
    D --> E[SA4: 16 centroids, r=0.8, k=32]
    E --> F[Feature propagation: 16 → 64 → 256 → 1,024 → 10,000]
    F --> G[Per-point binary logits]
```

Geometry operations (farthest-point sampling, radius neighborhoods and interpolation coordinates) always use XYZ. Runs 1–3 have no additional point features. Run 4 supplies estimated normals as a separate 3-channel feature tensor; they do not alter geometric distance calculations. All stages, centroid counts, radii, neighborhood sizes and MLP widths remain fixed. RGB is label supervision only.

## Experiment and reproducibility flow

```mermaid
flowchart LR
    A[Scientific config] --> B[Frozen manifest]
    B --> C[Training run]
    C --> D[Contract-checked checkpoint]
    D --> E[Validation]
    E --> F[Analysis]
    F --> G[Cross-run comparison]
```

Runtime options such as workers, pinning and cache location are recorded separately from scientific choices. See [reproducibility](reproducibility.md) and [experiments](experiments.md).
