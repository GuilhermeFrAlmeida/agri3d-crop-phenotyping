# Architecture

Agri3D implements semantic training plus a model-neutral inference, full-cloud projection, geometric structure and metric-measurement pipeline. The frozen downstream pipeline completed val78 engineering qualification with limitations (156 successful plant/mode executions, zero failures). Biological instance/trait accuracy remains unqualified. Run 1 is the semantic backend; stronger network comparisons remain planned.

## High-level system

```mermaid
flowchart TD
    A[PLY input in meters] --> B[Finite filter and preserve full metric XYZ]
    B --> C[Full-cloud normalization and deterministic 10k sample]
    C --> D[SemanticPredictor: frozen Run 1 XYZ]
    D --> E[Full-cloud nearest projection]
    E --> F[Geometric leaf instances and metric traits]
    F --> G[Checksummed artifacts and reports]
    B --> H[Validation-only RGB supervision decode]
    H --> I[Separate development evaluation]
    E --> I
```

RGB is never a model feature. The historical Run 4 normal-feature experiment remains implemented but was rejected and is not the frozen downstream backend. See [dataset](dataset.md), [Phase 3D](phase3d_run4_normals.md), and [Phases 4/5](phases4_5_phenotyping_pipeline.md).

## Training and checkpoint selection

```mermaid
flowchart TD
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
flowchart TD
    A[Scientific config] --> B[Frozen manifest]
    B --> C[Training run]
    C --> D[Contract-checked checkpoint]
    D --> E[Validation]
    E --> F[Analysis]
    F --> G[Cross-run comparison]
```

Runtime options such as workers, pinning and cache location are recorded separately from scientific choices. See [reproducibility](reproducibility.md) and [experiments](experiments.md).

## Frozen downstream boundary

```mermaid
flowchart TD
    A[InputAdapter and full metric Preprocessor] --> B[SemanticPredictor protocol]
    B --> C[SemanticProjector: metric nearest neighbor]
    C --> D[InstanceExtractor: capped radius graph components]
    D --> E[PhenotypeExtractor: metric geometry only]
    E --> F[ArtifactWriter: JSON, arrays, scalar PLY, HTML]
```

Only the backend adapter imports PointNet++ internals. The CLI composes it with downstream modules; the latter do not inspect the architecture. Clean metric coordinates and normalized 10k model coordinates remain separate. RGB is never a model feature. See [Phases 4/5](phases4_5_phenotyping_pipeline.md) and [interoperability](model_interoperability.md) for contract details. Val78 engineering execution is qualified with limitations; biological phenotype accuracy and ONNX remain unqualified.
