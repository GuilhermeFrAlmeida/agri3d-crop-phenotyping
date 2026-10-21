# Roadmap

Status labels describe implementation state, not a promise of completion dates.

## Completed

- PLY loading, finite-point validation and RGB semantic decoding for the segmented MaizeField3D subset.
- Frozen group-aware plant-identity train/validation/test manifest integration.
- Geometric validation baseline.
- PointNet++ SSG XYZ baseline and fixed-point validation.
- Validation-only per-plant error analysis.
- Controlled augmentation (Run 2) and validation-mIoU scheduler (Run 3) comparisons.
- Phase 3P profiler, bounded loader/cache benchmarks and batch-throughput engineering.
- Run 4 estimated-normal preprocessing/model path, contract checks and normals-compatible tiny-overfit mechanism.

## In progress

- Run 4 deterministic kNN-PCA normals audit and tiny-overfit gate on real data.
- Run 4 full training and validation comparison, pending completion of the gate.

## Planned

- Leaf-instance segmentation and instance reconstruction.
- Phenotypic trait extraction and validation against references.
- Evaluation under genotype-held-out and other robustness splits.
- Cross-dataset validation, including external LiDAR sources.
- Evaluation of stronger 3D architectures under controlled comparisons.
- User-facing CLI/API packaging for an end-to-end workflow.

Planned items have no completed implementation or result claim in the current repository.
