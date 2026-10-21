# Experiment registry

## Shared protocol

The supervised experiments use MaizeField3D segmented 100k, a frozen plant-identity split (364/78/78), seed 42, 10,000 sampled points, PointNet++ SSG, batch size 2, Adam, initial learning rate 0.001, 100 epochs, masked unweighted cross-entropy and model selection by validation mIoU. RGB supplies labels only. No experiment reports test performance. Run 1 is the reference; changes are controlled and compared on the same validation points.

| Experiment | Hypothesis | Controlled change | Outcome | Decision |
|---|---|---|---|---|
| Geometric baseline | Transparent rules can provide a simple reference for leaf/stalk separation. | Geometry rules rather than a learned network. | Validation mIoU 0.622986; leaf IoU 0.817337; stalk IoU 0.428635. | Retain as reference. |
| Run 1 | A PointNet++ model can improve over geometry-only rules using XYZ geometry. | First learned baseline; no augmentation or scheduler. | Best epoch 76; validation mIoU 0.8366027; leaf IoU 0.9202424; stalk IoU 0.7529630. | Current champion and reference for later runs. |
| Run 2 | Mild geometry-preserving augmentation may improve generalization. | Z rotation, isotropic scaling and clipped coordinate jitter only. | Best epoch 93; mIoU 0.8295737; stalk precision rose to 0.8828109, recall fell to 0.8196907, and stalk IoU fell to 0.7392524. | Rejected; aggregate performance below Run 1. |
| Run 3 | Reducing Adam LR when validation mIoU stalls may improve late training generalization. | ReduceLROnPlateau monitored on validation mIoU only. | Best epoch 99; mIoU 0.8317225588; leaf IoU 0.918582; stalk IoU 0.744863. | Rejected; below Run 1. |
| Run 4 | Explicit local surface orientation may help separate stalk and leaf. | XYZ plus deterministic full-cloud kNN-PCA estimated normals only. | Implemented; real-data normal audit/overfit gate and full-run result pending. | In progress; no success claim. |

## Interpretation

The geometric baseline establishes a simple non-learned point of comparison. Run 1 is the current champion because it has the strongest validation mIoU and stalk IoU among the measured neural runs. Run 2 traded recall for precision without improving aggregate validation. Run 3 completed the scheduler intervention, but its best mIoU remained below Run 1. Run 4 returns to the Run 1 policy and tests only an added geometric feature; it is not yet a result.

Runs are single-seed controlled comparisons, not a statistical study. The validation set has already informed model selection and successive hypotheses, so these metrics are not an independent test estimate.

## Exact run contracts

- [Run 1 config and workflow](phase3_pointnet2.md)
- [Run 2 augmentation contract](phase3b_run2_augmentation.md)
- [Run 3 scheduler contract](phase3c_run3_scheduler.md)
- [Run 4 normal-feature contract](phase3d_run4_normals.md)

The separate [Phase 3P performance work](phase3p_training_performance.md) changes runtime instrumentation and optional data plumbing, not these scientific comparisons.
