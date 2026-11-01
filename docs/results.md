# Results

## Evaluation scope

All values below are **validation-only** on the frozen plant-identity split and the same deterministic 10,000 sampled points per plant. These are not full-cloud metrics. The 78-plant test split remains untouched. Current neural runs use one seed; no statistical significance claim is made.

## Selected-checkpoint comparison

| Experiment | Selected epoch | Val accuracy | Val mIoU | Leaf IoU | Stalk IoU | Stalk precision | Stalk recall | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Geometric baseline | — | — | 0.6229862 | 0.8173371 | 0.4286353 | — | — | Reference |
| Run 1: PointNet++ XYZ | 76 | 0.9358397 | **0.8366027** | **0.9202424** | **0.7529630** | 0.8521421 | 0.8661208 | Champion |
| Run 2: augmentation | 93 | 0.9347205 | 0.8295737 | 0.9198950 | 0.7392524 | 0.8828109 | 0.8196907 | Rejected |
| Run 3: scheduler | 99 | Not recorded here | 0.8317225588 | 0.918582 | 0.744863 | Not recorded here | Not recorded here | Rejected |
| Run 4: estimated normals | Not supplied | 0.9263846 | 0.8173358 | 0.9085305 | 0.7261411 | 0.8194030 | 0.8644969 | Rejected |

Run 1's logged training mIoU at its selected epoch was 0.924027. Run 2 increased stalk precision while reducing stalk recall; its leaf/stalk IoUs and aggregate mIoU were lower than Run 1. Run 3 completed but also remained below Run 1. Run 4 completed and was rejected; Run 1 remains the frozen semantic baseline. Run 4 selected epoch was not supplied and is not inferred.

## What did not work

- **Run 2 augmentation did not surpass Run 1.** Higher stalk precision came with lower recall and lower stalk IoU, so the augmentation policy was not retained as the preferred model.
- **Run 3 scheduling did not surpass Run 1.** ReduceLROnPlateau on validation mIoU was evaluated as a controlled change; its best mIoU was 0.8317225588 at epoch 99, below 0.8366027.

These runs narrow the current direction and keep the comparison accountable. They do not establish that augmentation or scheduling cannot help under other settings, and they do not justify claims about test generalization.

## Validation and artifact notes

Checkpoint selection uses global validation mIoU, with the existing validation-loss tie-break. The reported class labels are class 0 = leaf and class 1 = stalk. Runs 1–3 used XYZ-only inputs. Run 4 used estimated normals as separate features and did not surpass Run 1. Do not combine last-epoch training metrics with best-checkpoint validation metrics.

See the [Phase 3A validation-analysis guide](phase3a_validation_analysis.md) for per-plant and error-mode metrics, and the [experiment registry](experiments.md) for decision context. No test-set evaluation has been run.

## Downstream system status

Phases 4/5 are **IMPLEMENTED — NOT YET EXPERIMENTALLY QUALIFIED**. The validation harness separates 10k sampled neural metrics from full-cloud projected metrics, and oracle-semantic from predicted-semantic structure/trait outputs. No real-data projection, instance or phenotype result is reported yet. Synthetic execution tests are not biological accuracy evidence. See [the pipeline protocol](phases4_5_phenotyping_pipeline.md).
