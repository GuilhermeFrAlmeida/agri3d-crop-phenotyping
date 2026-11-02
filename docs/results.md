# Results

## Evaluation scope

The historical model-comparison table below is **validation-only**, using deterministic 10,000 sampled points per plant on the frozen plant-identity split. The separate val78 section reports sampled and projected full-cloud metrics; do not conflate these scopes. The 78-plant test split remains untouched. Current neural runs use one seed; no statistical significance claim is made.

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

## Completed val78 evaluation

**Current engineering status: QUALIFIED WITH LIMITATIONS.** The frozen configuration completed **78 validation plants × two semantic modes = 156 successful executions**, with **zero failures**. No test evaluation, retraining or instance-parameter retuning is represented by these results.

The source is the [current-stage technical report](reports/Agri3D_Current_Stage_Technical_Report.pdf) and its approved aggregate validation summary. Raw run bundles, checkpoint bytes and manifests remain external; these results were not recomputed for the documentation release.

### Semantic projection

| Metric | Sampled 10k | Projected 100k |
|---|---:|---:|
| Accuracy | 0.93585 | 0.93676 |
| mIoU | 0.83662 | 0.83967 |
| Leaf IoU | 0.92025 | 0.92111 |
| Stalk IoU | 0.75299 | 0.75822 |
| Stalk precision | 0.85214 | 0.85564 |
| Stalk recall | 0.86615 | 0.86944 |

The existing deterministic nearest-neighbor projection preserves aggregate semantic quality under this frozen experiment. The small aggregate increase is not evidence that projection intrinsically improves model generalization. The selected training-run record (mIoU 0.836603) and this later sampled validation record (0.83662) are retained separately. Their small cross-record difference is not silently rounded away or assigned an unverified cause; exact reproduction requires the corresponding versions and external artifacts.

### Structural diagnostics

| Diagnostic | Val78 result |
|---|---:|
| Oracle-semantic mean accepted components | 15.94 |
| Predicted-semantic mean accepted components | 17.92 |
| Reported mean paired predicted − oracle gap | +1.99 |
| Plants with predicted > oracle | 63 |
| Plants with predicted = oracle | 7 |
| Plants with predicted < oracle | 8 |

Means are rounded separately; the reported paired gap uses the underlying observations rather than subtracting the displayed rounded means. **Component counts are not verified biological leaf counts.** The association is consistent with semantic errors propagating as additional structural fragmentation, but these diagnostics do not establish a causal decomposition or instance accuracy.

The graph was frozen **before** val78 at radius 0.020 m, maximum 32 neighbors, minimum 100 points and attachment radius 0.03 m. [Phase 4A](phase4a_instance_diagnostics.md) preserves the development/qualification evidence. Its five plants overlap val78, and the validation set has already supported model selection: the evaluation is not an independent final holdout.

## Qualification matrix

| Component | Current status |
|---|---|
| Run 1 semantic inference | Qualified on validation |
| 10k → 100k semantic projection | Qualified on validation |
| End-to-end val78 execution | Qualified; 156 successful plant/mode executions |
| Artifact/provenance pipeline | Qualified at the evaluated engineering scope |
| Plant height | Computationally qualified; no biological reference-accuracy claim |
| Instance extraction | Structurally qualified; touching-leaf merges/fragments remain possible |
| Biological leaf count | Not qualified |
| Leaf-length estimates | Geometric proxies |
| Canopy-volume estimates | Geometric proxies |
| Biological phenotype accuracy | Not qualified |
| ONNX backend | Not qualified |

The implemented outputs include full-cloud height, structural component count, PCA and graph-geodesic length proxies, convex-hull and voxel-occupancy volumes, and stalk descriptors. Qualified execution does not validate anatomical identity or biological measurements. See [limitations](limitations.md) and the [pipeline guide](phases4_5_phenotyping_pipeline.md).

## Evaluation discipline

Val78 was an evaluation run, not a tuning run. Report discovered failure modes without retrospectively optimizing parameters against those same 78 plants. New algorithmic revisions require a separately stated development protocol and experiment. The test split remains untouched.
