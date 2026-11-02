# Dataset

## Source and subset

Experiments use the segmented 100k-point subset of [MaizeField3D](https://huggingface.co/datasets/BGLab/MaizeField3D), an external field-grown maize point-cloud dataset. The upstream card lists 520 segmented plant PLY files at 100,000 points per plant. This repository uses those 520 annotated plants; dataset files and experiment artifacts are not stored here.

Each relevant PLY contains XYZ coordinates and RGB annotation colors. In this segmented export, RGB is semantic supervision:

- `(0, 0, 0)` → stalk, class `1`
- non-black RGB → leaf, class `0`

RGB is decoded for labels only. It is never a model feature because it directly defines the target.

## Frozen split

| Split | Plants |
|---|---:|
| Train | 364 |
| Validation | 78 |
| Test | 78 |
| **Total** | **520** |

The split is frozen, seed 42, and grouped by plant identity. It is not genotype-held-out. Use the existing manifest unchanged across runs. Public instructions use a placeholder for the manifest because it is supplied outside the repository.

Across the segmented subset, stalk points comprise approximately 22.63% of points. This prevalence motivates per-class reporting but does not imply class weighting; current neural runs use unweighted masked cross-entropy.

## Preprocessing and sampling contract

For XYZ-only runs, the sequence is PLY read → finite-point filtering with aligned attributes → RGB label decoding → centroid/unit-sphere normalization of the cleaned full cloud → deterministic fixed-size sampling. Neural experiments use 10,000 points per plant. Sampled XYZ, labels, valid mask and source indices stay aligned. Padding, if present, is excluded from loss and metrics.

Run 4 estimates normals from the cleaned full-resolution XYZ cloud before normalization and sampling, then carries sampled normals as separate features. XYZ normalization does not transform the direction vectors. The exact policy is in [Phase 3D](phase3d_run4_normals.md).

## Test isolation

The split manifest contains a test assignment, but current training/validation/analysis tools do not expose a test evaluation path. Run selection uses validation mIoU. **The test set remains untouched and no test metric is reported.**

## Dataset, code and artifacts

- **Dataset:** obtained separately from the upstream distribution and governed by its own CC BY-NC 4.0 terms.
- **Repository code:** source and tests licensed under Apache 2.0.
- **Experiment artifacts:** manifests, checkpoints, logs and predictions are local/user-managed outputs; they are not committed or distributed here.

See [dataset and licensing](dataset_and_licensing.md) for attribution and reuse details. Do not commit the dataset, checkpoint files, or private manifests.

## Completed validation scope

The frozen downstream val78 evaluation is complete (156 successful plant/mode executions, zero failures). This is engineering/structural qualification, not verified biological instance or phenotype accuracy. [Results](results.md) distinguish historical 10k model selection from the later sampled/projected evaluation. The test split remains untouched, and the validation execution is not a tuning opportunity.
