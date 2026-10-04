# Phase 2 audit and Phase 3 readiness

Repository reviewed at main commit 64c0357: only README and LICENSE. Source was extracted from notebook writefile cells, not from an existing repo implementation.

| Area | Finding | Change |
|---|---|---|
| Ingestion | RGB automatically interpreted as annotation; geometry validation never called | Geometry/property-only reader; nonempty shape, dtype and attribute alignment checks; original indices |
| Label decoding | Largest RGB-group Z span used as stalk; max(label)>1 remapping ambiguous | Explicit verified palette or integer mapping; unknown labels fail; no heuristic ground truth |
| Preprocessing | Mutable data; empty cloud crashes; voxelization discarded all attributes | Copy-based selections, aligned fields, empty rejection, original voxel representatives |
| Normalization | Repeat normalization overwrote inverse metadata | Single normalization guard, float64 statistics, centroid/scale restoration, degenerate cloud handling |
| Sampling | Missing | Random, voxel and FPS exact-N sampling; no replacement until required; validity mask and original indices |
| Dataset split | Missing; unsorted first-five evaluation | Explicit metadata groups, deterministic group splits, manifest leakage/duplicate validation |
| Dataset | Missing | PyTorch Dataset, XYZ-only features, deterministic per-sample RNG, train epoch resampling, stable validation/test |
| Evaluation | No precision/recall/F1; per-cloud metrics only; invalid shapes could broadcast | Dataset-wide confusion matrix, strict shape/ID checks, masks, per-class metrics, explicit absent-class policy |
| Baseline | Normal orientation unnecessary and could fail on small clouds; fixed radii used on normalized units | Sign-invariant normals; baseline evaluated in meters; single manifest-driven command |
| Downloads | First matching archive and unpinned source | Exact archive name, required source revision, archive path validation |

## Remaining gates before training
1. Retrieve and verify official supplementary Table S1 RGB encoding, compare it against observed PLY colors, and save a versioned mapping. The paper states a consistent palette but the inspected sources do not give numerical RGB values. Current code refuses to guess.
2. Supply dataset metadata, choose plant/genotype grouping, and save a reviewed split manifest including all resolutions/repeats. No real split was generated without that metadata.
3. Run label counts, class balance, coordinate-unit and Z-axis checks on real annotated PLY files. No dataset was attached here.
4. Rerun baseline on the frozen validation set. Old notebook mIoU measured against heuristic labels is not valid evidence of segmentation quality.
5. Review SOR's effect on thin leaves/stalks before enabling it for training; Dataset defaults to finite-point filtering only.
6. Benchmark point counts and sampling cost on Colab. FPS is correctness-first and not a scalable GPU implementation.

Optional visualization, mesh and phenotype helpers migrated from the notebook; they are not the training path. Canopy convex hull is a proxy, not measured volume. Leaf count/instances and production API/CLI remain later phases. The notebook Dockerfile pointed to a nonexistent API. It is replaced by a CPU baseline image; container build has not been validated here. Fixed sampled-point metrics are not a full-cloud score; full coverage inference/projection is a later requirement.

Tests use synthetic ASCII/binary PLY fixtures, alignment checks, exact-N padding, duplicate-coordinate FPS, inverse normalization, explicit decoding, grouped leakage, known confusion matrices, and multiprocess DataLoader integration. Real-data quality and GPU throughput remain unvalidated.

## Validation in this workspace
- `python -m pytest -q`: 22 passed, 1 skipped.
- Multiprocess tensor DataLoader test skipped because the workspace forbids Unix sockets. It remains enabled in CI where sockets are available.
- Editable package installation succeeded; Python compilation and baseline CLI help succeeded.
- No real-data run, GPU benchmark or Docker build performed.
