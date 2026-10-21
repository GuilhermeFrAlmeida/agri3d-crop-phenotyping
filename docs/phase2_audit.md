# Phase 2 audit and Phase 3 readiness

> **Historical snapshot:** this file records the early data-foundation phase. Current implementation and experiment status are summarized in the root README and current documentation index.

Repository reviewed at main commit 64c0357: only README and LICENSE. Source was extracted from notebook writefile cells, not from an existing repo implementation.

| Area | Finding | Change |
|---|---|---|
| Ingestion | RGB automatically interpreted as annotation; geometry validation never called | Geometry/property-only reader; nonempty shape, dtype and attribute alignment checks; original indices |
| Label decoding | Largest RGB-group Z span used as stalk; max(label)>1 remapping ambiguous | MaizeField3D black/non-black decoder; explicit mappings retained for other datasets; no heuristic ground truth |
| Preprocessing | Mutable data; empty cloud crashes; voxelization discarded all attributes | Copy-based selections, aligned fields, empty rejection, original voxel representatives |
| Normalization | Repeat normalization overwrote inverse metadata | Single normalization guard, float64 statistics, centroid/scale restoration, degenerate cloud handling |
| Sampling | Missing | Random, voxel and FPS exact-N sampling; no replacement until required; validity mask and original indices |
| Dataset split | Missing; unsorted first-five evaluation | Explicit metadata groups, deterministic group splits, manifest leakage/duplicate validation |
| Dataset | Missing | PyTorch Dataset, XYZ-only features, deterministic per-sample RNG, train epoch resampling, stable validation/test |
| Evaluation | No precision/recall/F1; per-cloud metrics only; invalid shapes could broadcast | Dataset-wide confusion matrix, strict shape/ID checks, masks, per-class metrics, explicit absent-class policy |
| Baseline | Normal orientation unnecessary and could fail on small clouds; fixed radii used on normalized units | Sign-invariant normals; baseline evaluated in meters; single manifest-driven command |
| Downloads | First matching archive and unpinned source | Exact archive name, required source revision, archive path validation |

## Remaining gates before training
1. Completed from recorded real-data Colab validation: binary RGB convention is black=stalk(1), non-black=leaf(0). Exact leaf palette verification is not a Phase 3 gate; original RGB remains available for Phase 4 instance validation.
2. Supply dataset metadata, choose plant/genotype grouping, and save a reviewed split manifest including all resolutions/repeats. No real split was generated without that metadata.
3. Completed in recorded Colab validation: all 520 annotated PLYs contain 100k finite points, meter-consistent scale, vertical Z, and both black/non-black groups. Black-point ratio mean≈0.2263, median≈0.2181, range≈0.0925–0.3939.
4. Rerun baseline on the frozen validation set. Old notebook mIoU measured against heuristic labels is not valid evidence of segmentation quality.
5. Review SOR's effect on thin leaves/stalks before enabling it for training; Dataset defaults to finite-point filtering only.
6. Benchmark point counts and sampling cost on Colab. FPS is correctness-first and not a scalable GPU implementation.

Optional visualization, mesh and phenotype helpers migrated from the notebook; they are not the training path. Canopy convex hull is a proxy, not measured volume. Leaf count/instances and production API/CLI remain later phases. The notebook Dockerfile pointed to a nonexistent API. It is replaced by a CPU baseline image; container build has not been validated here. Fixed sampled-point metrics are not a full-cloud score; full coverage inference/projection is a later requirement.

Tests use synthetic ASCII/binary PLY fixtures, alignment checks, exact-N padding, duplicate-coordinate FPS, inverse normalization, explicit decoding, grouped leakage, known confusion matrices, and multiprocess DataLoader integration. Real-data validation was recorded from Colab; the implementation environment used synthetic fixtures. GPU throughput was not measured at this historical phase.

## Validation in this workspace
- `python -m pytest -q`: 22 passed, 1 skipped.
- Multiprocess tensor DataLoader test skipped because the workspace forbids Unix sockets. It remains enabled in CI where sockets are available.
- Editable package installation succeeded; Python compilation and baseline CLI help succeeded.
- No real-data run, GPU benchmark or Docker build performed.

## Real MaizeField3D follow-up
Download paths corrected to `datasets/FielGrwon_ZeaMays_SegmentedPCD_100k.zip` and `datasets/FielGrwon_ZeaMays_RawPCD_100k.zip`, retaining revision pinning and archive path safety.
The dataset-specific decoder validates integer RGB shape/range, preserves original colors and assigns int64 binary labels without geometric heuristics or a giant palette. Dataset opts in via `annotation_mode="maizefield3d"`; baseline CLI defaults to this mode and retains explicit mapping compatibility.
Recorded Colab preprocessing, sampling and Dataset/DataLoader validation passed. `num_workers=0` worked; spawn failed for an interactively defined temporary class. That workaround is replaced by repository code and is not a current blocker.
Ready for baseline evaluation once verified grouping metadata is used to freeze the train/val/test manifest. Test must remain untouched while tuning. Scores are fixed-sampled-point metrics, not full-cloud projected metrics.

### Follow-up repository validation
`python -m pytest -q`: **42 passed, 1 skipped**. The unchanged multiprocess DataLoader test is skipped locally because Unix sockets needed for tensor transfer are prohibited; it remains enabled in CI. Added ASCII/binary RGB-only fixtures, rare/darkened-color decoding, malformed-RGB rejection, processing/index alignment, epoch/determinism/padding checks, mocked download path/revision/extraction checks, and real baseline execution on a synthetic RGB fixture. The 520-plant validation above was recorded from Colab and was not repeated in that implementation workspace.

## Phase 3 handoff update
The plant-identity split was subsequently completed (364 train / 78 validation / 78 test, seed 42) and the 10k sampled-point geometric validation run: mIoU 0.62299, leaf IoU 0.81734, stalk IoU 0.42864. The earlier manifest/baseline gates above are historical. Phase 3 reuses the externally supplied frozen manifest and does not evaluate test plants. See `phase3_pointnet2.md` for the neural implementation, mandatory real tiny-overfit gate and Colab workflow.
