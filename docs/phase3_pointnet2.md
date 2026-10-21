# Phase 3 — XYZ-only PointNet++ semantic baseline

> **Historical phase guide:** Run 1 remains the current reference, but Runs 2–4 have since been completed or started. See [results](results.md) and [experiments](experiments.md) for current status.

## Experiment contract

Use the existing `maizefield3d_manifest_seed42.json` unchanged: 364 train, 78 validation, 78 test plants grouped by plant identity, seed 42. This repository change does not create or modify a split file. The runner validates the entire manifest for duplicate identities/group leakage, records its exact-byte SHA-256, and opens only the selected train/validation PLYs. Test records may be listed in the manifest, but test files are never required or evaluated by the new scripts.

The model receives normalized float32 `(B,N,3)` XYZ only. Dataset RGB provides int64 labels (0=leaf, 1=stalk); no annotation color reaches the network. The existing finite filtering, normalization metadata, original point indices, semantic decoding and sampling remain intact. SOR is disabled. There is no class weighting, augmentation, dropout, scheduler, AMP, or compiled PointNet++ extension in this first baseline. The optimizer is Adam with constant learning rate 0.001.

Reference: Qi et al., [PointNet++ (2017)](https://arxiv.org/abs/1706.02413). This is an independent pure-PyTorch single-scale grouping implementation of hierarchical local PointNets plus skip-connected feature propagation. It is an engineering baseline, not a claim of reproducing an original benchmark configuration.

## Architecture and tensor shapes

Public model/layer interfaces use `(batch, points, channels)`. Channel-first conversions are confined to shared-MLP blocks. Radii are in the existing unit-sphere-normalized coordinate system.

| Stage | XYZ | Features | Radius / neighbors |
|---|---|---|---|
| Input | B × 10000 × 3 | XYZ only | — |
| SA1 | B × 1024 × 3 | B × 1024 × 64 | 0.1 / 32 |
| SA2 | B × 256 × 3 | B × 256 × 128 | 0.2 / 32 |
| SA3 | B × 64 × 3 | B × 64 × 256 | 0.4 / 32 |
| SA4 | B × 16 × 3 | B × 16 × 512 | 0.8 / 32 |
| FP4 | B × 64 × 3 | B × 64 × 256 | 3-NN interpolation + SA3 skip |
| FP3 | B × 256 × 3 | B × 256 × 256 | 3-NN + SA2 skip |
| FP2 | B × 1024 × 3 | B × 1024 × 128 | 3-NN + SA1 skip |
| FP1 | B × 10000 × 3 | B × 10000 × 128 | 3-NN + input XYZ skip |
| Classifier | — | B × 10000 × 2 logits | shared MLP |

**963,106 trainable parameters** with the default configuration. SA shared MLP widths are `(32,32,64)`, `(64,64,128)`, `(128,128,256)`, `(256,256,512)`. FP widths are `(256,256)`, `(256,256)`, `(256,128)`, `(128,128,128)`. Hidden layers use BatchNorm and ReLU.

FPS is deterministic and samples unique point indices, including coincident-coordinate cases. Radius query selects the nearest available neighbors within the radius; underfilled balls repeat their nearest point. An empty external-query ball falls back to its nearest point. SA centroids are selected from the cloud, so each has a self-neighbor. Interpolation uses inverse squared distance to up to three neighbors; one/two-point sparse clouds and exact coordinate matches are supported. Geometry searches have no gradients; gathered/interpolated features do.

Smaller synthetic inputs clamp centroid counts to available points. Model inputs need at least two points. Do not reduce real experiment density drastically while assuming the original radii will remain meaningful.

## Colab setup

Run from the repository checkout. Use the PR branch until merged, then use main. Record the resulting commit with each experiment. Keep the frozen manifest and checkpoints on persistent Drive storage; PLYs can remain on the Colab local disk for faster reads.

```bash
git fetch origin
git switch phase3-pointnet2
git pull --ff-only
git rev-parse HEAD
python -m pip install -e '.[dev,geometry]'
python -m pytest -q
```

Colab already provides a CUDA-enabled PyTorch build. Do not replace it with a CPU wheel. For a CPU-only local environment, install a CPU Torch build before installing this package. In a Colab notebook execute these commands in a `%%bash` cell after `%cd /content/agri3d-crop-phenotyping`.

The examples use generic placeholders for the externally stored manifest and data root. Replace them with your local paths; do not regenerate or rewrite the manifest to change the root directory.

## Mandatory tiny-overfit experiment

Run this as a `%%bash` cell in Colab:

```bash
set -e
python scripts/train_pointnet2.py \
  --config configs/pointnet2.yaml \
  --manifest "<MANIFEST>" \
  --root "<DATA_ROOT>" \
  --points 10000 --epochs 200 --batch-size 2 --lr 0.001 \
  --seed 42 --num-workers 0 --device cuda \
  --overfit-samples 4 \
  --output-dir "<ARTIFACT_DIR>/pointnet2_overfit"
```

The subset is drawn deterministically from sorted training sample IDs. It contains no validation/test plants. The same four plants and the same sampled points (Dataset epoch 0) are used throughout; there is no augmentation. `run.json` records their IDs and the initial eval-mode loss. Each epoch logs both online training metrics (`train`) and a post-update eval-mode pass on those same training points (`train_eval`). This latter pass catches train/eval discrepancies, including BatchNorm running statistics.

Inspect the result in a Python cell:

```python
import json
from pathlib import Path
run = Path("<ARTIFACT_DIR>/pointnet2_overfit")
print(json.loads((run / 'summary.json').read_text()))
history = json.loads((run / 'metrics.json').read_text())
print(history[-1]['train_eval'])
print('Passing checkpoint exists:', (run / 'overfit_passed.pt').exists())
```

Passing criteria on the fixed training subset: loss <=20% of its initial value, accuracy >=0.98 and mIoU >=0.95. An epoch that passes writes `overfit_passed.pt`, retained even if a later epoch regresses. Inspect its recorded epoch/metrics rather than assuming that the final epoch is best. Do not begin a full run until this gate passes. A low train-mode loss alone is insufficient. If it fails, inspect labels, predictions, sampled class counts and BatchNorm convergence; extend the tiny run or diagnose before changing model complexity. The thresholds are engineering checks, not promises about how many epochs will be needed.

Resume an interrupted tiny run from the same output directory using the same command/config plus:

```bash
--resume "<ARTIFACT_DIR>/pointnet2_overfit/last.pt"
```

`--epochs` specifies total epochs, not additional epochs. Use a larger total when continuing a completed run. A new run refuses to overwrite a nonempty output directory.

## First regular training run

Only after inspecting the tiny-overfit results:

```bash
python scripts/train_pointnet2.py \
  --config configs/pointnet2.yaml \
  --manifest "<MANIFEST>" \
  --root "<DATA_ROOT>" \
  --points 10000 --epochs 100 --batch-size 2 --lr 0.001 \
  --seed 42 --num-workers 0 --device cuda \
  --overfit-checkpoint "<ARTIFACT_DIR>/pointnet2_overfit/overfit_passed.pt" \
  --output-dir "<ARTIFACT_DIR>/pointnet2_run1"
```

The passing checkpoint is evidence of the smoke check; the full model starts from a fresh seeded initialization. Regular training resamples train points by epoch and keeps validation points fixed. The gate requires matching manifest hash, model geometry settings, point count, seed, batch size, learning rate and data preprocessing contract. It does not carry over tiny-overfit weights.

`best.pt` is selected by validation mIoU (validation loss breaks exact ties). It never selects using test metrics. Metrics reuse `SegmentationEvaluator`: accuracy, mIoU, both class IoUs, precision, recall and F1. Labels `class_0_*` are leaf metrics; `class_1_*` are stalk metrics. Loss and metrics exclude invalid padding and are accumulated across all valid points rather than averaged per batch.

## Validation command

```bash
python scripts/validate_pointnet2.py \
  --manifest "<MANIFEST>" \
  --root "<DATA_ROOT>" \
  --checkpoint "<ARTIFACT_DIR>/pointnet2_run1/best.pt" \
  --device cuda --num-workers 0 \
  --output "<ARTIFACT_DIR>/pointnet2_run1/validation.json"
```

This command loads model/data settings from the checkpoint and only exposes validation. The manifest must match the saved hash. All scores concern fixed sampled points, **not full-cloud projected segmentation**. Keep the point count, sampling seed and manifest identical to the geometric baseline for comparison. No test-set evaluation is implemented for Phase 3 development.

## Artifacts and reproducibility

| File | Purpose |
|---|---|
| `run.json` | Config, manifest hash, split counts, sample IDs, environment and code commit |
| `metrics.json` | Complete epoch history; point-weighted loss and global segmentation metrics |
| `summary.json` | Completion status, best mIoU, tiny-overfit gate status |
| `last.pt` | Latest complete epoch, model/Adam states, history, global and loader RNG states |
| `best.pt` | Best validation mIoU for regular runs; best train-eval mIoU for tiny runs |
| `overfit_passed.pt` | Most recently passing tiny-overfit checkpoint; absent until the gate passes |
| `validation.json` | Explicit validation-command results with checkpoint/manifest fingerprints |

Resume requires `output-dir/last.pt`, unchanged experiment/batch/worker/LR settings and a matching manifest. Restore the complete output directory after a Colab disconnect. Checkpoints and JSON files are replaced atomically. Bitwise resumed-vs-uninterrupted CPU equivalence is tested. Exact reproducibility across different hardware, PyTorch versions, CUDA versions or CPU/GPU transfers is not guaranteed. Strict deterministic algorithms are enabled; unsupported nondeterministic CUDA operations will fail rather than silently change behavior. The CUDA execution path still needs validation in Colab.

Default `num_workers=0` matches the verified Colab setup. Worker processes, when requested through the script, use module-defined classes and spawn, with persistent workers disabled so Dataset epoch changes propagate. The workspace's multiprocess tensor test is skipped because Unix sockets are unavailable; it remains enabled in CI. Use scripts rather than defining Dataset subclasses interactively.

## Targets and limitations

Recorded geometric baseline on the frozen 78-plant validation split at 10k points:

| Metric | Geometry | Initial PointNet++ target |
|---|---:|---:|
| Accuracy | 0.83935 | Report |
| mIoU | 0.62299 | >0.70 |
| Leaf IoU | 0.81734 | Approximately >=0.82 |
| Leaf F1 | 0.89949 | Report |
| Stalk IoU | 0.42864 | >0.55 |
| Stalk precision | 0.68516 | Report |
| Stalk recall | 0.53377 | Report |
| Stalk F1 | 0.60006 | Report |

These neural targets are not measured results or guarantees. Plant-identity grouping does not establish genotype-held-out generalization; retain the agreed split and describe that limitation when reporting results.

- **FPS:** Python iterates 1024+256+64+16 centroid selections. It is O(B*N*S) and likely to be a GPU kernel-launch bottleneck. It stores linear-size working arrays, not N×N distances.
- **Neighborhoods:** search is chunked (128 query points). At B=2,N=10000 one largest distance tensor is `2×128×10000` float32, about 9.8 MiB. Temporary arithmetic buffers and saved activations add memory; this is not a peak-VRAM estimate. SA1 grouping/MLPs and final dense FP activations are also significant allocations.
- **Interpolation:** chunked queries avoid a full 10000×1024 allocation. Feature gathers still retain training activations. Later bounded T4 profiling measured VRAM and batch throughput; this early implementation note does not attribute the full GPU bottleneck. Batch 2 remains the scientific setting. If batch 1 is needed, rerun the tiny gate with that same batch setting.
- **I/O:** each Dataset access reads and processes the full 100k PLY before sampling. This can dominate time, especially from Drive. Copy PLYs to Colab local storage while keeping manifest identity and output artifacts persistent.
- **BatchNorm:** small subsets and severely sparse/near-empty local neighborhoods can show low train loss yet poor or unstable eval behavior. The synthetic stress check exposed this, so the gate uses eval-mode train metrics. Keep the first real tiny-overfit at 10k points and allow running statistics to settle. Do not bypass the gate using train-mode scores.
- **Padding:** repeated padding points affect feature context/BatchNorm, but do not contribute directly to loss or metrics. Real validated 100k→10k clouds do not need padding. Degenerate undersized clouds are not evidence of equivalent training behavior.
- **Provenance:** checkpoint hashes identify the manifest, not the PLY byte content. Preserve the pinned dataset revision and immutable extracted files externally. No dataset files or frozen split were regenerated here.

At the time of this initial implementation note, Colab CUDA and real-data overfit checks were still open. Runs 1–3 have since been measured. Consult current [results](results.md) and [performance](performance.md); Run 4 remains in progress.

## Implementation verification and repository review

- Full existing + new suite: **60 passed, 1 skipped** (Unix-socket-dependent multiprocess DataLoader test). This includes a CPU synthetic overfit regression that reaches >=0.98 accuracy and >=0.95 mIoU in eval mode, and exact CPU checkpoint/resume equivalence.
- Default network CPU forward: `(1,10000,3)` → `(1,10000,2)`; parameter count and all SA/FP stage shapes checked.
- Installed-package CLI smoke: two epochs on synthetic RGB-only PLYs, with an absent test file; produced `run.json`, `metrics.json`, `summary.json`, `last.pt`, and `best.pt`. As expected, two epochs did not satisfy the overfit gate, and no `overfit_passed.pt` was emitted.
- New Python modules pass lint; Python compilation and editable installation passed.
- Review confirmed unweighted masked loss, no annotation features, no test-file access by the new runner, manifest hash checks, validation-only model selection, and bounded query dimensions. No known code blocker remains for the Colab sequence.
- Real MaizeField3D neural training and CUDA benchmarking have **not** been run here. The real tiny-overfit and CUDA/VRAM checks remain mandatory before the first regular run. The sparse-cloud/BatchNorm and padded-context limitations above remain relevant.

## Phase 3A: diagnose Run 1 validation errors

Run 1's recorded best epoch 76 achieved validation mIoU 0.8366027, leaf IoU 0.9202424, and stalk IoU 0.7529630. Before changing model/training choices, run the validation-only [Phase 3A analysis workflow](phase3a_validation_analysis.md). It adds per-plant metrics, exact Tagname metadata joins, descriptive associations, and optional comparable GT/prediction/error views. The existing validator and checkpoint semantics remain unchanged; test remains excluded.

## Phase 3B: one controlled augmentation experiment

Use [the Run 2 workflow](phase3b_run2_augmentation.md) and `configs/pointnet2_run2_aug.yaml` to add training-only Z rotation, mild isotropic scale jitter, and clipped coordinate jitter after normalization/sampling. All other Run 1 experiment settings remain fixed. Legacy unaugmented checkpoints and the existing passing tiny-overfit gate remain compatible; augmentation is recorded/hashed and must match on resume. Validation/Phase 3A analysis remain unaugmented. Test remains excluded.

## Phase 3P: recovery and training performance

[Phase 3P](phase3p_training_performance.md) adds dataset-free checkpoint inspection, short train/val torch/Nsight profiling, opt-in detailed timing/VRAM telemetry, loader benchmarks, and a provenance-checked local normalized-cloud cache. The Run 2 YAML/scientific contract remains unchanged. Phase 3P also records T4 memory/epoch observations and bounded batch-throughput results; see the current [performance summary](performance.md) for the evidence limits.

## Phase 3C: Run 3 plateau scheduling

[Run 3](phase3c_run3_scheduler.md) uses Run 1's no-augmentation setup plus ReduceLROnPlateau on validation mIoU. Full scheduler policy/state is checkpointed and checked on resume; unscheduled Run 1/2 remain compatible. Separate artifacts: `artifacts/pointnet2_run3_scheduler`. Run 3 is complete; its validation result is recorded in the project results registry. Test evaluation remains out of scope.

## Phase 3D: Run 4 normal features

[Run 4](phase3d_run4_normals.md) adds full-cloud estimated kNN-PCA normals to Run 1's unaugmented, constant-LR policy. XYZ remains the only geometry for sampling/neighborhood queries. Feature inputs increase parameters by 480, require a new matching overfit gate, and are hashed in checkpoints/cache provenance. The normal audit is train/val-only; test stays untouched.
