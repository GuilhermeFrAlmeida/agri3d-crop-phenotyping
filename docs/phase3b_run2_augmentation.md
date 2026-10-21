# Phase 3B — controlled augmentation Run 2

## Hypothesis and scope

Phase 3A validation analysis found more variability in stalk IoU than leaf IoU, essentially no association with GT stalk ratio, a weak negative size association, and both stalk false-negative and false-positive failure modes. Run 2 tests whether mild geometry augmentation improves geometric robustness/generalization. It is one policy, not an augmentation sweep, and introduces no class weighting.

The only experimental change from Run 1 is training augmentation. Keep the frozen plant-identity manifest (364 train / 78 val / 78 test), seed 42, XYZ-only 10,000 points, PointNet++ SSG geometry/width/depth, batch 2, Adam LR 0.001, 100 epochs, masked unweighted cross-entropy, no SOR, no scheduler, no dropout, no RGB/normals, and selection by validation mIoU (existing loss tie-break). Test remains inaccessible through all Phase 3 development CLIs. Run 2 initializes fresh seeded weights; it does not resume or warm-start Run 1 or the tiny-overfit model.

## One augmentation policy

`configs/pointnet2_run2_aug.yaml` duplicates Run 1 configuration with only this block added:

```yaml
augmentation:
  enabled: true
  rotate_z: true
  scale_min: 0.95
  scale_max: 1.05
  jitter_sigma: 0.005
  jitter_clip: 0.02
```

Each training plant receives a uniform Z angle in [0, 2π), one uniform isotropic scale factor in [0.95, 1.05], and zero-mean Gaussian coordinate jitter with σ=0.005, clipped per coordinate to ±0.02. Units are normalized coordinates. There are no rotations around X or Y. Z rotation itself preserves Z exactly and distances within floating-point tolerance; scale/jitter mildly perturb geometry without tilting the plant's vertical axis. No second normalization cancels the scale jitter; augmented points may extend slightly outside the original unit sphere.

Exact implemented order:

1. Read PLY XYZ/properties/source indices.
2. Remove nonfinite points with aligned attributes.
3. Decode MaizeField3D RGB supervision (black stalk=1; nonblack leaf=0).
4. Optional SOR in generic Dataset; disabled in both experiment configs.
5. Centroid shift and unit-sphere normalization.
6. Existing fixed-size voxel sampling/padding.
7. **Training only:** Z rotation → isotropic scale → clipped jitter.
8. Convert to the existing Dataset tensor output contract.

The finite filter precedes decoding, retaining the existing behavior. Augmentation returns a copy; labels, RGB attributes, `valid_mask`, `source_index`, order, and pre-augmentation normalization metadata are preserved exactly. Padded repetitions share source-point jitter instead of becoming distinct noisy locations. Padding remains masked out of loss/metrics. The real 100k→10k pipeline does not need padding.

`centroid`/`scale` remain provenance for the original normalization. Applying those values alone to augmented training XYZ does not reconstruct original raw coordinates. Use `source_index` to retrieve original PLY properties. Validation remains unaugmented, so its existing coordinate reconstruction and analysis NPZ semantics are unchanged.

## Determinism and compatibility

The Dataset retains its existing SHA256-derived RNG seed from `seed:epoch:sample_id`. Augmentation uses that same NumPy Generator **after** sampling. No augmentation draw precedes sampling, and no global NumPy/Python/Torch RNG is consumed. Thus Run 1 and Run 2 select identical source points for the same seed/epoch/sample; their training XYZ differs only by the configured augmentation. The same seed/epoch/sample is repeatable, independent of access order. Training epochs vary both resampling and augmentation; validation always uses epoch 0 and does not augment or consume extra randomness.

Run 1 configuration remains unchanged. An absent augmentation field, `{}`, or `augmentation: {enabled: false}` disables it and preserves the historical `contract.augmentation: null`. Existing Run 1 checkpoints without the new config field or contract hash still validate, analyze, and resume unaugmented. Disabled latent rotation/scale/jitter settings have no effect and are not experiment invariants.

For enabled augmentation, the experiment contract includes the full six settings, Z-angle range, post-normalization/sampling training-only placement, duplicate-source jitter semantics, and policy version 1. Numeric policy values are canonicalized to floats. New `run.json` and checkpoints include a SHA256 of canonical sorted JSON for the complete experiment contract. Resume/validation/analysis verify both the restored contract and its hash; augmented checkpoints require the hash. Resume rejects enabling/disabling augmentation or changing any active policy parameter. Checkpoint format/model state structure remains unchanged. Older code cannot evaluate Run 2 checkpoints; pull this implementation first.

The tiny-overfit gate stays unaugmented. Using enabled augmentation with `--overfit-samples` fails explicitly; use `configs/pointnet2.yaml` for a gate. A previously passing Run 1 gate is valid evidence for Run 2 **only when all non-augmentation contract fields match**. The gate must itself have null augmentation; this one narrowly scoped exception never applies to resume. Gate provenance records its original contract/hash and file hash. No augmented gate or new real tiny-overfit is required.

## Colab sequence

Before merge, pull the PR branch; after merge, use main instead:

```bash
git fetch origin
git switch phase3b-run2-augmentation
git pull --ff-only
python -m pip install -e '.[dev,geometry,analysis]'
python -m pytest -q
```

Adjust only file locations below to match your existing manifest, local PLY root, and Drive artifacts. Preserve the manifest and use a new/empty Run 2 output directory. Keep Colab's GPU PyTorch installation.

```bash
python scripts/train_pointnet2.py \
  --config configs/pointnet2_run2_aug.yaml \
  --manifest "<MANIFEST>" \
  --root "<DATA_ROOT>" \
  --points 10000 --epochs 100 --batch-size 2 --lr 0.001 \
  --seed 42 --num-workers 0 --device cuda \
  --overfit-checkpoint "<ARTIFACT_DIR>/pointnet2_overfit/overfit_passed.pt" \
  --output-dir "<ARTIFACT_DIR>/pointnet2_run2_aug"
```

If interrupted, supply the same YAML/arguments and `--resume .../pointnet2_run2_aug/last.pt`; epochs is the total target. Resume requires the same output directory and restores optimizer, loader/global RNG, and history. Do not use a Run 1 checkpoint to resume Run 2.

Validate selected weights only on frozen validation:

```bash
python scripts/validate_pointnet2.py \
  --manifest "<MANIFEST>" \
  --root "<DATA_ROOT>" \
  --checkpoint "<ARTIFACT_DIR>/pointnet2_run2_aug/best.pt" \
  --device cuda --num-workers 0 \
  --output "<ARTIFACT_DIR>/pointnet2_run2_aug/validation.json"
```

For plant-level comparison, use the unchanged Phase 3A analysis command with Run 2 `best.pt` and a new `.../pointnet2_run2_aug/validation_analysis` output directory. The saved augmentation policy is verified but never applied to those validation samples. No test split option is added.

## Recorded Run 2 result

| Metric at selected checkpoint | Run 1 reference | Run 2 |
|---|---:|---:|
| Best epoch | 76 | 93 |
| Validation accuracy | 0.9358397 | 0.9347205 |
| Validation mIoU | 0.8366027 | 0.8295737 |
| Leaf IoU | 0.9202424 | 0.9198950 |
| Stalk IoU | 0.7529630 | 0.7392524 |
| Stalk precision | 0.8521421 | 0.8828109 |
| Stalk recall | 0.8661208 | 0.8196907 |
| Logged training mIoU at best epoch | 0.92403 | Not in the recorded result summary |
| Logged train–validation mIoU gap | ~0.0874 | Not in the recorded result summary |

These are recorded Colab results. Run 2 did not improve aggregate performance: stalk precision increased while stalk recall, stalk IoU and mIoU decreased. See the project results registry for the current comparison.

**Gap interpretation:** Run 2's logged training metrics are online/train-mode scores on augmented inputs; Run 1's are on unaugmented inputs. Therefore a smaller logged gap can partly reflect harder training inputs, not improved generalization. Do not treat gap reduction alone as success; establish it with the unchanged validation metrics first. This implementation does not add a separate clean-train evaluation experiment. A small single-seed gain is preliminary evidence, not proof of statistical consistency; do not silently launch extra variants or touch test to resolve it.

Artifacts follow the existing layout: `run.json` (config/contract/hash/gate provenance), `metrics.json` (per-epoch train/val metrics), `summary.json`, `best.pt`, `last.pt`, and the separate `validation.json`. Run 1 artifacts are never overwritten.

## Verification and remaining empirical checks

Tests isolate Z rotation distances/Z invariance, scale factors/range, clipped jitter, copied source data/metadata, exact label/mask/source-index preservation, duplicate padding, seed/epoch repeatability, changed training epochs, unaugmented validation, disabled Run 1 sampling parity, config equality outside augmentation, legacy checkpoints, gate compatibility, augmented contract/hash failures, and exact synthetic CPU checkpoint/resume. Synthetic Run 2 validation also matches the existing validator/Phase 3A analysis.

Run 2 was completed on the recorded Colab setup. The local synthetic tests validate software behavior; the recorded real-data comparison is summarized above and in the project results registry. The existing FPS/neighborhood/PLY-I/O bottlenecks remain; augmentation adds small CPU coordinate-copy/transform costs after sampling. No architecture, optimizer, class weighting, scheduler, sampling density, or additional augmentation variants are introduced.
