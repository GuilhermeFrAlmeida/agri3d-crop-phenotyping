# Phase 3D — Run 4: estimated surface normals

## Hypothesis and experimental boundary

Explicit local surface orientation may help distinguish maize stalk and leaf points that an XYZ-only network must infer implicitly. This is a hypothesis, not a result. Run 1 remains the champion; Run 2 augmentation and Run 3 plateau scheduling did not improve validation mIoU.

| Frozen-validation reference | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---:|---:|---:|---:|
| mIoU | 0.8366027 | 0.8295737 | 0.8317226 | pending |
| Leaf IoU | 0.9202424 | 0.9198950 | — | pending |
| Stalk IoU | 0.7529630 | 0.7392524 | — | pending |
| Stalk precision | 0.8521421 | 0.8828109 | — | pending |
| Stalk recall | 0.8661208 | 0.8196907 | — | pending |

The Run 4 YAML is identical to Run 1 except for `features`. Frozen manifest 364/78/78; plant grouping; seed 42; 10,000 points; SSG; batch 2; Adam LR 0.001 constant; 100 regular epochs; no augmentation, scheduler, weighting, SOR, AMP, TF32, added dropout, or changed neighborhood/MLP widths. Checkpoint selection retains validation mIoU and the existing lower-loss exact-tie rule. Test files are never loaded.

RGB defines supervision in segmented MaizeField3D: black means stalk and nonblack means leaf. Feeding RGB to the model would leak the target. Dataset model inputs remain geometry and estimated normals only. RGB is used exclusively for label decoding. The dataset's PLYs have no normal fields; any normal fields encountered in another PLY are deliberately overwritten by estimation in this mode.

## Frozen feature policy and algorithm

```yaml
features:
  mode: xyz_normals
  normals:
    source: estimated
    method: knn_pca
    k: 32
    orientation: deterministic
    policy_version: 1
```

The `deterministic` configuration alias resolves to this explicit contract:

```json
{
  "geometry": "xyz",
  "point_features": "estimated_normals",
  "normal_estimation": {
    "source": "estimated",
    "method": "knn_pca",
    "k": 32,
    "orientation": "largest_abs_component_positive",
    "policy_version": 1,
    "neighbors": "include_self_then_distance_source_row_ties",
    "covariance": "centered_float64_divide_by_effective_k",
    "eigen_gap_rtol": 1e-8,
    "degenerate_fallback": [0.0, 0.0, 1.0]
  }
}
```

Order is **PLY → finite filtering → semantic decode → full-resolution normal estimation → XYZ normalization → 10k sampling**. The reader's existing float32 XYZ contract is preserved; estimation promotes those coordinates to float64 internally. Normal estimation is never performed on the 10k sampled cloud. No RNG is consumed, so the same seed/epoch/plant selects the same source points as Run 1.

For each cleaned source point:

1. Include the point itself, then its nearest `min(32,N)-1` other points. `k` includes self and is capped at cloud size.
2. Query SciPy `cKDTree` with exact search (`eps=0`) and one worker. Sort candidates by float64 squared distance, then cleaned source-row index. Expand ambiguous cutoff ties with a radius query before sorting/truncating. Original retained row order preserves source identity after filtering.
3. Center that local neighborhood in float64. Compute covariance `centered.T @ centered / effective_k`, then its symmetric eigendecomposition. The eigenvector with smallest eigenvalue is the local surface normal.
4. Require a unique minimum-eigenvalue direction: if the largest eigenvalue is nonpositive, or `(lambda_1 - lambda_0) <= 1e-8 * max(lambda_2,0)`, mark the neighborhood degenerate. This covers coincident points, lines, insufficient points, and ambiguous/isotropic minimum eigenspaces. Emit the explicit finite unit fallback `(0,0,1)` and count/retain the degeneracy flag.
5. Otherwise choose the sign making the largest absolute normal component positive. Exact component-magnitude ties prefer X, then Y, then Z. Normalize to unit length and store float32 normals.

The estimator works in bounded chunks (default 2,048 queries) and never builds an N×N matrix. Ordinary kNN work is approximately O(N log N + Nk), plus N small 3×3 eigendecompositions. Pathological huge duplicate/cutoff-tie sets can make deterministic tie expansion more expensive. The eigensolver's arbitrary +/- sign never determines the final orientation. Repeated values are tested, including chunk changes, tied neighborhoods, and deliberately inverted eigensolver signs. Cross-version/platform bitwise identity is not promised; cache provenance includes the implementation and SciPy version.

**Sign limitation:** this is a canonical mathematical convention, not a physical outward normal or a globally smooth orientation field. Signs can switch at dominant-component boundaries, and fallback normals carry no reliable measured surface orientation. The policy uses no labels/RGB, camera metadata, validation-derived fitting, test information, or dataset-global fitting. High degeneracy counts require inspection before interpreting the experiment; unit length alone does not establish useful normals.

XYZ normalization translates and isotropically scales only `points`; normals are untouched direction attributes. Sampling/padding copies normals, labels, masks, and source indices together. Padded duplicates carry identical normals and remain excluded by the existing valid mask.

## Model and checkpoint changes

Dataset batches expose `points: (B,N,3)` and `features: (B,N,3)` separately. `model(xyz, features)` never passes feature channels to FPS, radius search, or interpolation coordinates. Only these feature-dependent dimensions change:

| Layer | Run 1 input channels | Run 4 input channels | Extra parameters |
|---|---:|---:|---:|
| First SA shared Conv2d | 3 | 6 | 96 |
| Last FP shared Conv1d | 131 | 134 | 384 |
| **Total** | | | **480** |

All SA counts/radii/neighborhood sizes, MLP widths, stage count, and classifier width remain unchanged. Total parameters: **963,106 → 963,586**. Run metadata records total parameters, point-feature channels, added parameters, and normal implementation/SciPy provenance.

XYZ mode adds no `features` tensor and retains the exact legacy `"features": "xyz"` experiment-contract representation. The default model has the same state-dict keys/shapes and numerically identical XYZ-only forward path. Run 1/2/3 validation/analysis remain compatible even when old configs lack a feature field. Run 4 records/hashes the full feature policy and rejects mismatched policies or missing required hashes. An XYZ-only checkpoint cannot initialize/resume Run 4 through the standard runner.

Training, validation, analysis, profiling, and throughput helpers all build the model from the same feature contract. Feature tensors use the same pinned/nonblocking H2D path as XYZ. Geometry-only loader benchmarks also include feature tensors in transfer and identity checks.

## Cache and quality audit

The Phase 3P cache boundary now includes normal estimation: clean → decode → estimate → normalize. Sampling stays live per epoch. Entries include feature mode/policy, k, orientation/version, normal implementation SHA256, and SciPy version in addition to existing source/manifest/sample/preprocessing checksums. XYZ-only entries can never be used as normal-feature entries. Changed policies/code create new entries; previous cache files and scientific artifacts are not deleted or overwritten. The cache is optional execution infrastructure and does not enter the scientific experiment contract. Keep it on Colab local `/content`; it can be rebuilt after runtime loss.

The audit checks full-cloud `(N,3)`, finiteness, unit norm tolerance 1e-6, repeated recomputation, and alignment of sampled XYZ/normals/labels/source indices. It reports checked plants/points, finite ratio, norm distribution, component statistics, and degeneracy counts. Cached and uncached outputs are also covered by tests. A successful technical audit does not establish predictive benefit. By default the CLI checks a bounded set; increase counts to 364 train and 78 val for an all-development audit. It does not read test. All training accesses independently validate generated normal shape/finiteness/unit length.

## Colab commands: audit, new gate, then full run

Pull `phase3d-run4-normals` and run from the repository root:

```bash
git fetch origin
git switch phase3d-run4-normals
git pull --ff-only
python -m pip install -e '.[dev,geometry,analysis]'
python -m pytest -q
```

Use this variable block and the relevant command in the **same `%%bash` cell**, since shell variables do not persist across cells:

```bash
MANIFEST=/content/drive/MyDrive/Agri3D/data/splits/maizefield3d_manifest_seed42.json
PLY_ROOT=/content/agri3d-crop-phenotyping/data/raw
ARTIFACTS=/content/drive/MyDrive/Agri3D/artifacts
NORMAL_CACHE=/content/agri3d_run4_normal_cache
CONFIG=configs/pointnet2_run4_normals.yaml

python scripts/audit_pointnet2_normals.py \
  --config "$CONFIG" --manifest "$MANIFEST" --root "$PLY_ROOT" \
  --split train --max-plants 5 --cache-dir "$NORMAL_CACHE" \
  --output "$ARTIFACTS/pointnet2_run4_normals_audit/train.json"

python scripts/audit_pointnet2_normals.py \
  --config "$CONFIG" --manifest "$MANIFEST" --root "$PLY_ROOT" \
  --split val --max-plants 5 --cache-dir "$NORMAL_CACHE" \
  --output "$ARTIFACTS/pointnet2_run4_normals_audit/val.json"
```

Inspect the audit, particularly degeneracy counts. Use new output filenames for repeat audits. Cache population/recomputation takes CPU time before training.

**Run 4 requires a new passing tiny-overfit gate.** Run 1's XYZ gate is not sufficient because features and input dimensions changed. The existing deterministic 2–8 training-plant mechanism and criteria remain: loss ≤20% of initial, accuracy ≥0.98, mIoU ≥0.95. Use four fixed plants below. Its 200-epoch memorization budget is a separate engineering gate, not Run 4's 100-epoch regular experiment. Augmentation and scheduling remain disabled.

```bash
MANIFEST=/content/drive/MyDrive/Agri3D/data/splits/maizefield3d_manifest_seed42.json
PLY_ROOT=/content/agri3d-crop-phenotyping/data/raw
ARTIFACTS=/content/drive/MyDrive/Agri3D/artifacts
NORMAL_CACHE=/content/agri3d_run4_normal_cache
CONFIG=configs/pointnet2_run4_normals.yaml

python scripts/train_pointnet2.py \
  --config "$CONFIG" --manifest "$MANIFEST" --root "$PLY_ROOT" \
  --points 10000 --epochs 200 --batch-size 2 --lr 0.001 --seed 42 \
  --overfit-samples 4 --device cuda --num-workers 2 \
  --pin-memory --non-blocking --prefetch-factor 2 \
  --cache-dir "$NORMAL_CACHE" \
  --output-dir "$ARTIFACTS/pointnet2_run4_normals_overfit"
```

Review the gate's `summary.json` and confirm `overfit_passed.pt` exists. That file is written only when the existing criteria pass; no real normals-gate pass is claimed from Work. The regular run below verifies its feature/data/model contract before loading train/val clouds:

```bash
MANIFEST=/content/drive/MyDrive/Agri3D/data/splits/maizefield3d_manifest_seed42.json
PLY_ROOT=/content/agri3d-crop-phenotyping/data/raw
ARTIFACTS=/content/drive/MyDrive/Agri3D/artifacts
NORMAL_CACHE=/content/agri3d_run4_normal_cache
CONFIG=configs/pointnet2_run4_normals.yaml

python scripts/train_pointnet2.py \
  --config "$CONFIG" --manifest "$MANIFEST" --root "$PLY_ROOT" \
  --points 10000 --epochs 100 --batch-size 2 --lr 0.001 --seed 42 \
  --device cuda --num-workers 2 \
  --pin-memory --non-blocking --prefetch-factor 2 \
  --cache-dir "$NORMAL_CACHE" \
  --overfit-checkpoint "$ARTIFACTS/pointnet2_run4_normals_overfit/overfit_passed.pt" \
  --output-dir "$ARTIFACTS/pointnet2_run4_normals"
```

The user-qualified T4 runtime is workers 2, pinning true, nonblocking true, prefetch 2, persistence **false** (default; do not pass `--persistent-workers`). Cache is optional and is explicitly enabled above to avoid repeated full-cloud estimation. The scientific YAML retains Run 1's worker default so its only block change is features; CLI overrides are recorded in run metadata (`num_workers` remains a legacy RunConfig execution field excluded from the experiment hash), and the existing worker-count equality check on resume remains in force. GPU/worker execution must be verified in Colab where unavailable in Work.

Validate the selected checkpoint on validation only:

```bash
python scripts/validate_pointnet2.py \
  --manifest /content/drive/MyDrive/Agri3D/data/splits/maizefield3d_manifest_seed42.json \
  --root /content/agri3d-crop-phenotyping/data/raw \
  --checkpoint /content/drive/MyDrive/Agri3D/artifacts/pointnet2_run4_normals/best.pt \
  --device cuda --num-workers 2 \
  --output /content/drive/MyDrive/Agri3D/artifacts/pointnet2_run4_normals/validation.json
```

Validation/analysis reconstruct the same full-cloud feature policy from the checkpoint. Their existing CLI does not accept cache flags, so validation currently recomputes normals; training, auditing, and Phase 3P benchmarks can reuse the cache.

## Decision criteria

Primary: validation mIoU > **0.8366027**. Secondary: stalk IoU > **0.7529630**, leaf IoU near/above **0.92024**, and inspect stalk precision/recall balance. Keep Run 1 as champion unless real Run 4 evidence supports replacement. Single-seed validation comparisons do not establish statistical significance or untouched-test performance. No additional scientific variant, test evaluation, or real GPU run is performed in Work.
