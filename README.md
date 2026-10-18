# Agri3D crop phenotyping
Phase 2 geometry baseline and Phase 3 XYZ-only PointNet++ semantic segmentation.

See [the Phase 3 workflow](docs/phase3_pointnet2.md) for installation, mandatory tiny-overfit, training, validation, checkpoint/resume contracts and known limitations.

## Setup
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev,geometry,visualization,data]'
pytest -q
```
Colab: clone this repository at a specific commit, install it, and import the package.
Do not use notebook `%%writefile` cells to redefine source files. Keep dataset and experiment artifacts outside git.
Install the appropriate CUDA Torch build in Colab. The package accepts the installed Torch build.

## Supervision and splits
Use the annotated PLY geometry directly for supervised training. Raw and segmented exports are not assumed to have pointwise correspondence.
Create a JSON list of records with `path` (relative to dataset root), `sample_id`, `group_id`, and `split` (`train`, `val`, `test`).
All resolutions, repeats, and related samples must share the selected metadata group.
`make_group_split` assigns approximately 70/15/15 percent of **groups**; counts of plants can differ.
Choose plant or genotype grouping from verified dataset metadata, save the manifest, and reuse it unchanged across experiments.

For segmented MaizeField3D exports, select `annotation_mode="maizefield3d"`:
- RGB `(0,0,0)` → stalk → class `1`.
- Every non-black RGB, including rare/darkened colors → leaf → class `0`.

This binary rule was verified in the user's real-data Colab validation across all 520 annotated plants. No exact leaf-color palette is needed for Phase 3. Missing/malformed RGB fails clearly. Original RGB is preserved through decoding, filtering, normalization and sampling for Phase 4 instance validation; it is never a model input. Dataset outputs retain the existing XYZ-only contract; original RGB can be retrieved from source PLY using `source_index`.
The old height-based ground-truth heuristic is retired. This mode is only for **segmented annotation exports**, not natural-color raw point clouds. Generic PLY ingestion remains dataset-agnostic. Other datasets can still use explicit `color_map`/`label_map` in the default `explicit` mode.

```python
from agri3d.training.dataset import SegmentationDataset
from torch.utils.data import DataLoader
# Real MaizeField3D segmented PLYs contain RGB annotations, not label fields:
ds = SegmentationDataset('data/splits/manifest.json', 'data/raw', 'train',
                         annotation_mode="maizefield3d", num_points=10000)
ds.set_epoch(0)
loader = DataLoader(ds, batch_size=2, num_workers=0)
```
Output: float32 `(N,3)` points, int64 `(N,)` labels, bool validity mask, original point indices, sample identity, centroid and scale.
Padding duplicates are invalid: mask them in loss and metrics. Epoch changes only training samples.
Recreate worker iterators after `set_epoch`; persistent workers do not receive parent epoch changes.

Voxel mode picks original representatives, fills from unused original points, then repeats only if the cloud is smaller than N.
FPS is a CPU correctness implementation with O(input points × output points) work; avoid it for 100k→50k runs.
Voxel size in the Dataset is applied after unit-sphere normalization and is dimensionless; geometric filtering voxel sizes are meters.
Normalization statistics come from the full cleaned cloud before sampling. Physical measurements should use the cleaned full cloud, not repeated/sampled points.

## Baseline evaluation
```bash
python scripts/evaluate_baseline.py --manifest data/splits/manifest.json --root data/raw \
  --annotation-mode maizefield3d --split val --points 10000 --output artifacts/baseline.json
```
The command evaluates every selected sample, accumulates a confusion matrix, masks padding, and reports sampled-point coverage explicitly.
Keep test untouched while tuning. Geometric thresholds are meters; the command restores scale before baseline inference.
Latency is baseline inference only on this machine, not the specification's future GPU inference + phenotyping SLA.
No full-cloud projection or leaf instance counting is implemented yet.
See `docs/phase2_audit.md` for findings and remaining Phase 3 gates.

## Dataset downloads and validation
The pinned Hugging Face archive paths are:
- `datasets/FielGrwon_ZeaMays_SegmentedPCD_100k.zip`
- `datasets/FielGrwon_ZeaMays_RawPCD_100k.zip`

```bash
python scripts/download_ground_truth.py --revision <dataset-commit> --output data/raw
python scripts/download_maize_data.py --revision <dataset-commit> --output data/raw
```
Revision pinning and archive path validation remain enforced.

User-reported Colab validation passed for all 520 annotated plants: 100k points each, finite coordinates, Z vertical, meter-consistent scale, and both black/non-black annotations. Black-point ratio: mean ≈0.2263, median ≈0.2181, range ≈0.0925–0.3939. Ingestion, preprocessing, normalization, sampling and Dataset/DataLoader integration passed. `num_workers=0` is the verified Colab setting; the interactive temporary-subclass spawn failure does not establish a repository defect. Use the repository Dataset directly.

The user has now frozen the plant-identity split (364/78/78, seed 42) and completed the 10k-point geometric validation baseline. Reuse that existing manifest unchanged for Phase 3. Preserve dataset revision, code commit, sample count and manifest alongside Colab results.

## Phase 3: PointNet++
The pure-PyTorch SSG network accepts float32 `(B,N,3)` XYZ and returns `(B,N,2)` logits. Training uses unweighted masked cross-entropy and Adam. Default input is 10k points, batch size 2, no SOR and no augmentation. RGB only supplies labels.

1. Install: `python -m pip install -e '.[dev,geometry]'` and run `python -m pytest -q`.
2. Run `scripts/train_pointnet2.py --overfit-samples 4` on the existing frozen manifest. The [workflow](docs/phase3_pointnet2.md) gives complete Colab commands.
3. Inspect `train_eval` in `metrics.json`. A passing eval-mode training check writes `overfit_passed.pt`; regular training requires it via `--overfit-checkpoint`.
4. Train on the 364 training plants; choose `best.pt` by validation mIoU on the 78 validation plants. No test-evaluation path is provided.
5. Use `scripts/validate_pointnet2.py` to reproduce sampled-point validation from a regular-run checkpoint.

User-reported frozen validation geometric baseline: mIoU **0.623**, leaf IoU **0.817**, stalk IoU **0.429**. Initial neural engineering targets: mIoU >0.70, stalk IoU >0.55, leaf IoU approximately >=0.82. These are targets, not measured neural results. Preserve plant grouping, seed 42, and the 364/78/78 frozen split; the training code does not generate splits.

### Phase 3A — validation diagnosis

Use `scripts/analyze_pointnet2.py` with Run 1's `best.pt` and the unchanged frozen manifest to produce per-plant metrics, metadata-conditioned summaries, and optional interactive GT/prediction/error views. The analysis only loads validation plants and does not train or change the model. Install optional dependencies with `pip install -e '.[analysis]'`. See [the Colab analysis command and artifact contract](docs/phase3a_validation_analysis.md).

### Phase 3B — controlled augmentation Run 2

`configs/pointnet2_run2_aug.yaml` changes only training augmentation: Z rotation, scale [0.95, 1.05], and normalized-coordinate jitter σ=0.005 clipped to ±0.02. Validation stays fixed and unaugmented. Run 2 can reuse the existing unaugmented tiny-overfit gate and writes separate `artifacts/pointnet2_run2_aug` outputs. See [the Run 2 Colab commands, compatibility contract, and comparison template](docs/phase3b_run2_augmentation.md).

### Phase 3P — checkpoint recovery and performance

Inspect `last.pt` without loading plants using `scripts/inspect_pointnet2_checkpoint.py`. Use bounded train/val workloads in `scripts/profile_pointnet2.py`, optional `scripts/nsys_pointnet2.py`, and loader/cache/geometry benchmark CLIs before choosing infrastructure options. Existing Run 2 defaults and scientific settings remain unchanged. See [the exact recovery command, Colab profiling commands, measured CPU findings, and limitations](docs/phase3p_training_performance.md).

### Phase 3C — controlled scheduler Run 3

Run 3 returns to Run 1's unaugmented setup and changes only validation-mIoU ReduceLROnPlateau scheduling. See [the exact policy, checkpoint semantics, Colab commands, and comparison table](docs/phase3c_run3_scheduler.md). Real Run 3 results are pending; test stays excluded.
