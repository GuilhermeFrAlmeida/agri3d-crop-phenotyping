# Agri3D crop phenotyping
Phase 2 geometry baseline and Phase 3 data foundation. PointNet++ is intentionally not implemented.

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

The code is ready for group-aware manifest creation and real baseline evaluation. Remaining input: verified grouping metadata and a frozen manifest; no final split or real baseline score is claimed here. Preserve dataset revision, code commit, sample count and manifest alongside Colab results.
