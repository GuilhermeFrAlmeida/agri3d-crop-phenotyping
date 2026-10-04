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

Provide a verified RGB palette JSON (`rgb_to_class`, keys such as `"R,G,B"`) or an explicit integer mapping (`label_to_class`).
RGB colors never become labels automatically. The dataset fails on missing/unknown annotations.
The official Table S1 palette has not yet been supplied/verified; do not invent it or use the old height heuristic.

```python
from agri3d.training.dataset import SegmentationDataset
from torch.utils.data import DataLoader
# For PLY files already verified to contain semantic integer IDs:
ds = SegmentationDataset('data/splits/manifest.json', 'data/raw', 'train',
                         label_map={0: 0, 1: 1}, num_points=10000)
ds.set_epoch(0)
loader = DataLoader(ds, batch_size=2, num_workers=2, multiprocessing_context="spawn", persistent_workers=False)
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
  --encoding configs/verified_encoding.json --split val --points 10000 --output artifacts/baseline.json
```
The command evaluates every selected sample, accumulates a confusion matrix, masks padding, and reports sampled-point coverage explicitly.
Keep test untouched while tuning. Geometric thresholds are meters; the command restores scale before baseline inference.
Latency is baseline inference only on this machine, not the specification's future GPU inference + phenotyping SLA.
No full-cloud projection or leaf instance counting is implemented yet.
See `docs/phase2_audit.md` for findings and remaining Phase 3 gates.
