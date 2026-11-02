# Reproducibility

## Environment

Start with [getting started](getting_started.md) for dependency profiles, Ubuntu `libegl1`, notebooks and external checkpoint/manifest requirements. Canonical CI is Python 3.11 / Ubuntu 24.04; ≥3.10 is the declared minimum, not a fully tested version matrix.

Use Python 3.10 or newer and install a PyTorch build appropriate for the target device before the project extras. On a CPU machine:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,geometry,analysis]'
pytest -q
```

For a CUDA machine, install the matching CUDA-enabled PyTorch build according to the official PyTorch installation selector, then install the project extras. Do not replace a hosted CUDA build with a CPU wheel.

## Dataset and frozen manifest

Install download dependencies with `python -m pip install -e '.[data]'` if needed. Acquire the segmented 100k MaizeField3D archive under its separate CC BY-NC 4.0 terms. Pin the upstream Hugging Face dataset revision and keep it with the experiment metadata:

```bash
DATASET_REVISION="<pinned-hugging-face-commit>"
python scripts/download_ground_truth.py --revision "$DATASET_REVISION" --output "<DATA_ROOT>"
```

The split manifest is an external, frozen input. Use the existing file with 364/78/78 plant-identity records and seed 42; do not regenerate it. Store the manifest and dataset outside the source checkout. Keep exact manifest bytes, dataset revision, source commit, config and artifact directory with every experiment.

## Scientific configuration and runtime settings

Scientific YAMLs define data features, model, optimizer, learning schedule and other experiment decisions. Runtime CLI controls define workers, pinning, nonblocking transfer, prefetch, cache location, profiling and output paths. Runtime settings do not change the scientific run contract when they pass the identity/equality checks, but they are recorded for audit and resume may enforce matching loader settings.

Configs:

- `configs/pointnet2.yaml` — Run 1 XYZ reference.
- `configs/pointnet2_run2_aug.yaml` — Run 2 augmentation.
- `configs/pointnet2_run3_scheduler.yaml` — Run 3 scheduler.
- `configs/pointnet2_run4_normals.yaml` — Run 4 estimated normals.

All neural runs use seed 42, 10,000 points, PointNet++ SSG, batch size 2, Adam with initial learning rate 0.001, and 100 regular epochs. Run 1 is unaugmented and unscheduled. Later run-specific differences are documented in [experiments](experiments.md).

## Baseline reproduction sequence

Use the frozen manifest, no augmentation, and the required four-plant tiny-overfit gate before regular training:

```bash
MANIFEST="<path-to-frozen-manifest>"
DATA_ROOT="<path-to-segmented-ply-root>"
ARTIFACT_DIR="<path-to-artifacts>"

python scripts/train_pointnet2.py \
  --config configs/pointnet2.yaml --manifest "$MANIFEST" --root "$DATA_ROOT" \
  --points 10000 --epochs 200 --batch-size 2 --lr 0.001 --seed 42 \
  --overfit-samples 4 --output-dir "$ARTIFACT_DIR/pointnet2_overfit"

python scripts/train_pointnet2.py \
  --config configs/pointnet2.yaml --manifest "$MANIFEST" --root "$DATA_ROOT" \
  --points 10000 --epochs 100 --batch-size 2 --lr 0.001 --seed 42 \
  --overfit-checkpoint "$ARTIFACT_DIR/pointnet2_overfit/overfit_passed.pt" \
  --output-dir "$ARTIFACT_DIR/pointnet2_run1"
```

Do not run the regular command until a real gate checkpoint exists and passes the documented criteria. The gate is a compatibility check, not initial weights for the full run.

## Validation and analysis

Validation is exposed separately from training. It restores the checkpoint's settings and uses the matching manifest. The analysis CLI also restricts itself to validation:

```bash
python scripts/validate_pointnet2.py \
  --manifest "$MANIFEST" --root "$DATA_ROOT" \
  --checkpoint "$ARTIFACT_DIR/pointnet2_run1/best.pt" \
  --output "$ARTIFACT_DIR/pointnet2_run1/validation.json"

python scripts/analyze_pointnet2.py \
  --manifest "$MANIFEST" --root "$DATA_ROOT" \
  --checkpoint "$ARTIFACT_DIR/pointnet2_run1/best.pt" \
  --split val --output-dir "$ARTIFACT_DIR/pointnet2_run1/validation_analysis"
```

These metrics concern 10,000 fixed sampled points per plant, not projected predictions over the full cloud. The neural/downstream validation harnesses restrict evaluation to validation. Legacy baseline split options and standalone PLY entry points do not enforce membership for arbitrary files; the project policy still prohibits test evaluation.

## Historical Run 4 gate and commands

Run 4 completed and was rejected. Its historical procedure required its own XYZ+normal tiny-overfit checkpoint; the XYZ Run 1 gate is not sufficient. See [Phase 3D](phase3d_run4_normals.md) for the normal audit and detailed command sequence. The full run uses the same 10k points, batch 2, Adam, LR 0.001 and 100 epochs, and changes only the point-feature contract.

## Checkpoints, resume and artifacts

`best.pt` is selected by validation mIoU. `last.pt` contains recovery state including model, optimizer, RNG/loader state, history and relevant scheduler state. Resume with the same config/manifest/output directory and compatible runtime settings. Feature, scheduler, manifest or scientific-contract mismatches are rejected. Exact bitwise equivalence is only claimed where tests establish it; hardware/library changes may alter results.

Typical outputs are `run.json`, `metrics.json`, `summary.json`, `last.pt`, `best.pt`, validation JSON and optional analysis files. Keep artifacts and caches out of Git; inspect `.gitignore` before staging.

## Colab

In Colab, clone the reviewed source branch, install package extras without replacing its CUDA PyTorch build, place the dataset on local runtime disk for reads, and keep manifests/checkpoints on persistent storage. Use shell variables for local locations rather than committing mounted-drive paths. Qualified runtime controls are documented in [performance](performance.md); scientific batch size remains 2.

## Frozen inference and Phases 4/5

No new training is required: supply Run 1 `best.pt` to `python -m agri3d.pipeline analyze`. The backend verifies checkpoint contracts; the manifest evaluation harness also checks its exact-byte manifest hash and accepts validation only. Use the exact manifest sample ID for epoch-0 sampling parity. Measurements use full cleaned metric geometry, never model-normalized coordinates.

[Phases 4/5 commands and artifact contract](phases4_5_phenotyping_pipeline.md) include single-plant CPU inference, bounded five-plant validation and all-78 validation experiments. Oracle mode can run geometry without a model. Public code accepts externally stored checkpoint paths and never redistributes weights/data. Where Git metadata is unavailable, supply `--code-commit` with the reviewed full SHA. Optional ONNX attempts and qualification rules are in [model interoperability](model_interoperability.md); current export is NOT QUALIFIED.

## Frozen downstream evaluation

The default instance graph is 0.020 m / 32 neighbors, minimum 100 points and attachment radius 0.03 m. It was frozen before the completed val78 run: 156 successful plant/mode executions, zero failures. See [results](results.md) and [the current-stage report](reports/README.md). No metric is recomputed by this guide.

```bash
python scripts/evaluate_phenotyping_pipeline.py \
  --manifest <MANIFEST> --root <DATA_ROOT> --checkpoint <CHECKPOINT> \
  --config configs/phenotyping_pipeline.yaml --split val --mode both \
  --device cpu --output-dir <OUTPUT_DIR>
```

Use a new output directory and the exact external artifacts. Val78 was evaluation, not tuning; a repetition cannot silently become parameter optimization. No test access is authorized. For single-plant artifact generation use [getting started](getting_started.md). Historical Phase 4A replay requires `configs/phenotyping_pipeline_phase4a_initial.yaml`, not current defaults.

## CI and evidence provenance

The public suite uses synthetic fixtures and synthetic notebook backends, no dataset/checkpoint download during tests, and no required GPU. Ubuntu CI installs `libegl1` before Open3D import. The canonical dependency groups are described in [getting started](getting_started.md). CI resolves supported packages rather than asserting a fully pinned numerical environment; experiment provenance records exact versions and contract hashes.

The current-stage PDF and approved aggregate summary support the published val78 figures. The repository deliberately does not contain the original dataset, checkpoints, private manifest or raw val78 bundle. Scientific reproduction therefore requires obtaining the exact external inputs; passing synthetic tests alone does not reproduce model accuracy.
