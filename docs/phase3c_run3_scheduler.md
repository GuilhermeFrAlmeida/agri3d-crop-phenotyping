# Phase 3C — Run 3: validation-mIoU learning-rate scheduling

## Hypothesis and controlled comparison

A constant Adam LR of 0.001 may be too aggressive late in training. Run 3 tests whether reducing LR when validation mIoU stalls improves generalization and late validation stability. This is a hypothesis, not an established explanation of Runs 1/2.

Run 2 augmentation did not improve aggregate validation performance: stalk precision increased while recall decreased. Run 1 remains the champion. Run 3 therefore returns to **Run 1's unaugmented setup**, starting from the original seed, not from Run 1/2 trained weights. Its only scientific change is ReduceLROnPlateau. Both earlier runs peaked before epoch 100; reported late instability motivates this controlled experiment.

User-reported frozen-validation references:

| Metric | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Selected epoch | 76 | 93 | pending |
| Accuracy | 0.9358397 | 0.9347205 | pending |
| mIoU | 0.8366027 | 0.8295737 | pending |
| Leaf IoU | 0.9202424 | 0.9198950 | pending |
| Stalk IoU | 0.7529630 | 0.7392524 | pending |
| Stalk precision | 0.8521421 | 0.8828109 | pending |
| Stalk recall | 0.8661208 | 0.8196907 | pending |
| Train mIoU at selected epoch | 0.924027 | not supplied | pending |
| LR used at selected epoch | 0.001 | 0.001 | pending |
| Final LR | 0.001 | 0.001 | pending |
| LR reduction epochs | none | none | pending |

## Exact contract

`configs/pointnet2_run3_scheduler.yaml` is identical to `configs/pointnet2.yaml` except for this block:

```yaml
scheduler:
  enabled: true
  name: reduce_on_plateau
  mode: max
  factor: 0.5
  patience: 8
  threshold: 0.0005
  threshold_mode: abs
  min_lr: 0.00001
  cooldown: 0
  eps: 0.00000001
```

The hashed contract additionally records `monitor: val_mIoU` and `step: after_validation`. Initial LR remains 0.001. With PyTorch plateau semantics, 8 consecutive epochs without improvement greater than 0.0005 are tolerated; reduction occurs on the ninth bad epoch. An improvement resets the counter. The scheduler's threshold affects LR scheduling only: checkpoint selection still uses maximum validation mIoU, with lower validation loss breaking exact ties, without this threshold.

Unchanged: MaizeField3D segmented 100k; frozen plant split 364/78/78; seed 42; 10,000 XYZ points; PointNet++ SSG with the Run 1 geometry parameters; batch 2; Adam and initial LR; 100 total epochs; masked unweighted loss; normalization/sampling; no augmentation, RGB, normals, class weighting, added dropout, early stopping, AMP, TF32, or architecture changes. Existing deterministic algorithms/cuDNN settings and default loader/runtime controls are unchanged. Only train and val clouds are loaded. Test remains inaccessible during development.

Order remains load → finite filtering → decode labels → normalize → deterministic sampling → model/Adam updates → frozen deterministic validation → scheduler step → checkpoint save. No augmentation is applied.

## History, selection, and recovery

Each epoch records `epoch`, `train`, `val`, `lr` (LR **used during this epoch**), `next_lr` (optimizer LR **after** the scheduler step, used next epoch), `lr_changed`, and a compact scheduler summary (monitor, best significant metric, bad-epoch count, cooldown counter, last epoch). LR reductions at epoch E take effect in epoch E+1. This avoids assigning a reduced LR retroactively to already-completed updates.

Both `last.pt` and selected `best.pt` save scheduler state after the validation step, together with optimizer/model, history, best score/loss, global RNGs and loader RNGs. `summary.json` includes selected epoch, LR used at that epoch, final optimizer LR, and reduction epochs. Scheduler best significant metric may differ from the selected checkpoint's best mIoU because of the threshold.

Resume from this run's `last.pt`, with the same YAML/options and `--epochs 100` as total target, replacing `--overfit-checkpoint ...` with `--resume .../pointnet2_run3_scheduler/last.pt`. No new gate is needed. Scheduler settings must match exactly and scheduler state must exist. CPU tests verify exact uninterrupted/resumed state across reductions; equality across different hardware/PyTorch/CUDA versions is not promised.

Legacy Run 1/2 config and checkpoints omit the scheduler. Disabled scheduling adds no field to the experiment contract and preserves the legacy hash. Validation and per-plant analysis remain compatible. The existing unaugmented, unscheduled passing tiny-overfit gate is reusable only when all other contract fields match; scheduler use is rejected in tiny-overfit mode. Regular resume never exempts scheduler differences. Do not resume a Run 1/2 checkpoint into Run 3.

The dataset-free inspector also reports scheduler policy/state, LR at the selected epoch, and LR reduction epochs. For legacy history without LR entries, LR at best epoch is unknown (null).

## Exact Colab workflow

Pull the Phase 3C branch, then run from the repository root:

```bash
git fetch origin
git switch phase3c-run3-scheduler
git pull --ff-only
python -m pip install -e '.[dev,geometry,analysis]'
python -m pytest -q

python scripts/train_pointnet2.py \
  --config configs/pointnet2_run3_scheduler.yaml \
  --manifest /content/drive/MyDrive/Agri3D/data/splits/maizefield3d_manifest_seed42.json \
  --root /content/agri3d-crop-phenotyping/data/raw \
  --points 10000 \
  --epochs 100 \
  --batch-size 2 \
  --lr 0.001 \
  --seed 42 \
  --num-workers 0 \
  --device cuda \
  --overfit-checkpoint /content/drive/MyDrive/Agri3D/artifacts/pointnet2_overfit/overfit_passed.pt \
  --output-dir /content/drive/MyDrive/Agri3D/artifacts/pointnet2_run3_scheduler
```

Use a new/empty Run 3 directory. Preserve all prior artifacts. Then evaluate the selected checkpoint on validation only:

```bash
python scripts/validate_pointnet2.py \
  --manifest /content/drive/MyDrive/Agri3D/data/splits/maizefield3d_manifest_seed42.json \
  --root /content/agri3d-crop-phenotyping/data/raw \
  --checkpoint /content/drive/MyDrive/Agri3D/artifacts/pointnet2_run3_scheduler/best.pt \
  --device cuda --num-workers 0 \
  --output /content/drive/MyDrive/Agri3D/artifacts/pointnet2_run3_scheduler/validation.json
```

## Interpretation

Primary success: validation mIoU > Run 1's 0.8366027. Secondary: preserve/improve stalk IoU, avoid meaningful leaf degradation, and reduce late validation instability. Compare a fixed late window (e.g. epochs 81–100) using mIoU standard deviation/range for each run; report these alongside selected-checkpoint metrics, not as substitutes for them. A single seed on a repeatedly examined validation set does not establish statistical significance or test-set generalization. Keep test unopened. No Run 3 result or scheduler success is claimed until the real Colab experiment is completed. Do not add further scientific variants within this run.
