> **Historical technical evidence:** this record preserves the phase protocol and measured results. Current project qualification and completed val78 results are in [results](results.md) and the [current-stage report](reports/README.md). Older pending statements describe their original phase, not current status.

# Phase 3A — PointNet++ validation error analysis

> **Historical phase guide:** this document describes the validation-only analysis tooling. Its Run 1 reference metrics remain current; implementation-environment test counts are a phase snapshot.

This phase diagnoses the existing XYZ-only PointNet++ SSG Run 1 checkpoint. It does not train, change the model, change preprocessing, or evaluate test plants. The frozen 364/78/78 plant-level manifest remains the experiment contract.

## Reference experiment

The following are recorded Colab results, not measurements produced by implementing this tooling. Run 1 selected epoch 76 with validation accuracy 0.9358397, mIoU 0.8366027, leaf IoU 0.9202424, stalk IoU 0.7529630, stalk precision 0.8521421, and stalk recall 0.8661208. The geometric validation baseline had mIoU 0.6229862, leaf IoU 0.8173371, and stalk IoU 0.4286353.

The question now is where performance varies among the 78 validation plants. The next experiment should follow this analysis, rather than an architecture or hyperparameter change made in advance.

## Colab command

From the repository directory, install the optional analysis dependencies. Keep Colab's GPU PyTorch installation; the analysis does not require custom CUDA extensions.

```bash
python -m pip install -e '.[dev,geometry,analysis]'
python -m pytest -q

python scripts/analyze_pointnet2.py \
  --manifest "<MANIFEST>" \
  --root "<DATA_ROOT>" \
  --checkpoint "<ARTIFACT_DIR>/pointnet2_run1/best.pt" \
  --metadata "<METADATA_XLSX>" \
  --output-dir "<ARTIFACT_DIR>/pointnet2_run1/validation_analysis" \
  --device cuda --num-workers 0 --top-k 5 \
  --visualize --save-predictions
```

Adjust the manifest, PLY root, and workbook locations to the existing files. Do not regenerate the manifest. Use a new or empty output directory; reruns require another directory. Omit `--metadata` if the workbook is unavailable. Omit `--visualize` and `--save-predictions` for the core metrics pipeline, which needs neither Plotly nor openpyxl. `--top-k` controls summary rankings, not the fixed qualitative case selection.

`--split val` is the only accepted split. Both the CLI and Python API reject train/test analysis before opening a checkpoint or dataset. Whole-manifest structural/leakage validation still runs, but only validation PLY paths are checked and loaded. Reading the supplied workbook does not evaluate any non-validation plants; unused workbook rows contribute only to the join coverage count.

## Evaluation fidelity

The regular-run checkpoint must retain its existing format, configuration, model state, and experiment contract. Analysis restores the original points, seed, model settings, batch size, annotation mode, voxel sampling, normalization, and no-SOR behavior through the existing validation helpers. The manifest's exact byte SHA256 must match the checkpoint contract. There are no point-count, sampling, architecture, or batch-size overrides in this CLI. Class 0 is leaf and class 1 is stalk; RGB is annotation only and never a model input.

Inference uses `model.eval()` and no gradients. Dataset-wide metrics come from a global `SegmentationEvaluator` confusion matrix, not the average of plant metrics. Loss is weighted by valid-point count using the same masked cross-entropy as the existing validator. `validate_pointnet2.py`, training, checkpoint format, and model source are unchanged. Elapsed time naturally differs between the two commands; last-bit numeric reproducibility across hardware/library versions is not guaranteed.

## Artifacts

- `per_sample_metrics.csv`: one row per validation plant with exact `sample_id` and PLY `Tagname`, evaluated valid-point count, accuracy, mIoU, both class IoUs, precision, recall, F1, GT/predicted class counts and ratios, and both directional error counts/rates. If metadata is supplied, all nine source columns are retained plus `metadata_matched`.
- `summary.json`: global metrics/loss, global confusion matrix, plant-metric distributions (mean, median, population std, min, max, available/missing counts), best/worst K by mIoU, lowest stalk recall, highest stalk false-negative rate, metadata coverage/associations/issues, checkpoint/workbook hashes, checkpoint epoch, original contract, and runtime provenance.
- `visualizations/<hash>.html` (optional): standalone Plotly HTML with embedded JavaScript, three interactive panels for GT, prediction, and error map. Panels use identical normalized XYZ points, axis ranges, aspect scale, and synchronized cameras. Only valid points appear. Error colors distinguish correct leaf (green), correct stalk (blue), leaf→stalk false positive (orange), and stalk→leaf false negative (red).
- `predictions/<hash>.npz` (optional): selected samples' exact normalized `points`, `labels`, `predictions`, `valid_mask`, `source_index`, `centroid`, `scale`, and `sample_id`. These arrays include masked padding so alignment can be audited. Load using `numpy.load(path, allow_pickle=False)`. Original coordinates are reconstructed as `points * scale + centroid`; this is still the sampled cloud, not a full-cloud prediction.

Artifact filenames are filesystem-safe SHA256 keys. `summary.json` maps exact sample IDs to paths and selection reasons. Selection is always the best 3 by mIoU, worst 5 by mIoU, and worst 5 by stalk IoU, deduplicated. Ties use lexical sample ID order. At most 13 candidate payloads are retained during the single inference pass; selected visualizations reuse those predictions without resampling or re-running inference.

Counts/ratios exclude padding. `leaf_as_stalk_rate` divides leaf→stalk errors by GT leaf count; `stalk_as_leaf_rate` divides stalk→leaf errors by GT stalk count. The latter equals one minus stalk recall when defined. Ratios divide by evaluated valid points. Undefined metrics retain the evaluator's `None` convention (JSON null, CSV empty cell); absent-class IoUs are excluded from per-plant mIoU and explicitly counted as missing in distributions. An entirely invalid plant fails clearly.

## Metadata contract

The first worksheet must contain the unique columns `Tagname`, `IsConnected`, `IsEasyToCountLeaves`, `AreLeavesTouching`, `Comment`, `HasColor`, `HasEar`, `Size`, and `HasTassel`. Blank rows are ignored. Duplicate or invalid Tagname keys and missing/duplicate required headers fail before inference. Exact matching means **manifest PLY basename, including case and extension, equals Tagname**. No substring matching, extension stripping, identifier guessing, or genotype inference occurs. Ambiguous validation basenames fail. Unmatched validation plants remain in all core metrics and are listed in `metadata_join`.

Categorical summaries cover IsConnected, IsEasyToCountLeaves, AreLeavesTouching, HasEar, and HasTassel. Boolean values, numeric 0/1, and case-insensitive yes/no, true/false, and string 0/1 are normalized explicitly to true/false. Other values (including formulas) are reported as malformed and excluded from that field's summaries. Each group reports plant count, mean/median mIoU and stalk IoU, available/missing metric counts, and a `small_group` flag for fewer than five plants.

Size and sampled GT stalk ratio have pairwise Pearson and Spearman correlations with mIoU/stalk IoU. Size accepts finite numeric cells or numeric strings; booleans, formulas, categorical sizes such as "large", and nonfinite/malformed values are excluded and reported, rather than encoded as invented ordinal measurements. Original workbook values remain in CSV. Correlations require at least three paired plants and nonconstant variables; otherwise they are null with a reason and exclusion count. GT stalk ratio associations are available even without a workbook.

## Interpretation and limits

Per-plant mIoU gives each plant equal weight; global mIoU aggregates all valid-point confusion counts. These quantities need not match. Class prevalence affects metric stability, especially stalk IoU in plants with fewer stalk points. Missing-class conventions can make a one-class plant look unusually strong, so inspect GT ratios alongside rankings.

These are descriptive validation statistics. Small or imbalanced metadata groups and multiple exploratory correlations are not evidence of causality or statistical significance. Size is used only when genuinely numeric, and no genotype information is inferred. Stalk false-negative ranking largely repeats stalk-recall ranking by definition.

All results apply to the checkpoint's fixed normalized sampled points, not the full 100k cloud, instance reconstruction, or an independent held-out test estimate. Qualitative cases are intentionally selected extremes, not a representative random sample. The manifest hash protects split/config identity but does not hash every PLY file; keep the same pinned dataset files. Metadata mismatch diagnostics should be resolved before interpreting metadata-conditioned findings.

## Implementation validation

Tests cover known confusion matrices, masked padding, class ratios, directional errors, absent classes, deterministic rankings and case retention, exact Tagname joins, workbook schema/key failures, malformed/missing metadata, correlations, validation-only rejection, and standalone HTML/NPZ generation. A synthetic RGB-only PLY integration test compares aggregate metrics exactly (excluding elapsed time) against `validate_checkpoint`, repeats per-plant analysis deterministically, checks saved source-index/normalization alignment, and verifies the checkpoint stays byte-identical. Train and test PLY files are deliberately absent.

Real Run 1 artifacts are not present in the implementation environment, so the 78-plant diagnosis must be run in Colab. No new training or real-data analysis results are claimed here.
