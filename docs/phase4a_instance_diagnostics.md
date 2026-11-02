# Phase 4A — instance diagnostic protocol

## Context

This focused CPU-first tool qualifies `geometric_graph_v1` on the same five previously inspected validation plants: **0006, 0014, 0015, 0022, 0029**. Oracle semantics are primary; unchanged Run 1 projected semantics are secondary. Component count is not verified biological leaf count. No neural inference, training, sampling, projection, phenotype-formula change, test access or val78 evaluation is performed by this runner.

The implementation and synthetic verification are documented here. The completed private val5 experiment's detailed measurements, recommendation receipt and point-cloud views are delivered separately for review. They are intentionally not published in this PR: automatic approval review rejected disclosure of detailed derived private-artifact results. No public instance-accuracy claim or automatic configuration promotion is made.

## Baseline and grid

The production config remains unchanged at radius **0.015 m**, maximum **16 neighbors**, minimum **100 points** and attachment diagnostic radius **0.03 m**.

The exact controlled grid is radii **0.010 / 0.015 / 0.020 / 0.025 m** × neighbor caps **16 / 24 / 32**: 12 configurations. Minimum size and attachment radius remain fixed. Both modes on five plants make 120 extractor calls, oracle first. The graph computation is unchanged; only an undirected-edge metadata counter is added.

See [sweep configuration](../configs/phase4a_instance_sweep.yaml) and [production config](../configs/phenotyping_pipeline.yaml).

## Artifact reuse and isolation

The loader reads metric full-cloud `structure.ply`, aligned semantic/instance label arrays, reports and checksum inventories from existing val5 outputs. It never opens raw dataset PLYs, checkpoint weights or sampled prediction NPZs. Both modes must identify the exact five plants; matching geometry, source indices, scalar labels, model policy, manifest metadata and preprocessing config are checked. Arrays become read-only; reused files are rehashed after execution. An explicit non-validation marker is rejected before extraction. The original batch lacks an explicit split field, so membership remains the caller's prior-val5 assertion constrained to these exact IDs, not a newly verified raw-manifest lookup.

No verified numeric supplementary palette is present in the repository, and the reusable PLYs lack original annotation RGB. The existing verified-palette adapter cannot establish ground truth from these artifacts. Unique component IDs or colors are never treated as biological annotation.

## Diagnostic definitions

All fractions use a stated denominator; undefined quantities are JSON `null` / empty CSV cells.

| Diagnostic | Definition |
|---|---|
| Raw / accepted / rejected counts | Connected components before size filtering / size ≥100 / size <100 |
| Rejected points; unassigned fraction | Sum of rejected sizes; rejected leaf points divided by all semantic leaf points |
| Accepted point coverage | 1 − unassigned fraction; undefined when there are no leaf points |
| Largest / median component | Largest raw component; median **accepted** size (raw median also recorded) |
| Small-component fraction | Rejected components / raw components |
| Raw-to-accepted ratio | Raw components / accepted components |
| Near-threshold fraction | Accepted components of **100–200 points inclusive** / all accepted components |
| Very-small rejected fraction | Rejected components of **≤25 points** / all rejected components |
| Attachment consistency | Accepted components with closest leaf/stalk distance ≤0.03 m / components with an available stalk distance |
| Attachment distances | Mean and max of each accepted component's closest leaf/stalk distance |
| Edge count | Number of undirected edges after graph symmetrization |
| Runtime | Wall milliseconds for `InstanceExtractor.extract` only; excludes loading, diagnostics, serialization and plotting |

Component size distributions and per-component details are persisted. A component is flagged heuristically for size >3× the per-plant accepted median, vertical extent >60% of plant height, nearby stalk-contact Z span >0.15 m, or at least two contact-height groups separated by >0.05 m with ≥5 unique stalk points per group. Contacts are nearest stalk rows within 0.03 m, not biological attachment annotations. These proxies can flag valid curved/large leaves and miss touching leaves at similar heights; their union is **not a merge classifier**. Closest-point attachment consistency alone does not establish instance correctness.

Adjacent radius settings at fixed neighbor cap, and adjacent neighbor caps at fixed radius, are compared. A count drop of ≥25% **and** ≥3 accepted components is a conservative collapse warning. Adjusted Rand index compares leaf-point partitions including noise as a cluster, independent of ID numbering. Absorption counts prior accepted components with ≥90% of their points in one current accepted component; combining multiple prior components is descriptive, not proof of a merge error. IDs/colors are configuration-local.

## Reproduce the bounded sweep

```bash
python -m pip install -e '.[dev,analysis,visualization]'
python scripts/evaluate_instance_sweep.py \
  --input-artifacts <EXTRACTED_VAL5_ROOT> \
  --config configs/phenotyping_pipeline.yaml \
  --sweep-config configs/phase4a_instance_sweep.yaml \
  --mode both --plants 0006 0014 0015 0022 0029 \
  --output-dir <NEW_PHASE4A_OUTPUT>
```

Missing reusable evidence fails; no inference fallback or different plant subset is selected. New-only output directories protect existing artifacts. `--mode oracle` and `--mode predicted` are available, but final qualification requires both modes and exact archived baseline partition reproduction.

## Results, visuals and review

Output files: `summary.json`, `per_plant_config.csv`, `oracle_comparison.csv`, `predicted_comparison.csv`, `runtime.csv`, `recommendation.json`, `configuration.json`, `provenance.json`, `report.md`, `stability.csv`, `partitions/`, `component_details/`, `charts/`, `visualizations/` and a checksum inventory. The mode-join CSV reports oracle/predicted differences; full per-mode diagnostics remain in `per_plant_config.csv`.

One compact HTML and XZ/YZ PNG per plant/mode compare the baseline, lower connectivity (0.010/16), higher connectivity (0.025/32), and the reviewed candidate, deduplicated. Geometry is metric; dark stalk, grey unassigned points and deterministic component colors/IDs remain explicit. Views sample up to 6,500 fixed rows; extraction uses the full cloud. IDs are configuration-local.

Without a bound multi-criteria review, the output conservatively says `KEEP CURRENT DEFAULT` / `NOT READY FOR VAL78`. Review must address fragmentation, absorption/merge proxies, stability, coverage, all five plants, predicted robustness and runtime. Do not minimize component count or tune toward an assumed maize leaf count. Review JSON schema:

```json
{
  "input_artifact_fingerprint": "<FROM_PROVENANCE_JSON>",
  "reviewer": "<REVIEW_METHOD_AND_ATTRIBUTION>",
  "recommended_config_id": "<ENUMERATED_GRID_ID>",
  "decision": "KEEP CURRENT DEFAULT",
  "phase4_readiness": "NOT READY FOR VAL78",
  "reasoning": ["<MULTI_CRITERIA_EVIDENCE>"],
  "plant_observations": {
    "0006": "<OBSERVATION>", "0014": "<OBSERVATION>", "0015": "<OBSERVATION>",
    "0022": "<OBSERVATION>", "0029": "<OBSERVATION>"
  }
}
```

Allowed decisions: `KEEP CURRENT DEFAULT`, `PROMOTE NEW CONFIG`, or `NO CONFIG READY — NEED ALGORITHM CHANGE`. Readiness is `READY FOR VAL78` or `NOT READY FOR VAL78`; it does not imply botanical instance/trait accuracy. For a keep decision use baseline ID `r015_k16`. A promote decision must choose another enumerated ID.

Finalize the new sweep using its private reviewed receipt, without repeating extraction or inference:

```bash
python scripts/evaluate_instance_sweep.py \
  --input-artifacts <EXTRACTED_VAL5_ROOT> \
  --output-dir <COMPLETED_PHASE4A_OUTPUT> \
  --finalize-review --review-json <PRIVATE_REVIEW_JSON>
```

This validates the completed output inventory and the original input fingerprint. A different source bundle needs a fresh review. It changes only the new sweep's review/report/config-copy/selected visual assets and inventory, never the source val5 artifacts or production config. `recommended_pipeline.yaml` is explicitly unpromoted.

## Approval boundary

Do not overwrite `configs/phenotyping_pipeline.yaml`. After explicit approval of the privately reviewed candidate and a val78 run, use its unpromoted config copy:

```bash
python scripts/evaluate_phenotyping_pipeline.py \
  --manifest <FROZEN_MANIFEST> --root <DATA_ROOT> \
  --checkpoint <RUN1_BEST_PT> \
  --config <COMPLETED_PHASE4A_OUTPUT>/recommended_pipeline.yaml \
  --split val --mode both --device cpu --visualize \
  --output-dir <NEW_VAL78_OUTPUT>
```

This command is documented, not executed. No test split is permitted.

## Verification and limitations

Full suite: **201 passed, 4 skipped** for unavailable CUDA/workspace tensor-worker sockets. Changed Python files pass Ruff; repository-wide lint has 45 pre-existing findings in unchanged files. Tests cover grid validation, checksum reuse, exact mode separation, deterministic instances, diagnostics/aggregation, review binding/serialization, rejection of test/cohort/model-policy mismatches, no inference rerun, review reuse without extraction, and unchanged production config.

Five development plants do not establish population or botanical accuracy. Static projections can hide overlapping/touching surfaces. Relative-size and contact proxies can flag valid leaves and miss real merges. Timing is one shared-host wall-time observation per cell, with possible concurrent activity; percentiles across five plants are descriptive rather than repeated-benchmark uncertainty. Cross-version kNN ties may differ; exact archived baseline partition reproduction is required before reviewed qualification. Full validation remains a separate authorized action.
