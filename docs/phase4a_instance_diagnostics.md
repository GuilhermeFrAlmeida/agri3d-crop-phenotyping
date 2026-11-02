# Phase 4A — Instance Reconstruction Diagnostics

**Phase 4A: COMPLETE. Instance configuration: FROZEN.**

> **Historical Phase 4A qualification record:** the configuration freeze below occurred before val78. Full val78 has since completed with 156 successful plant/mode executions and zero failures. Current engineering status is **QUALIFIED WITH LIMITATIONS**; see [results](results.md). Forward-looking wording below preserves the decision-time protocol.

## Objective

Determine whether increased connectivity in `geometric_graph_v1` reconnects artificial leaf fragments without unacceptable component merging, point rejection or CPU cost. This bounded study qualifies a configuration for the next full validation evaluation; it does not establish biological leaf-instance accuracy.

**Component counts are structural diagnostics and are not verified biological leaf counts.** No verified instance-level ground truth is currently available.

## Experimental protocol

The experiment reused existing val5 full-resolution metric geometry, aligned semantic labels and baseline instance partitions. Oracle semantic labels were the primary tuning signal. Unchanged Run 1 projected semantic predictions were a secondary robustness check, keeping semantic-model and graph errors distinguishable.

Twelve configurations × five plants × two semantic modes produced **120 instance extractions**. All ten archived initial-configuration partitions reproduced exactly. No semantic inference, training, projection, split regeneration, test evaluation or val78 execution occurred. Run 1, 10k neural sampling, semantic labels, projection and phenotype formulas remained unchanged. Only graph radius and neighbor cap varied. The extraction algorithm was not replaced.

## Parameter grid

| Parameter | Evaluated values |
|---|---|
| Neighborhood radius | 0.010, 0.015, 0.020, 0.025 m |
| Maximum neighbors | 16, 24, 32 |
| Minimum accepted component size | 100 points, fixed |
| Attachment diagnostic radius | 0.03 m, fixed |

The historical initial development setting was **0.015 m / 16 neighbors**. The selected frozen setting is **0.020 m / 32 neighbors**. The [sweep grid](../configs/phase4a_instance_sweep.yaml) remains unchanged. The [historical input configuration](../configs/phenotyping_pipeline_phase4a_initial.yaml) is retained for original val5 artifact replay; it is not the production default.

## Frozen development cohort

The study used exactly **0006, 0014, 0015, 0022 and 0029**, the same five validation plants inspected previously. No replacement plants or additional validation plants were used for tuning. This subset is a development/qualification cohort, **not an independent final validation cohort**. The dataset's plant-identity split remains 364 train / 78 validation / 78 test, seed 42; test remains untouched.

## Oracle-semantic results

Measured accepted geometric component counts:

| Plant | Initial development setting | Selected setting |
|---|---:|---:|
| 0006 | 15 | 13 |
| 0014 | 15 | 14 |
| 0015 | 25 | 18 |
| 0022 | 9 | 9 |
| 0029 | 17 | 13 |
| **Total** | **81** | **67** |

**Interpretation:** additional connectivity reduced partition fragmentation on several plants. A smaller count alone is not evidence of better biological segmentation: distinct touching leaves can also merge. The decision therefore included size/coverage diagnostics, stability and visual inspection rather than count minimization.

## Predicted-semantic results

Measured accepted counts with the unchanged Run 1 projected labels:

| Plant | Initial development setting | Selected setting | Selected predicted − oracle |
|---|---:|---:|---:|
| 0006 | 20 | 17 | +4 |
| 0014 | 16 | 14 | 0 |
| 0015 | 26 | 19 | +1 |
| 0022 | 16 | 16 | +7 |
| 0029 | 19 | 15 | +2 |
| **Total** | **97** | **81** | **+14** |

Plant **0022 retains a seven-component oracle/predicted gap**. The oracle/predicted contrast strongly associates this issue with semantic-model error; graph connectivity alone does not resolve it. This is a diagnostic association, not instance-GT verification. Parameters were not selected to force 0022 toward an assumed biological count.

## Fragmentation diagnostics

The near-threshold band is **100–200 points inclusive** among accepted components, with minimum accepted size fixed at 100.

| Oracle diagnostic | Initial development setting | Selected setting |
|---|---:|---:|
| Near-threshold accepted components / all accepted components | 9 / 81 | 4 / 67 |

**Interpretation:** fewer accepted components close to the cutoff support the hypothesis that added connectivity reconnects some detached blade/tip fragments. This is not proven biological leaf recovery.

### Unassigned-point behavior

**Measured result:** candidate unassigned fractions remained **below 0.177%** across both oracle and predicted semantic modes. The denominator is all semantic leaf points in the relevant plant/mode; unassigned points are rejected small-component leaf points.

This indicates that increased connectivity did not cause substantial additional point rejection. It does not prove correct instance identities or exclude merging.

### Diagnostic definitions and stability

The sweep records raw connected components before filtering, accepted/rejected counts, rejected point counts, unassigned fraction, accepted point coverage, largest raw and median accepted size, component-size distributions, raw/accepted ratio, attachment diagnostics, graph edge count and extractor runtime.

Small-component fraction means rejected / raw components. Very-small rejected fraction uses rejected components of ≤25 points. Attachment consistency means accepted components with an available closest stalk distance ≤0.03 m; it is a geometric proximity proxy, not an anatomical annotation.

Conservative merge-risk flags cover size >3× the accepted median, vertical extent >60% of plant height, contact-height span >0.15 m, or multiple contact-height regions separated by >0.05 m with ≥5 unique stalk points per group. Relative-size flags can disappear when the median grows. None is a definitive merge classifier.

Adjacent settings are compared by component count, leaf-partition adjusted Rand index and component absorption. A ≥25% and ≥3-component count drop is a warning, not a verified merge. A prior component is counted as absorbed when ≥90% of its points map to one current accepted component.

**Plant 0015 remains parameter-sensitive.** The selected setting balances connectivity with retaining separation; full parameter invariance is not claimed. No expected maize leaf-count range was used as a tuning target.

## Runtime tradeoff

Measured **CPU instance-extractor** wall times combine the **ten plant/mode observations** at each compared setting. Loading, additional diagnostics, serialization, projection and phenotype extraction are outside this timing scope.

| Configuration | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| Previous development default | 3.22 s | 3.25 s | 3.78 s | 3.88 s |
| Selected candidate | 4.14 s | 4.19 s | 4.51 s | 4.52 s |

These observations came from a **shared execution environment**. Percentiles describe these observations, not repeated-run uncertainty. They are not production-latency claims and do not establish an isolated benchmark speed regression. The selected configuration trades additional CPU time for a more connected structural partition; no speedup is claimed.

## Visual review findings

Qualitative metric-space review supported reconnection of detached blade/tip portions while preserving recognizable large-scale plant structure. Component-size/absorption checks were considered alongside the views; visually plausible reconnection was not treated as verified biological identity. Residual sensitivity on 0015 and the semantic-mode discrepancy on 0022 remain explicit.

Static geometric diagnostics cannot exclude all merge cases. Increased connectivity can reconnect fragments **and increase merge risk**. No point-cloud renders or raw dataset content are redistributed with this report; public redistribution of such assets has not been established here.

## Selected configuration

The approved [production/default configuration](../configs/phenotyping_pipeline.yaml) is:

```yaml
instances:
  method: geometric_graph_v1
  neighborhood_radius_m: 0.020
  max_neighbors: 32
  min_leaf_points: 100
  attachment_radius_m: 0.03
```

| Setting | Initial development | Phase 4A frozen |
|---|---:|---:|
| Radius | 0.015 m | 0.020 m |
| Maximum neighbors | 16 | 32 |
| Minimum component size | 100 | 100 |
| Attachment radius | 0.03 m | 0.03 m |

The choice combines lower near-threshold fragmentation, improved structural connectivity, low unassigned fractions, acceptable qualitative visual behavior and a known CPU-runtime tradeoff. It does **not** claim to maximize biological accuracy. Unrelated configuration values and the geometric extraction algorithm are preserved. Python API defaults and YAML defaults agree; historical replay uses an explicit initial configuration.

## Limitations

- No verified instance-level ground truth is currently available; component count is not biological leaf count.
- Plant 0015 remains parameter-sensitive.
- Plant 0022 retains a seven-component oracle/predicted gap strongly associated with semantic-model error, unresolved by graph connectivity alone.
- Static diagnostics cannot exclude all merges; increased connectivity can merge distinct structures.
- Five development plants do not constitute an independent final validation cohort.
- Shared-environment CPU timings are descriptive and do not establish deployment latency.
- Full Phase 4/5 validation and biological phenotype accuracy remain pending.
- No tuning is permitted on val78 after this configuration freeze.

## Qualification decision

- **Phase 4A: COMPLETE**
- **Decision: PROMOTE NEW CONFIG**
- **Instance configuration: FROZEN**
- **Readiness: READY FOR VAL78**
- **Decision-time Phase 4/5 status: IMPLEMENTED — FULL VALIDATION PENDING**

Readiness means a fixed configuration for a descriptive full validation evaluation, not established biological leaf-instance or phenotype accuracy. The approved measured tables are sanitized; no review receipt, storage/account identifiers, manifests, checkpoint locations or dataset artifacts are published.

## Historical next-step protocol

**The 78-plant validation execution is an evaluation run, not a tuning run.**

Run the frozen configuration once the full evaluation is scheduled. If val78 exposes failures, report them without retrospectively tuning against the same 78 plants. Any algorithmic revision is a new experiment requiring an explicit development protocol. The five Phase 4A plants overlap the full validation cohort; interpretation must retain this dependence. Advanced model work remains deferred.

The following command is provided for the next experiment; **it was not executed during this publication/configuration-freeze task**:

```bash
python scripts/evaluate_phenotyping_pipeline.py \
  --manifest <FROZEN_MANIFEST> --root <DATA_ROOT> \
  --checkpoint <RUN1_BEST_PT> \
  --config configs/phenotyping_pipeline.yaml \
  --split val --mode both --device cpu --visualize \
  --output-dir <NEW_VAL78_OUTPUT>
```

Omitting `--max-plants` requests the full validation split. Supply the existing frozen manifest and Run 1 selected checkpoint; do not regenerate the split or evaluate test.

### Historical sweep reproducibility

For authorized original val5 artifacts, the historical sweep can be replayed with its explicit original input contract. This command is documentation only; no sweep is rerun in this task:

```bash
python scripts/evaluate_instance_sweep.py \
  --input-artifacts <EXTRACTED_VAL5_ROOT> \
  --config configs/phenotyping_pipeline_phase4a_initial.yaml \
  --sweep-config configs/phase4a_instance_sweep.yaml \
  --mode both --plants 0006 0014 0015 0022 0029 \
  --output-dir <NEW_SWEEP_OUTPUT>
```

The loader preserves exact cohort/mode separation, checksum and alignment checks, read-only inputs and non-validation rejection. Historical source artifacts are never interpreted as having been generated with the newly promoted settings.
