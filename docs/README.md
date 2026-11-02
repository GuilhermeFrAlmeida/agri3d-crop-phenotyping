# Documentation

Agri3D's current engineering stage is **QUALIFIED WITH LIMITATIONS**. Val78 completed with 156 successful plant/mode executions and zero failures; biological instance/phenotype accuracy and ONNX remain unqualified.

## Start Here

- [Project overview](../README.md) — purpose, concise evidence, installation and workflows.
- [Getting started](getting_started.md) — dependency profiles, Linux runtime, external inputs and execution journeys.
- [Public notebooks](../notebooks/README.md) — inspection, frozen semantic inference and full analysis examples.

## Technical Report

- [Current-stage report index](reports/README.md) — scope, evidence and licensing boundary.
- [Canonical PDF](reports/Agri3D_Current_Stage_Technical_Report.pdf) — primary long-form presentation of current methods, results and limitations.

## Architecture and Dataset

- [Architecture](architecture.md) — model-neutral composition, geometry/model coordinates and PointNet++ hierarchy.
- [Dataset](dataset.md) — external MaizeField3D schema, annotation semantics and frozen split.
- [Dataset and licensing](dataset_and_licensing.md) — Apache-2.0 code, separate CC BY-NC 4.0 data and attribution.

## Results and Experiments

- [Results and qualification matrix](results.md) — historical model comparisons, completed val78 and biological boundaries.
- [Experiment registry](experiments.md) — controlled hypotheses, rejected alternatives and downstream evaluation scope.
- [Phase 4/5 pipeline](phases4_5_phenotyping_pipeline.md) — current inference, component, proxy and artifact contracts.
- [Model interoperability](model_interoperability.md) — canonical `.pt` backend and unqualified ONNX status.

## Performance and Reproducibility

- [Performance](performance.md) — measured T4 throughput, execution controls and caveats.
- [Reproducibility](reproducibility.md) — external manifest/checkpoint requirements, historical training, validation and resume.

## Current Limitations and Roadmap

- [Limitations](limitations.md) — engineering qualification versus biological accuracy and generalization.
- [Roadmap](roadmap.md) — completed work, current release review and explicitly unimplemented directions.
- [Public-release audit](public_release_audit.md) — current tree/history review, document classification and manual release actions.

## Historical Experiment Records

These are **historical technical evidence**, not current installation/status entry points. Their commands preserve original experiment contracts; they do not authorize new training or test evaluation.

- [Phase 2 audit](phase2_audit.md) — data-foundation invariants and early implementation review.
- [Phase 3 PointNet++](phase3_pointnet2.md) — Run 1 protocol and tiny-overfit gate.
- [Phase 3A](phase3a_validation_analysis.md) — sampled-point error-analysis workflow.
- [Run 2 augmentation](phase3b_run2_augmentation.md) — controlled augmentation policy and rejected result.
- [Run 3 scheduling](phase3c_run3_scheduler.md) — plateau scheduler/resume contract and rejected result.
- [Run 4 normals](phase3d_run4_normals.md) — deterministic normal features and rejected result.
- [Phase 3P](phase3p_training_performance.md) — detailed performance methodology and benchmark commands.
- [Phase 4A](phase4a_instance_diagnostics.md) — five-plant development study and configuration frozen before val78.
- [CPU geometry benchmark](benchmarks/phase3p_geometry_cpu.json) — synthetic geometry timings, not T4 measurements.
- [CPU pipeline benchmark](benchmarks/phase3p_pipeline_cpu.json) — synthetic reduced-workload timing evidence.
