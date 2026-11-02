# Documentation

## Overview

- [Project README](../README.md) — concise project purpose, status and starting points.
- [Roadmap](roadmap.md) — completed, in-progress and planned work.
- [Limitations](limitations.md) — evaluation and implementation boundaries.

## Architecture

- [System architecture](architecture.md) — preprocessing, training, model and reproducibility diagrams.

## Dataset

- [Dataset guide](dataset.md) — MaizeField3D subset, labels, split and sampling contract.
- [Dataset and licensing](dataset_and_licensing.md) — code license, external data terms and attribution.

## Experiments

- [Experiment registry](experiments.md) — controlled baseline and Runs 1–4.
- [Results](results.md) — validation-only metrics and decisions.

## Performance Engineering

- [Performance summary](performance.md) — Phase 3P profiling observations, throughput and qualified runtime controls.
- [Detailed Phase 3P report](phase3p_training_performance.md) — profiling commands, cache and benchmark methodology.

## Reproducibility

- [Reproducibility guide](reproducibility.md) — install, data, gate, train, validate, analyze and resume.

## Current Development

- [Phase 4A diagnostics](phase4a_instance_diagnostics.md) — completed CPU val5 connectivity sweep; artifact reuse, diagnostics and explicit review before val78.
- [Phases 4/5 pipeline](phases4_5_phenotyping_pipeline.md) — inference/structure/trait implementation contracts; full validation and biological accuracy remain unqualified.
- [Model interoperability](model_interoperability.md) — canonical PyTorch backend and explicit ONNX blocker/qualification protocol.
- [Phase 3D: Run 4 normals](phase3d_run4_normals.md) — completed normal-feature comparison, rejected.
- [Public-release audit](public_release_audit.md) — repository and history review with pre-publication action.

## Historical / Phase Notes

These documents preserve implementation detail. The project README, results and roadmap are the current summaries; older “pending” language is a snapshot unless explicitly updated.

- [Phase 2 audit](phase2_audit.md) — early data-foundation review; historical snapshot.
- [Phase 3 PointNet++](phase3_pointnet2.md) — baseline implementation and workflow.
- [Phase 3A validation analysis](phase3a_validation_analysis.md) — validation-only diagnosis tooling.
- [Phase 3B Run 2](phase3b_run2_augmentation.md) — augmentation contract and Run 2 result.
- [Phase 3C Run 3](phase3c_run3_scheduler.md) — scheduler contract and Run 3 result.
- [Initial development plan](devlopment_plan.md) — historical planning record; not an implementation status source.
- [Initial system specification](project_specification.md) — original target outline, not a claim that all target features exist.
- [CPU geometry benchmark](benchmarks/phase3p_geometry_cpu.json) and [CPU pipeline benchmark](benchmarks/phase3p_pipeline_cpu.json) — synthetic CPU measurements, not T4 results.
