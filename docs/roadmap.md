# Roadmap

Status distinguishes implementation from real-data qualification. No production-readiness claim is made.

## Completed

- Dataset validation, RGB supervision and frozen plant-identity split integration.
- Geometric baseline and PointNet++ SSG XYZ Run 1, current semantic champion.
- Validation error analysis and controlled Runs 2/3/4 (augmentation, scheduler, estimated normals), all rejected relative to Run 1.
- Phase 3P profiling, runtime qualification and bounded throughput tools.
- Phases 4/5 software: model-neutral inference, canonical `.pt` adapter, full-cloud nearest projection, geometric instance baseline, metric trait methods, artifact writer, visualization, validation harness and synthetic tests.
- Phase 4A CPU-only val5 graph sweep: 12 configurations, both semantic modes, archived baseline parity and structural/visual diagnostics; private results delivered separately; no production promotion.
- ONNX attempt/qualification tooling with explicit NOT QUALIFIED capture blocker; no portable PointNet++ artifact is qualified.

## In progress

- Phases 4/5 real MaizeField3D development-data qualification.
- Independent verification of exact instance palette and quantitative phenotype references.
- Review and approval of the Phase 4A candidate, then a separately authorized val78 descriptive evaluation. See [the qualification report](phase4a_instance_diagnostics.md).

## Planned / deferred

- Improved touching-leaf splitting, instance reconstruction and trait-reference validation after baseline diagnosis.
- Stronger 3D architectures only after downstream implementation and evaluation; replace the semantic backend rather than rewriting traits.
- Robustness/genotype-held-out and cross-dataset/external LiDAR validation.
- Broader CLI/API packaging and deployment qualification.

The frozen test set remains untouched. Planned items have no completed-result claim.
