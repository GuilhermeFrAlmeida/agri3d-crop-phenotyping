# Limitations

**Current status: QUALIFIED WITH LIMITATIONS** at the evaluated engineering scope. Read the [qualification matrix](results.md#qualification-matrix) alongside performance figures.

- Evidence is validation-only; the test split remains untouched. Neural comparisons use one seed and do not establish statistical significance.
- Validation has informed model selection and multiple development experiments. The five Phase 4A plants overlap the completed val78 cohort; it is not an independent final holdout.
- Published Run 1 selection metrics and later val78 sampled metrics are distinct recorded evaluations with a small numerical difference. They are not silently substituted; raw reproduction requires matching external artifacts/environment.
- Full-cloud projection is evaluated on val78 and preserves aggregate semantic quality. It does not intrinsically improve model generalization or prove pointwise/topological correctness.
- The split groups plant identity, not genotype; no cross-dataset or external LiDAR validation is complete.
- MaizeField3D black/non-black RGB is dataset-specific supervision, not a general semantic convention and never model input.
- Components are not verified biological leaves. Touching structures can merge; disconnected portions can split. Static diagnostics and a stable component count cannot exclude identity errors.
- Height is computationally qualified as full-cloud Z extent, not independently validated biological height. PCA/geodesic lengths and hull/voxel volumes are geometric proxies, not biologically validated traits. Leaf instance segmentation accuracy and final phenotype accuracy remain unqualified.
- Run 4 normals use a canonical mathematical sign, not physical outward orientation, and did not improve the baseline.
- Pure-PyTorch PointNet++ geometry is not production-optimized. T4 microbenchmarks and shared-host CPU measurements do not establish deployment latency or universal speedups.
- PyTorch `.pt` is canonical. ONNX remains unqualified because of the documented capture/parity blocker. A trained checkpoint and exact research manifest are external and are not distributed here.
- Standalone PLY commands cannot infer split membership. Supply development data only. Legacy baseline CLI split options do not authorize test evaluation.
- Val78 is complete and cannot become a retrospective tuning set. Any revision needs a new experiment/development protocol.

These boundaries distinguish reproducible engineering execution from biological validity. See [results](results.md), [dataset licensing](dataset_and_licensing.md) and [roadmap](roadmap.md).
