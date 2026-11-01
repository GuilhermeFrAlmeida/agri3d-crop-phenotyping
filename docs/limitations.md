# Limitations

- Results are validation-only; the test split has not been evaluated.
- Current neural comparisons use one seed and do not establish statistical significance.
- Published neural comparisons remain fixed 10,000-point validation scores. Full-cloud projection evaluation is now implemented but has no real-data result yet.
- The frozen split is grouped by plant identity, not held out by genotype.
- No cross-dataset validation has been performed.
- RGB supervision uses a dataset-specific black/non-black encoding; it is not a general semantic label convention.
- Estimated Run 4 normals used a canonical mathematical sign convention, not physical outward orientation. The measured normal-feature run did not improve validation mIoU.
- Geometric instances/trait measurements are implemented but unqualified on real plants. Touching leaves may merge and disconnected leaves may split. PCA/graph lengths and hull/voxel volumes are geometric proxies, not biologically validated traits.
- The PointNet++ geometry operators are pure PyTorch and are not production-optimized; FPS and neighborhood operations require substantial geometry work.
- T4 benchmark throughput is bounded and hardware-specific; it does not establish production latency or general deployment performance.

- Modern ONNX capture currently fails at an existing tensor-boolean validation guard; PyTorch remains canonical. Standalone PLY inference cannot infer held-out membership from filenames; use development files only and the validation-only manifest harness for experiments.
