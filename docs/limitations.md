# Limitations

- Results are validation-only; the test split has not been evaluated.
- Current neural comparisons use one seed and do not establish statistical significance.
- Evaluation is on fixed 10,000-point samples, not full-cloud projected predictions.
- The frozen split is grouped by plant identity, not held out by genotype.
- No cross-dataset validation has been performed.
- RGB supervision uses a dataset-specific black/non-black encoding; it is not a general semantic label convention.
- Estimated Run 4 normals have a canonical mathematical sign convention, not physical outward orientation; quality and predictive value remain to be assessed on real plants.
- Leaf-instance segmentation and final phenotypic trait extraction are not implemented as a validated end-to-end workflow.
- The PointNet++ geometry operators are pure PyTorch and are not production-optimized; FPS and neighborhood operations require substantial geometry work.
- T4 benchmark throughput is bounded and hardware-specific; it does not establish production latency or general deployment performance.
