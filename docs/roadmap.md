# Roadmap

Agri3D is an ongoing research-engineering project. Its current engineering stage is **QUALIFIED WITH LIMITATIONS**; this is not biological phenotype or deployment qualification. See the [current-stage report](reports/README.md).

## Completed

- Dataset validation and frozen plant-identity split: 364 train / 78 validation / 78 untouched test plants.
- Geometric baseline; frozen PointNet++ XYZ Run 1; validation error analysis.
- Controlled augmentation, scheduling and estimated-normal experiments, rejected relative to Run 1.
- Phase 3P profiling, qualified T4 execution settings and bounded throughput measurements.
- Model-neutral `.pt` inference, deterministic full-cloud projection, geometric components, metric proxy methods and checksummed artifacts.
- Phase 4A five-plant configuration study; **0.020 m / 32 neighbors** frozen before val78, with minimum 100 points and attachment radius 0.03 m unchanged.
- Full val78 engineering evaluation: **156 successful plant/mode executions, zero failures**. Projection preserved aggregate semantic quality; structural and phenotype-accuracy limitations remain explicit.
- Current-stage technical report and public execution notebooks.

## In progress

- Public release-candidate review, documentation and CI/environment hardening.
- Verification strategy for biological instance identities and quantitative phenotype references. Existing repository adapters do not constitute verified ground truth.

## Planned

### Advanced semantic architecture

Evaluate a stronger 3D network, such as a Point Transformer/KPConv-class architecture, through the model-neutral interface. Compare semantic and full-cloud mIoU, downstream structural stability, runtime, GPU memory and parameter count. No such replacement is implemented by this release.

### Robustness experiments

Use explicit development protocols for point dropout, coordinate noise, density reduction, rotations, occlusion and partial scans. Measure degradation rather than only nominal accuracy; do not tune against the completed val78 evaluation retrospectively.

### External datasets and domain transfer

Evaluate additional public 3D datasets. **Pullally is a planned sensor-native external integration target**, involving LiDAR, GNSS/RTK, robotic raw sensing, registration and spatial mapping. No Pullally adapter, registration pipeline or domain-transfer result exists yet. Genotype-held-out and cross-dataset qualification also remain future work.

### Industrial interface

Stabilize the existing CLI and JSON contracts; investigate REST/API access, automated reports, deployment-oriented configuration, containerization and runtime observability. Existing development tooling is not a qualified deployment service. No package/model hosting or release automation is provided in this stage.

## Scientific boundary

The 78-plant validation execution was evaluation, not tuning. Report failures; any new algorithmic revision is a new experiment with an explicit development protocol. The frozen test split remains untouched. Planned work must not be represented as measured implementation.
