# Performance engineering

Phase 3P adds profiling and bounded runtime benchmarks around the existing PointNet++ implementation. These measurements support engineering choices; they do not alter scientific experiment contracts.

## T4 observations

Recorded Colab measurements used a Tesla T4. Peak allocated and reserved VRAM were approximately 325 MB and 359 MB. A late Run 2 training epoch took roughly 136–140 seconds and validation about 26 seconds. The low memory footprint indicates VRAM capacity is not the current constraint; epoch duration alone does not attribute time to data, kernels, or Python overhead.

The pure-PyTorch PointNet++ geometry path is an important profiling target. Farthest-point sampling in the first Set Abstraction stage processes 10,000 input points to choose 1,024 centroids, with theoretical work O(B × N × npoint); neighborhood queries and interpolation also perform geometry work. A per-stage profile is needed to quantify each contribution. The checked-in full-size geometry timings are synthetic CPU measurements, not T4 timings.

For context only, the synthetic CPU report measured data wait at about 82.8% of a reduced 256-point profile, while a separate full-size CPU microbenchmark measured first-stage FPS at about 255 ms (B=2, N=10,000, S=1,024). Those workloads do not identify the T4 critical path and are not speedup estimates.

## Loader and runtime findings

The qualified T4 runtime controls are:

```text
num_workers=2
pin_memory=true
non_blocking=true
prefetch_factor=2
persistent_workers=false
```

These settings are execution controls. Their use does not change point identity or the scientific contract under the repository's reproducibility checks. The project does not claim that every control independently produced a measured speedup; the full loader/cache matrix and exact-equality checks are described in [Phase 3P](phase3p_training_performance.md).

## Batch-size throughput benchmark

The bounded T4 throughput benchmark reported:

| Benchmark batch size | Samples/s | Approx. train-set pass estimate at 364 plants |
|---:|---:|---:|
| 2 | 2.58 | 141 s |
| 4 | 3.58 | 102 s |
| 8 | 4.28 | 85 s |

The estimates are 364 divided by the measured samples/s; they are not observed full training-epoch times. Larger benchmark batches improve throughput, but change optimizer update grouping and BatchNorm statistics. **Official scientific experiments keep batch size 2.** Batch-size variants are runtime throughput measurements only and would require a new scientific experiment to become official training settings.

The baseline full training epoch observation of 136–140 seconds is in the same range as the batch-2 estimate, but the benchmark and full run have different measurement conditions. Do not present the difference as a verified speedup.

## Interpretation and scope

The profiler aggregates a continuous window and reports CPU/CUDA time, memory, operation families, PointNet++ stages and logical ranges for FPS, neighborhood query, interpolation, forward, backward, optimizer and host-to-device copies. The current repository also has bounded DataLoader/cache and disposable batch-size CLIs. Measurement synchronization and profiler overhead can change overlap; diagnostic percentages are not utilization figures.

For exact commands, synthetic CPU references, cache caveats and report templates, see [Phase 3P training performance](phase3p_training_performance.md). AMP, TF32, custom CUDA kernels, third-party PointNet++ operations and larger official batch sizes are not enabled.

## Downstream evaluation context

The frozen pipeline has completed val78 engineering evaluation; see [results](results.md) and the [current-stage report](reports/README.md). The Phase 4A shared-environment CPU comparison is a configuration tradeoff, not an isolated production-latency benchmark. Training throughput figures above remain distinct from inference/structure/trait stage timing. No new performance experiment is performed by the public-release preparation.
