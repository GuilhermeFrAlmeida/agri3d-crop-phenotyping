# Phase 3P — recovery, profiling, and Colab efficiency

> **Phase record:** the early sections document recovery and profiling as implemented. Run 2 has since completed; later T4 observations and throughput measurements are recorded in the Phase 3P continuation below and summarized in [performance](performance.md).

## Purpose and scientific boundary

Run 2 was interrupted around epoch 70 by Colab runtime limits while validation mIoU was still improving; it was later completed. This phase added recovery inspection and bounded measurements to profile GPU workloads. It is infrastructure work, not another scientific experiment.

The official Run 2 YAML is unchanged: frozen plant split 364/78/78, 10k XYZ points, PointNet++ SSG, batch 2, Adam LR 0.001, seed 42, Z rotation, scale [0.95, 1.05], jitter σ=0.005/clip=0.02, masked unweighted loss, 100 total epochs, and validation-mIoU selection. Deterministic algorithms, disabled cuDNN benchmark, and disabled TF32 remain unchanged. No test PLY is loaded by profiling or benchmarking. No architecture/kernel replacement, AMP, class weighting, scheduler, or early stopping is enabled.

The recovery instructions below preserve the historical Run 2 workflow. Do not profile concurrently with training on the same GPU: contention invalidates throughput measurements. A disposable profiling process never saves weights or optimizer state back into the experiment.

## Exact Run 2 recovery

PyTorch `.pt` files contain recovery state; a TensorFlow `.pb` is not used. `last.pt` is the latest completed epoch with model, optimizer, history, best-score/loss, global RNGs, DataLoader RNGs, and contract. `best.pt` is for selected-checkpoint validation. Keep both and the run directory; do not initialize a fresh run or replace `last.pt` with `best.pt`.

```bash
python scripts/train_pointnet2.py \
  --config configs/pointnet2_run2_aug.yaml \
  --manifest "<MANIFEST>" \
  --root "<DATA_ROOT>" \
  --points 10000 \
  --epochs 100 \
  --batch-size 2 \
  --lr 0.001 \
  --seed 42 \
  --num-workers 0 \
  --device cuda \
  --resume "<ARTIFACT_DIR>/pointnet2_run2_aug/last.pt" \
  --output-dir "<ARTIFACT_DIR>/pointnet2_run2_aug"
```

`--epochs 100` means total target, not 100 extra epochs. If inspection says completed epoch 70, the next completed epoch is 71. Interrupted work inside an unsaved epoch is repeated; no minibatch-level recovery is claimed. Resuming needs no `--overfit-checkpoint`: gate evidence is already in the checkpoint. Existing format/contract hashes are preserved. Keep the same hardware/software for strongest reproducibility; the contract does not promise bitwise equality across different PyTorch/CUDA implementations.

Historical Run 1 and Run 2 checkpoints without Phase 3P runtime metadata remain accepted. Runtime controls are separate CLI/API options, not additions to the scientific RunConfig/contract. The existing strict `num_workers` equality check on resume is preserved; interrupted Run 2 stays at workers 0. Benchmark workers 2/4 for evidence and future runs, rather than silently changing that invariant.

## Inspect without loading plants

```bash
python scripts/inspect_pointnet2_checkpoint.py \
  --checkpoint "<ARTIFACT_DIR>/pointnet2_run2_aug/last.pt" \
  --recent 10 \
  --output "<ARTIFACT_DIR>/pointnet2_run2_aug/checkpoint_inspection.json"
```

This loads tensors on CPU using the restricted weights-only loader. It does not import Dataset/model modules, need a manifest/root, start CUDA training, or alter the checkpoint. Loading a full checkpoint still needs CPU RAM for its optimizer/history.

The report includes completed/next/configured epochs; best score and its loss; best epoch recovered from history using the existing mIoU/loss tie-break; epochs since improvement; latest metrics; recent validation trajectory; augmentation, manifest hash, model/optimizer/LR/batch settings; recorded environment/runtime; contract hash validity; and recovery-state presence. Missing/incomplete history produces nulls instead of guessed epochs. `best_validation_loss` is loss at selected mIoU, not minimum historical loss. `checkpoint_weights_epoch` distinguishes current `last.pt` weights from the best historical epoch. Tiny-overfit mode is labeled and never presented as a validation result.

## Short torch-profiler workload

From the repository root, after pulling the Phase 3P branch/commit:

```bash
python -m pip install -e '.[dev,geometry,analysis]'
python -m pytest -q
```

For the commands below, put the variable block and command in the same Colab `%%bash` cell (shell variables do not persist across cells):

```bash
MANIFEST="<MANIFEST>"
PLY_ROOT="<DATA_ROOT>"
RUN2="<ARTIFACT_DIR>/pointnet2_run2_aug"

python scripts/profile_pointnet2.py \
  --config configs/pointnet2_run2_aug.yaml \
  --manifest "$MANIFEST" --root "$PLY_ROOT" --checkpoint "$RUN2/last.pt" \
  --split train --device cuda --num-workers 0 \
  --warmup-batches 1 --profile-batches 2 \
  --telemetry --output-dir /content/phase3p/torch_train
```

Change `--split val` and use a new output directory to profile validation. Train profiling runs forward/backward/Adam on a **cloned disposable model**, including cloned BatchNorm state and saved optimizer state when available. Nothing is persisted into Run 2. Without a checkpoint, fresh seeded weights/config are used. With a checkpoint, manifest/scientific configuration must match; loader worker settings may vary only in this disposable workload. Train geometry uses the checkpoint's next zero-based epoch. The benchmark is not an exact replay of the next shuffled minibatches or a resumed training result. The low-level profile_model API consumes its provided loader; use a separate disposable loader, never a live training iterator.

Warmup occurs before trace capture. Total model batches are exactly warmup + measured counts, with no automatic cycling; counts exceeding split length fail. Workers may prefetch additional *train/val* data, but no extra model steps run. Shape/memory capture adds overhead; retain small counts to avoid giant traces from Python FPS. No data/preprocessing or worker-startup time before the measurement window is claimed as measured model time.

Artifacts:

- `trace.json`: Chrome/Perfetto/TensorBoard-compatible CPU/CUDA event trace; CPU-only when CUDA is not selected.
- `operators.txt`: compact self-CPU-time table.
- `summary.json`: configuration/contract, checkpoint snapshot checksum, environment, exact batch counts, metrics, memory telemetry, all operator rows by shape, and explicit operator-family sections.

Ranges identify FPS, squared distance, indexing/gather, radius query, interpolation, each SA stage with actual N/S, FP stages, H2D, forward/loss/backward, gradient checks, optimizer, and metric transfer/update. Native topk, Conv1d/2d, BatchNorm, and copy events remain visible. Operator `device_*` times represent CUDA when selected. Inclusive ranges are nested: do not add their totals or treat every `copy_` event as H2D. Use self times and trace lanes together. These are traces, not trained-model validation metrics.

## Detailed stage timing

Use a **different short profile directory** and add `--detailed-timing` to the command above. Training also accepts this explicit debug flag, but it should not be enabled for a normal long run.

`metrics.timing` contains mean, median, P95, sum/count for data wait, H2D, zero-grad/batch preparation, forward, loss/finite check, backward, gradient-finite checks, optimizer, metrics, and total batch wall time, plus samples/second. CUDA is synchronized at stage boundaries only in this opt-in mode, making device work attributable but removing overlap and perturbing throughput. Default training adds no such synchronization. Existing correctness checks that already synchronize remain unchanged.

Optional Dataset service timings split normalized-cloud preparation (PLY/clean/decode/normalize or cache verification/load), sampling, and augmentation. Main-process wait includes residual I/O/worker/collation delay. Worker per-sample service times can overlap with GPU work or one another; they cannot be added to the main-process wall time. Normal output tensors/keys are unchanged unless detailed data timing is explicitly enabled.

## NVIDIA Nsight Systems (optional)

No Nsight package is installed by this project and no Nsight Compute integration is added.

```bash
which nsys || true
python scripts/nsys_pointnet2.py
```

The helper reports version/path when available or exits cleanly with an unavailable status and torch-profiler fallback. To run a tiny workload on a runtime where `nsys` exists:

```bash
MANIFEST="<MANIFEST>"
PLY_ROOT="<DATA_ROOT>"
RUN2="<ARTIFACT_DIR>/pointnet2_run2_aug"

python scripts/nsys_pointnet2.py --run --report /content/phase3p/nsys_run2 -- \
  --config configs/pointnet2_run2_aug.yaml \
  --manifest "$MANIFEST" --root "$PLY_ROOT" --checkpoint "$RUN2/last.pt" \
  --split train --device cuda --num-workers 0 \
  --output-dir /content/phase3p/nsys_workload
```

The helper fixes warmup=1 and measured batches=2, uses CUDA/NVTX/OS-runtime tracing, and writes `.nsys-rep` (or the installed version's equivalent) at the report prefix. It invokes `--nvtx-only`, disabling torch's profiler to avoid competing CUPTI activity subscribers. Thus the workload directory has summary/text metadata but no torch `trace.json`; inspect the external Nsight report. Missing installation is nonfatal; an installed but unsupported/permitted profiler reports its own failure, and torch profiling remains the fallback. This optional Nsight path has not been verified on the recorded T4 runtime.

## DataLoader and transfer matrix

```bash
MANIFEST="<MANIFEST>"
PLY_ROOT="<DATA_ROOT>"

python scripts/benchmark_pointnet2_data.py \
  --config configs/pointnet2_run2_aug.yaml \
  --manifest "$MANIFEST" --root "$PLY_ROOT" --split train \
  --workers 0 2 4 --pin-memory both --non-blocking \
  --warmup-batches 1 --batches 5 --device cuda \
  --output-dir /content/phase3p/loader_matrix
```

`benchmark.json` reports requested/effective options, measured actual pinned tensors, data-wait/transfer distributions, Dataset service times, throughput, and unavailable/failed configurations individually. No model training occurs and no winner is automatically applied. A worker timeout prevents indefinite waits. Small samples and startup/prefetch effects need longer follow-up measurements before decisions.

Training/profile CLIs accept `--pin-memory`, `--prefetch-factor N`, `--persistent-workers`, `--non-blocking`, `--cache-dir`, `--detailed-timing`, and `--telemetry`. Defaults preserve the previous loader: no pinning, PyTorch's default prefetch for workers, recreated workers, blocking `.to`, no cache/debug timing/telemetry. Prefetch/persistence require workers >0; benchmark matrices leave those settings inactive for workers 0. Pinned memory without a CUDA accelerator is reported as ineffective, not a successful optimization.

Persistent workers use a shared epoch value so workers receive Dataset epoch updates. A loader shim consumes the same per-iterator base-seed generator draw as recreated workers, preserving RandomSampler order and boundary RNG state. Per-sample augmentation remains independent of worker RNG. Tests include a real spawn/persistence parity check; some local environments cannot execute that check because Unix sockets are unavailable. Confirm it in CI/Colab before opting into persistence. Never use persistence to bypass the existing Run 2 worker-count resume check.

## Optional normalized-cloud cache

The local disk boundary is exactly `PLY → finite clean → RGB semantic decode → centroid/unit-sphere normalize`. Every access still performs the original deterministic sampling and then training augmentation. Validation outputs are equivalent; no sampled points, epoch choices, or augmented arrays are cached. Cache support is restricted to train/val MaizeField3D without SOR; this avoids silently reusing unsupported generic-label or library-dependent SOR behavior.

Identity includes source resolved path and **full byte SHA256**, preprocessing implementation/boundary hashes, cache schema version, NumPy/PLY-reader versions, and normalization/annotation policy. Full source bytes are checked on every access, even if size/mtime are unchanged. Payload arrays/metadata have their own checksum; NPZ loads prohibit pickle. Cache schema v2 stores uncompressed NPZ arrays, avoiding compression in the hot path at the cost of local disk space. Its identity also includes canonical manifest contents plus exact sample ID, path, and split. Schema v1 entries are left untouched and not reused. Writes are atomic, partial/corrupt entries fail with a rebuild instruction, and source edits during population fail. Cached RGB/labels/source indices retain original dtypes and alignment.

This is a conservative prototype, not a guaranteed speedup: checksum reads and NPZ loading/checksum work can outweigh preprocessing savings, especially for small or fast binary PLYs. It does not eliminate all source I/O. Copy PLYs to local Colab storage first; keep cache on `/content`, not Drive. Local cache is disposable after runtime loss. Different source/pipeline/library provenance creates new entries; old entries can be removed manually when not in use.

```bash
MANIFEST="<MANIFEST>"
PLY_ROOT="<DATA_ROOT>"

python scripts/benchmark_pointnet2_data.py \
  --config configs/pointnet2_run2_aug.yaml \
  --manifest "$MANIFEST" --root "$PLY_ROOT" --split train \
  --workers 0 --pin-memory off --device cpu \
  --warmup-batches 0 --batches 5 \
  --cache-dir /content/agri3d_normalized_cache --compare-cache \
  --output-dir /content/phase3p/cache_comparison
```

The comparison runs uncached, population, and warm modes in order, using isolated fresh subdirectories. A nonempty comparison cache root is rejected; existing caches can still be benchmarked without --compare-cache. OS page-cache state and fixed ordering confound this short experiment. Confirm repeated/local-real-data results before enabling `--cache-dir` explicitly. Tests show exact cached/uncached tensors across training epochs and validation, plus matching model/loader state when an old checkpoint resumes with cache/timing infrastructure enabled.

## GPU memory and bottleneck interpretation

Optional telemetry records GPU name, allocated/reserved bytes, and peak allocated/reserved bytes. CPU-only reports remain valid with null CUDA memory values. Profiling resets CUDA peak counters after warmup; training telemetry uses process-lifetime peaks. No dependency on `nvidia-smi`, NVML, or fragile utilization polling is introduced. Utilization percentage is not measured here; use Nsight GPU lanes for occupancy/idle-time investigation.

Read results in this order:

1. Large main-process data wait: compare local PLYs, worker counts, and cache; inspect Dataset service times before concluding CPU preprocessing dominates.
2. Large H2D or GPU idle gaps around copies: compare actual pinning/nonblocking transfer measurements; debug synchronization removes overlap, so validate with an unsynchronized trace too.
3. Dense sequences of small FPS kernels/Python launches or gaps around indexing/finite checks: investigate trace CPU/GPU coordination. Existing bounds checks use Python tensor booleans that can synchronize CUDA; Phase 3P records but does not remove them.
4. Large Conv/BatchNorm or interpolation time: use shapes, self device times, and allocated-memory peaks. Do not substitute a new kernel or precision mode during Run 2.
5. Large metrics/gradient-check time: account for D2H/finite-check synchronization separately; these correctness checks are retained.

The first SA stage has N=10,000 and S=1,024. Pure PyTorch FPS costs O(B·N·S), keeps O(B·N) distance/selection state, and iterates S times in Python, issuing repeated reductions/updates. Radius-query distance work also scales O(B·N·S), with additional backend-dependent topk work. Chunk size 128 bounds each B=2 float32 distance tensor to about 9.77 MiB, not total VRAM. Final 3NN propagation searches dense-to-sparse points in chunks; gathered features/activations add memory. No original-resolution N×N tensor or compiled third-party ops is introduced.

## Observed CPU findings (not Colab GPU claims)

The checked-in [raw microbenchmark](benchmarks/phase3p_geometry_cpu.json) used Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.3.5, one CPU thread, B=2, synthetic random XYZ, N=10,000/S=1,024, one warmup, three repetitions, neighborhood K=32/chunk=128. Interpolation used 64 feature channels. These are isolated operations, not a whole SA forward/backward or real plant throughput.

| Operation | Median ms | P95 ms |
|---|---:|---:|
| FPS | 255.12 | 279.52 |
| Radius query, all 1,024 centers | 146.85 | 152.74 |
| Interpolation to 10,000 input points | 92.32 | 94.49 |
| Squared distance, one 128-center chunk | 11.53 | 12.00 |
| topk, one 128-center chunk | 7.37 | 7.63 |
| Centroid gather only | 0.04 | 0.04 |

FPS was the largest isolated geometry cost in this CPU measurement. Chunk-sized rows cannot be summed with whole-query rows; radius query already includes distance/topk. Three repetitions give a weak P95 estimate. Synthetic random clouds are not normalized maize geometry. No GPU/Colab bottleneck dominance or speedup has been established.

Reproduce on Colab CUDA, without reading a dataset:

```bash
python scripts/benchmark_pointnet2_geometry.py \
  --device cuda --points 10000 --centroids 1024 --batch-size 2 \
  --warmup 1 --repeats 3 --output /content/phase3p/geometry_cuda.json
```

For the CPU reference command, prefix `OMP_NUM_THREADS=1` and use `--device cpu`. A reduced synthetic RGB-PLY CPU CLI smoke also exported a torch trace, per-stage timing, operator summary, and loader/cache matrix successfully. These fixtures establish tooling execution, not learning performance. Nsight was unavailable and reported that status gracefully.

## Future options remain disabled

`configs/pointnet2_future_options.example.yaml` is a **documentation-only placeholder, not a training RunConfig**. It lists larger-batch exploration, AMP, TF32, optimized ops, and disabled early-stopping settings. No early-stopping implementation/CLI switch is added; it cannot stop the resumed 100-epoch Run 2. Enabling any of these requires a separately defined future experiment and compatibility tests, not editing the official Run 2 settings mid-run.

## Verification and changed files

Final local verification: **107 tests passed, 2 skipped**. Both skips require Unix sockets for worker tensor transfer; pin-memory-on-CPU also emits the expected no-accelerator warning. New/changed Python lint and compilation passed.

CPU tests cover dataset-free inspector behavior, history selection, bounded/read-only profiling, CUDA-free telemetry, no default timing synchronization, loader parsing, cached/uncached tensor equality, byte/pipeline invalidation and corruption, exact regular resume through new cache/timing options, epoch 70→71 arithmetic, unavailable Nsight, NVTX-only/CUPTI separation, and disabled future placeholders. Real worker parity is included but skipped where Unix sockets are prohibited. CUDA pinning/stream behavior, VRAM, Nsight collection, and real data performance require Colab measurements.

| File | Change |
|---|---|
| `src/agri3d/performance/__init__.py` | Performance package |
| `src/agri3d/performance/runtime.py` | Runtime options and epoch-aware loader |
| `src/agri3d/performance/ranges.py` | Opt-in torch/NVTX ranges |
| `src/agri3d/performance/timing.py` | Stage timing/statistics and telemetry |
| `src/agri3d/performance/inspection.py` | Dataset-free checkpoint report |
| `src/agri3d/performance/profiling.py` | Disposable bounded model profiling |
| `src/agri3d/performance/benchmark.py` | Loader/cache matrix |
| `src/agri3d/preprocessing/cache.py` | Provenance-checked normalized cache |
| `src/agri3d/training/dataset.py` | Cache boundary, shared epoch, optional data timing |
| `src/agri3d/training/engine.py` | Optional stages, nonblocking copies, bounded profiling hooks |
| `src/agri3d/training/pointnet2_runner.py` | Runtime plumbing/metadata; existing recovery preserved |
| `src/agri3d/models/pointnet2_ops.py` | Inactive-by-default function range wrappers only |
| `scripts/train_pointnet2.py` | Explicit optional infrastructure flags |
| `scripts/inspect_pointnet2_checkpoint.py` | Inspection CLI |
| `scripts/profile_pointnet2.py` | Torch profiler / NVTX-only workload CLI |
| `scripts/nsys_pointnet2.py` | Optional Nsight availability/launch helper |
| `scripts/benchmark_pointnet2_data.py` | Loader/cache benchmark CLI |
| `scripts/benchmark_pointnet2_geometry.py` | Isolated geometry benchmark CLI |
| `tests/test_performance.py` | Performance/recovery correctness tests |
| `configs/pointnet2_future_options.example.yaml` | Disabled future placeholders |
| `docs/benchmarks/phase3p_geometry_cpu.json` | Observed CPU geometry timings/provenance |
| `docs/phase3p_training_performance.md` | Recovery/profiling guide and report |
| `docs/phase3_pointnet2.md` | Phase 3P navigation |
| `README.md` | Performance tooling entry point |

No scientific Run 2 result is changed by this infrastructure work. Run 2 is complete and its validation comparison is recorded in [results](results.md). Test remains untouched.

## Phase 3P continuation — measured pipeline and compute comparisons

### Evidence boundary and current T4 recommendation

The following are **recorded real Colab observations**: Tesla T4, peak allocated ≈325 MB, peak reserved ≈359 MB, batch 2, late Run 2 train epochs ≈136–140 seconds and validation ≈26 seconds. Train throughput is therefore approximately 2.60–2.68 plants/s (364 plants), and validation throughput ≈3.00 plants/s (78 plants). Low allocated memory indicates VRAM capacity is not presently the constraint. It does not distinguish CPU data wait, Python/kernel-launch overhead, geometry kernels, or neural compute.

**These observations show that VRAM capacity is not the current constraint; they do not by themselves identify a single dominant T4 operator or establish an end-to-end speedup.** Keep official scientific configurations and current resumed-run settings unchanged until the commands below provide evidence. In particular, retain the existing `num_workers` on resume; the resume guard still rejects worker-count changes. The qualified T4 runtime controls are workers=2, pin_memory=true, non_blocking=true, prefetch_factor=2, persistent_workers=false. Treat this as one qualified execution preset; the available summary does not establish a separate speedup for every individual control. Cache remains optional.

### Profiler window and operator attribution

The profiler uses **one continuous measurement window**, with warmup completed before capture, `acc_events=True`, and **no `profiler.step()` calls**. There are no schedule cycles that can clear events. Tests assert that forward/H2D counts equal every requested measured batch and that four FPS/SA ranges occur per forward. The final summary and trace therefore cover the complete requested window. An allocator warning about allocations predating capture is separate and expected after warmup; memory deltas in operator tables are not full-model peak VRAM.

`summary.json → operators` now includes:

- Every operator grouped by input shape: calls; total/mean CPU and CUDA microseconds; self CPU/CUDA time; total and self CPU/device memory where PyTorch exposes them. CUDA values are zero in CPU profiles.
- `family_totals` and underlying `families` for FPS, square distance, matrix multiply, topk, gather, neighborhood query, interpolation, Conv1d/2d, BatchNorm, H2D, and copies.
- `set_abstraction`: explicit stage, input N, centroids, mean/total runtime, and share of forward CPU/CUDA time.
- `stage_fps`: the FPS portion of each SA stage, including its share of total forward time. The official first stage is **10,000 → 1,024**. FPS remains the existing deterministic pure-PyTorch loop with **O(B × N × npoint)** work and O(B × N) working memory.
- `forward_components`: FPS, neighborhood query, interpolation, and SA/FP shared MLP ranges compared with forward. The classifier/pooling/other work is residual. These are forward-only breakdowns; full backward compute appears in backward.

Existing logical ranges (`agri3d::fps`, `agri3d::radius_query`, `agri3d::interpolation`, `agri3d::h2d`, `agri3d::forward`, `agri3d::backward`, `agri3d::optimizer`) are retained. Disposable model wrappers add stage dimensions, stage-specific FPS/query, and MLP ranges. Normal tensor calculations and model modules are unchanged. Inclusive families can contain nested native operators: **do not sum families**, especially copies, Conv aliases, and BatchNorm aliases. CPU time includes launch/dispatch; CUDA time covers device kernels. Neither is automatically GPU utilization or critical-path time.

Run these in the same Colab `%%bash` cell after pulling `phase3p-throughput-optimization`:

```bash
MANIFEST="<MANIFEST>"
PLY_ROOT="<DATA_ROOT>"
RUN2="<ARTIFACT_DIR>/pointnet2_run2_aug"

python scripts/profile_pointnet2.py \
  --config configs/pointnet2_run2_aug.yaml \
  --manifest "$MANIFEST" --root "$PLY_ROOT" --checkpoint "$RUN2/best.pt" \
  --split train --device cuda --num-workers 0 \
  --warmup-batches 1 --profile-batches 2 \
  --telemetry --output-dir /content/phase3p_v2/operator_trace

python scripts/profile_pointnet2.py \
  --config configs/pointnet2_run2_aug.yaml \
  --manifest "$MANIFEST" --root "$PLY_ROOT" --checkpoint "$RUN2/best.pt" \
  --split train --device cuda --num-workers 0 \
  --warmup-batches 1 --profile-batches 2 \
  --detailed-timing --telemetry \
  --output-dir /content/phase3p_v2/stage_timing
```

The diagnostic `metrics.timing` reports percentage of total batch wall time for data wait, H2D, forward, backward, optimizer, metrics, and the separate loss/gradient-check/preparation stages, plus unattributed overhead and `largest_measured_stage`. Per-sample CPU service timings may overlap and are excluded from this percentage sum. Stage timing synchronizes CUDA and changes overlap; use it for attribution, **not** as the primary speedup measurement. Keep traced windows short because pure-PyTorch FPS generates many events.

### Full DataLoader/transfer matrix

```bash
MANIFEST="<MANIFEST>"
PLY_ROOT="<DATA_ROOT>"

python scripts/benchmark_pointnet2_data.py \
  --config configs/pointnet2_run2_aug.yaml \
  --manifest "$MANIFEST" --root "$PLY_ROOT" --split train \
  --workers 0 2 4 --pin-memory both --matrix --prefetch-factor 2 \
  --warmup-batches 1 --batches 5 --device cuda \
  --output-dir /content/phase3p_v2/loader_matrix
```

`--matrix` compares nonblocking off/on for each pinning setting and persistence off/on when workers >0. Nonblocking with pageable memory is allowed by PyTorch but is not labeled as an effective pinned asynchronous path. Each row records requested controls, actual pinning, two bounded traversals of the same prefix/epoch, median/P95 data wait, median H2D, median total batch time, samples/s, and iterator startup time. The second pass exposes worker reuse. Prefetch may read additional train/val plants; it never reads test. Failures/timeouts are recorded individually.

After each timed pass, the benchmark compares SHA256 fingerprints of exact sample identities, XYZ, labels, masks, source indices, centroid, and scale against the common reference. Hashing is outside the timed window. Timed host batches are retained until this check; this is a bounded diagnostic memory cost. Comparison is at fixed seed/epoch with augmentation still live. These are **loader+transfer** rates, without model compute; they cannot be used directly as training epoch estimates.

### Fresh cache population and warm reads

```bash
MANIFEST="<MANIFEST>"
PLY_ROOT="<DATA_ROOT>"

python scripts/benchmark_pointnet2_data.py \
  --config configs/pointnet2_run2_aug.yaml \
  --manifest "$MANIFEST" --root "$PLY_ROOT" --split train \
  --workers 0 --pin-memory off --warmup-batches 0 --batches 5 \
  --device cpu --cache-dir /content/agri3d_cache_v2_comparison --compare-cache \
  --output-dir /content/phase3p_v2/cache_comparison
```

Use a fresh/empty cache directory. The command refuses to reuse a populated comparison root, creates an isolated cache subdirectory per matrix configuration, and compares:

1. `uncached`: application cache disabled; first and repeated source traversals reported.
2. `cache_population`: fresh misses and writes, with first-pass timing selected.
3. `cache_warm`: verified hits, with warm-pass timing selected.

Both passes and measured hit/miss counts remain in the JSON; workers may populate prefetched entries outside the measured window. This is application-cache cold/warm measurement. OS file caches are uncontrolled and no privileged OS cache flush is attempted. `plants_per_second`, data-wait distributions, normalized-cloud/sampling/augmentation service times, and exact-output fingerprints are reported. Add `--workers 0 2 4 --pin-memory both --matrix --device cuda` with **new directories** to compare cache+workers.

Cache identity includes schema/storage version, source bytes SHA256 and resolved path, canonical manifest hash, sample ID/path/split, reader/preprocessing and boundary code fingerprints, annotation/normalization contract, and relevant package versions. Any relevant source/identity/preprocessing change produces a new entry. Corrupt payloads fail explicitly; schema v1 files are retained but not reused. Full source-byte verification still occurs on each access, so local PLY storage remains important. Uncompressed v2 NPZ trades extra local disk for avoiding compression. Sampling and augmentation remain live every epoch; no augmented/sampled output is cached. Tests compare normalized clouds **before sampling/augmentation**, final sampled/augmented tensors across epochs, invalidation, and validation metrics.

### Batch-size throughput benchmark — separate from experiments

```bash
MANIFEST="<MANIFEST>"
PLY_ROOT="<DATA_ROOT>"
RUN2="<ARTIFACT_DIR>/pointnet2_run2_aug"

python scripts/benchmark_pointnet2_throughput.py \
  --config configs/pointnet2_run2_aug.yaml \
  --manifest "$MANIFEST" --root "$PLY_ROOT" --checkpoint "$RUN2/best.pt" \
  --batch-sizes 2 4 8 --warmup-batches 1 --batches 3 \
  --num-workers 0 --device cuda \
  --output-dir /content/phase3p_v2/batch_throughput
```

Optionally add 16 to `--batch-sizes`. Candidate failures/OOMs are recorded and CUDA allocator cache is released between candidates. Each candidate uses full bounded batches from train only, cloned initial checkpoint weights and Adam state, and the same seeded data contract. A second independently initialized pass collects synchronized stage timing. No checkpoint is saved, scheduler is never stepped, and the source checkpoint is never modified. All RNGs are restored at exit.

`throughput.json` preserves the original **scientific contract** (batch 2) and records each candidate's benchmark-only batch size separately. Primary `throughput` reports samples/s, batches/s, elapsed time, and peak allocated/reserved VRAM **without torch profiler or per-stage synchronization**. The `diagnostic` pass reports forward/backward timing and stage shares. The rough epoch estimate is train plants / primary throughput, excluding validation/checkpoint overhead. Different batch sizes change optimizer update grouping and BatchNorm statistics, so they require a **new scientific experiment** even if faster. Official Run 1/2/3 batch size remains 2.

To measure end-to-end gains from a candidate runtime, repeat this command with `--batch-sizes 2` and the explicit candidate flags (`--pin-memory --non-blocking --cache-dir ...`, and worker/persistence options if independently qualified). Use new output directories and multiple independent repetitions. Do not infer a winner from a loader-only benchmark or from one short noisy run.

### Qualification and reporting table

A separate bounded Tesla T4 throughput benchmark reported **2.58 samples/s at batch 2, 3.58 at batch 4, and 4.28 at batch 8**. The batch-2 full-run epoch observation above is independent. Throughput-only pass estimates are 364 divided by measured samples/s; they are not measured full-epoch durations. Larger batches improve benchmark throughput but change optimizer grouping and BatchNorm statistics. Official experiment batch size remains 2.

| Configuration | batch | samples/s | train epoch est. | peak allocated / reserved VRAM | data wait | forward | backward |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (recorded Colab epoch) | 2 | 2.60–2.68 | 136–140 s observed | ≈325 / 359 MB | not isolated | not isolated | not isolated |
| qualified runtime preset (workers=2, pinned, nonblocking, prefetch=2) | 2 | not separately reported | not separately reported | not separately reported | not isolated | not isolated | not isolated |
| workers=4 | 2 | not reported | not reported | not reported | not reported | not reported | not reported |
| pin_memory + nonblocking | 2 | qualified in preset above; not isolated | — | — | — | — | — |
| cache | 2 | not reported on T4 | not reported | not reported | not reported | not reported | not reported |
| cache+workers | 2 | not reported on T4 | not reported | not reported | not reported | not reported | not reported |
| batch2 throughput benchmark | 2 | 2.58 | ≈141 s estimate, not observed | not reported | not reported | not reported | not reported |
| batch4 benchmark | 4 | 3.58 | ≈102 s estimate, not observed | not reported | not reported | not reported | not reported |
| batch8 benchmark | 8 | 4.28 | ≈85 s estimate, not observed | not reported | not reported | not reported | not reported |

Fill samples/s and VRAM from the primary batch-throughput pass, and stage timings from its separate diagnostic pass. Include units and repeat variability. Compare like-for-like checkpoints, software, hardware, warmed state, augmentation, and selected train prefix. A short-window epoch estimate is not an observed epoch duration. Validation ≈26 s is separate from the train estimate.

Pinned memory, nonblocking copies, local cache, and loader controls are runtime-only candidates for future scientific runs **after equality and throughput qualification**. CPU tests verify exact model updates, validation metrics, final sampled/augmented tensors, and resume states for supported paths; a CUDA test checks true pinned/nonblocking equivalence when an accelerator exists. Persistent-worker parity and CUDA equality are skipped when the environment cannot support them. Run `python -m pytest -q` on Colab as well before adopting these paths. No claim of bitwise GPU or worker equivalence is made from skipped tests. Existing strict worker-count resume checks stay in force.

AMP, TF32, custom kernels, nondeterminism, and official batch-size changes are not enabled. No Run 1/2 artifacts are modified or deleted, and Run 3 scheduling/scientific contracts are unchanged.

### Measured synthetic CPU findings

[Compact CPU report](benchmarks/phase3p_pipeline_cpu.json): one CPU thread, generated binary 100k-point clouds, 10k-point live sampling, workers 0, batch 2, three measured loader batches per pass. Application-cache-disabled first-pass loader+transfer throughput was **8.05 plants/s** (repeated pass **7.06**), population **6.95**, and warm reads **10.54**. Median data wait was approximately **252 ms**, **285 ms**, and **190 ms** for the selected first/population/warm passes, respectively. These are short synthetic CPU measurements, not T4 gains, not an uncompressed-vs-compressed comparison, and not a runtime recommendation.

A separate two-batch reduced 256-point model profile on those clouds measured **data wait ≈82.8%** of synchronized batch time, forward ≈9.1%, backward ≈4.3%. Within forward, FPS was ≈20.1%, neighborhood ≈6.0%, interpolation ≈30.8%, and SA/FP MLP ≈28.1% of inclusive CPU time. The **highest-impact measured stage in this synthetic CPU workload is data wait**. Its reduced geometry makes it unsuitable for extrapolating the 10k-point T4 compute balance. The earlier full-size isolated CPU geometry benchmark still measures FPS at ≈255 ms for B=2, N=10,000, S=1,024, but neither CPU result identifies the T4 critical path. The recorded T4 observations do not isolate a dominant stage; use the bounded T4 profile for CPU/CUDA attribution before making an operator-level bottleneck claim.
