# Getting started

## 1. Clone and select a revision

```bash
git clone https://github.com/GuilhermeFrAlmeida/agri3d-crop-phenotyping.git
cd agri3d-crop-phenotyping
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Record the reviewed commit for reproducible work. Canonical CI is Ubuntu 24.04 / Python 3.11; the package declares ≥3.10, but other combinations are not covered by a full matrix. CPU users may install `torch` from `https://download.pytorch.org/whl/cpu` before the project. Preserve a matching CUDA build on GPU systems.

## 2. Choose a dependency profile

Run the command for the desired profile from the repository root:

| Profile | Command | Purpose |
|---|---|---|
| Core | `python -m pip install -e .` | Package, PyTorch, NumPy/SciPy, PLY and YAML APIs |
| Geometry | `python -m pip install -e '.[geometry]'` | Open3D and scikit-learn geometry/baseline tools |
| Analysis / visualization | `python -m pip install -e '.[geometry,analysis,visualization]'` | Analysis workbooks, HTML and plotting |
| Development / tests | `python -m pip install -e '.[dev,geometry,analysis]'` | pytest, correctness lint and synthetic geometry tests |
| Interoperability | `python -m pip install -e '.[interoperability]'` | Optional ONNX export/qualification tools; backend still unqualified |
| Dataset download | `python -m pip install -e '.[data]'` | Optional Hugging Face download client |

Extras are additive. Interoperability tooling is not required for canonical PyTorch inference. Optional packages do not change frozen scientific settings.

### Ubuntu system requirement

The Open3D wheel requires EGL even when tests only import its CPU APIs:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends libegl1
python -c "import open3d; print(open3d.__version__)"
```

The observed CI error was a missing `libEGL.so.1`, not a failed segmentation assertion. No desktop session, X server or GPU is needed for synthetic CI. Other Linux distributions must supply the equivalent EGL runtime. The existing Dockerfile is a development/baseline example, not a qualified deployment image.

## 3. Inspect the project without data

Install the development profile, then:

```bash
python -m ruff check .
python -m pytest -q
```

Tests and notebook smoke tests use synthetic fixtures, not external data or model downloads. CUDA-specific tests may skip on CPU. Ruff gates fatal syntax/name and Pyflakes correctness (`E9,F`); expanding legacy formatting rules is a separate change. CI uses fresh dependency resolution, not a locked numerical environment; record exact versions for research reproduction.

Read the [technical report](reports/README.md), [results](results.md) and [architecture](architecture.md).

## 4. Obtain the external dataset

Review the separate [dataset terms](dataset_and_licensing.md). Obtain the **segmented 100k** archive from the original [MaizeField3D distribution](https://huggingface.co/datasets/BGLab/MaizeField3D). Pin a dataset commit, then optionally use:

```bash
python -m pip install -e '.[data]'
python scripts/download_ground_truth.py \
  --revision <DATASET_REVISION> --output <DATA_ROOT>
```

This downloads data; it does not create the research split. The extraction directory includes the archive's directory hierarchy. Set `<DATA_ROOT>` so every manifest-relative path resolves under it. Never substitute natural-color unsegmented files for the RGB-annotated subset.

## 5. Manifest, checkpoint and directory layout

Keep external resources separate from the source checkout:

| External resource | Required contents |
|---|---|
| `<DATA_ROOT>` | Segmented PLYs under paths referenced by the manifest |
| `<MANIFEST>` | Existing exact frozen 364/78/78 plant-identity manifest, seed 42 |
| `<CHECKPOINT>` | Compatible Run 1 selected `best.pt`, obtained externally |
| `<OUTPUT_DIR>` | New directory for generated outputs; never existing results |

The repository does **not distribute trained checkpoints or the research manifest**. No model-download URL is provided. Without these external inputs, run synthetic tests and notebook 01 on a supplied development PLY; full scientific reproduction is conditional on obtaining the exact artifacts. Do not regenerate the split or loosen contract checks to make an unrelated checkpoint load.

The canonical checkpoint backend is PyTorch `.pt`. It checks scientific contract, manifest metadata and strict model weights. Run 1 is frozen; ONNX remains **NOT QUALIFIED**. See [interoperability](model_interoperability.md).

## 6. Inspect a PLY and run inference

Install the visualization profile. To use the notebooks, optionally install JupyterLab separately with `python -m pip install jupyterlab`, then launch `jupyter lab notebooks/01_point_cloud_inspection.ipynb`. Set the input variables documented in each notebook. No dataset/checkpoint/output is embedded.

The [notebook guide](../notebooks/README.md) covers all three examples. For CLI analysis:

```bash
python -m agri3d.pipeline analyze \
  --input <PLY_PATH> --sample-id <SAMPLE_ID> \
  --checkpoint <CHECKPOINT> --config configs/phenotyping_pipeline.yaml \
  --device cpu --visualize --output <OUTPUT_DIR>
```

This calls the frozen backend, samples 10k XYZ, projects to the full cleaned metric cloud and computes structural/phenotype proxies. RGB is not a model feature. Outputs include reports, aligned scalar PLY/arrays, timings/provenance and optional HTML. Runtime provenance can contain input paths; review/sanitize outputs before redistribution.

## 7. Bounded validation or research reproduction

For authorized validation inputs and the original manifest:

```bash
python scripts/evaluate_phenotyping_pipeline.py \
  --manifest <MANIFEST> --root <DATA_ROOT> --checkpoint <CHECKPOINT> \
  --config configs/phenotyping_pipeline.yaml --split val --max-plants 5 \
  --mode both --device cpu --output-dir <OUTPUT_DIR>
```

This is an execution example, not permission to tune the frozen settings or choose a new tuning subset. The recorded Phase 4A cohort is fixed in [its report](phase4a_instance_diagnostics.md). Full val78 has already completed; it is evaluation, not tuning. Do not use the test split. Standalone PLY commands cannot infer split membership; supply development files only. Legacy baseline CLI split options are not a substitute for this policy.

For exact historical training/gate/resume protocols, see [reproducibility](reproducibility.md). No real experiment is run by installation or CI.
