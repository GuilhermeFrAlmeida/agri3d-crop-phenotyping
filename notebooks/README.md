# Public notebooks

Launch from the checkout after following [getting started](../docs/getting_started.md). For a notebook UI, install JupyterLab separately: `python -m pip install jupyterlab`. Then run `jupyter lab notebooks/01_point_cloud_inspection.ipynb`.

| Notebook | Demonstrates | External inputs |
|---|---|---|
| [01 — Point-cloud inspection](01_point_cloud_inspection.ipynb) | PLY ingestion, finite filtering, ranges, optional annotation counts and deterministic bounded display | Development PLY; no model |
| [02 — Frozen semantic inference](02_semantic_inference.ipynb) | Shared preprocessing/10k inference composition and full-cloud semantic projection inspection | PLY, compatible Run 1 selected checkpoint, sample identity |
| [03 — Full analysis](03_full_pipeline_analysis.ipynb) | Frozen components, metric phenotype proxies and checksummed artifacts/optional HTML | Same inputs plus a new output directory |

Set inputs in each notebook's first code cell. Run cells in order; no previous notebook state is required. Real data/checkpoints are not bundled and no model-download endpoint is invented. Use meters and vertical Z, retain source identity, and use development plants only. RGB is supervision/inspection only. Source notebooks have no execution outputs or attachments.

CI validates notebook structure, parses code and executes all code cells with synthetic PLYs, a synthetic semantic backend and temporary output paths. `_AGRI3D_NOTEBOOK_CONTEXT` is the explicit CI injection point; normal use leaves it unset and supplies the documented inputs. Synthetic smoke tests do not qualify trained-model accuracy. Matplotlib/HTML displays are skipped in CI; plotting dependencies belong to the optional visualization profile.

The shared analysis composition also computes downstream structures in notebook 02; no independent copy of sampling/projection code is maintained. ONNX and biological phenotype accuracy remain unqualified. See [current results](../docs/results.md).
