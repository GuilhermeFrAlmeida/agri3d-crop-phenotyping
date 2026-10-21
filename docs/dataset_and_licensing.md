# Dataset and licensing

## Repository source code

The repository includes the standard Apache License 2.0 text in [`LICENSE`](../LICENSE). This applies to the source-code repository subject to its terms. It does not grant rights to external datasets.

The existing license file contains the canonical Apache-2.0 terms but leaves the appendix's example copyright notice uncustomized. No ownership notice is added here. A maintainer may review whether a verified copyright notice should be added before release.

## MaizeField3D dataset

The project uses the segmented 100k MaizeField3D subset, obtained separately from the [official Hugging Face dataset](https://huggingface.co/datasets/BGLab/MaizeField3D). The dataset card identifies its terms as **CC BY-NC 4.0**. This is a non-commercial Creative Commons license and is separate from the repository's Apache-2.0 license. Users should review the upstream license and attribution requirements before use, redistribution or commercial activity.

The dataset is not bundled in this repository. Do not commit PLY files, dataset archives, metadata workbooks, split manifests tied to private storage, or generated dataset copies. Obtain the data from the original distribution and pin a dataset revision for reproducible experiments.

## Dataset citation

The upstream dataset card and accompanying paper provide this citation metadata:

> Kimara, E., Hadadi, M., Godbersen, J., Balu, A., Jubery, Z., Krishnamurthy, A., Schnable, P., & Ganapathysubramanian, B. “MaizeField3D: A curated 3D point cloud and procedural model dataset of field-grown maize from a diversity panel.” *Plant Phenomics* 8(1), 100108 (2026). https://doi.org/10.1016/j.plaphe.2025.100108

The repository does not claim authorship of MaizeField3D. The upstream card also provides a citation block; consult that source if citation details change.

## Repository citation

[`CITATION.cff`](../CITATION.cff) contains only verified repository metadata and no DOI, institutional affiliation or publication claim.
