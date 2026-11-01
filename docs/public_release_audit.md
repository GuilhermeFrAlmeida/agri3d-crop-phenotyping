# Public-release audit

> **Historical audit snapshot (2026-10-21):** experiment-state statements below describe that release review. Current Run 4 completion and Phases 4/5 development are tracked in [results](results.md) and [the pipeline guide](phases4_5_phenotyping_pipeline.md).

**Audit basis:** latest `main` at `53af6848a460d60f24c91f07416b872140b1ac1c` (2026-10-21), GitHub API recursive tree and commit listing, all 19 commits in the main ancestry, current tracked-file inventory, source/docs pattern scans, Apache license file, and upstream MaizeField3D dataset card. Repository visibility remains private. This review does not rewrite history or change visibility.

| Check | Status | Evidence / Notes | Action Required |
|---|---|---|---|
| Secrets / credentials | PASS | Scanned current source/docs/notebook and commit diffs for common API-key/token/private-key/password patterns. No credential-like value found. | Re-run a dedicated secret scanner before changing visibility; rotate any credential discovered then. |
| Personal paths / identifiers | FAIL | Current docs previously contained generic mounted-Drive paths. Commit author metadata across all 19 commits contains a personal Gmail address and author name. No home-directory path, account ID, or credential was found in tracked file content. | Decide whether to publish the existing author email. If it must be removed, rewrite affected Git history across branches before publicizing. Current docs use generic path placeholders. |
| Corporate/proprietary references | PASS | No internal company, employer, proprietary system, or non-public infrastructure references found. Dataset and public project references are intentional attribution. | None identified. |
| Large committed datasets | PASS | Current recursive tree is approximately 374 KB at repository level; no PLY, ZIP dataset archive, spreadsheet, or other dataset payload is tracked. | Keep external data outside the repository. |
| Generated model artifacts | PASS | No `.pt`, `.pth`, checkpoint, prediction archive, or experiment output is tracked in the current tree. | Keep run artifacts and private manifests outside Git. |
| Dataset redistribution risk | PASS | The segmented subset is external and absent from the tree. The upstream Hugging Face dataset card identifies CC BY-NC 4.0. The new docs distinguish dataset terms from the code license. | Users must obtain the dataset upstream and follow its current license/attribution. Do not redistribute it from this repository. |
| License consistency | PASS | GitHub `main` contains the standard Apache License 2.0 text; README and licensing guide identify Apache-2.0 for code and CC BY-NC 4.0 for the dataset. The appendix copyright example is uncustomized. | A maintainer should review whether to add a verified copyright notice. Do not infer ownership from commit metadata. |
| Stale experiment claims | FAIL | Tracked README/current guides now reflect Run 3 completed at epoch 99, Run 1 champion, Run 4 pending, and Phase 3P T4 measurements. The current GitHub repository description still describes individual-plant reconstruction and trait outputs as if implemented. | Update the repository description before public release to distinguish current semantic segmentation from planned reconstruction/trait work. |
| Broken documentation links | PASS | All local Markdown links and referenced config/script paths were checked after edits; no broken local references remain. | Re-run link checks when adding/removing docs. External links depend on upstream availability. |
| Reproducibility instructions | PASS | Public setup, manifest pinning, gate, train, validation, analysis, resume and runtime/scientific separation are documented with generic local-path placeholders. | Dataset revision and frozen manifest remain external inputs supplied by the experiment owner. |
| Git history review | FAIL | GitHub API exposes 19 commits in main ancestry; all carry the same author identity and personal Gmail metadata. Historical documentation also includes Colab mounted-Drive paths in phase guides. Current tree has no secret or dataset artifact. | Review the affected commit range below. Do not make the repository public until personal metadata is consciously accepted or history is cleaned. |

## History findings and scope

The `main` ancestry examined contains 19 commits, from initial commit `64c0357eeac671c0b7e4b16797851075b0664f19` through merge commit `53af6848a460d60f24c91f07416b872140b1ac1c`. Every commit's author metadata exposes the account's personal name and a Gmail address. The address is intentionally not repeated in this public-facing audit file. GitHub API commit objects expose it directly.

Old versioned phase guides also included mounted Google Drive path examples (without an account ID or token). Current documentation replaces these with placeholders. The paths occurred in phase documentation during the Run 1–4 and Phase 3P commits, including `docs/phase3_pointnet2.md`, `docs/phase3a_validation_analysis.md`, `docs/phase3b_run2_augmentation.md`, `docs/phase3c_run3_scheduler.md`, `docs/phase3d_run4_normals.md`, and `docs/phase3p_training_performance.md`. No dataset or checkpoint bytes were found in those commits.

For clarity, the 20 main-ancestry commits reviewed were:

```text
53af6848a460d60f24c91f07416b872140b1ac1c
5ad4f94934bc325d2f595d3ec8356acd9f7a399b
2c6f75cbd3e1b447afcb8bd5181224f48259539f
6516934ded1cc6d509c0880b13efffa6ab6e47fe
241411e5340cd9b362926962ad9a357a2c442dc3
c6715a1ca30117e02cae5bbc938bf9bd8cd5fb73
28d97d0cb17d924cf7ba45e0597139afd8f3b224
a57793997d76dc29038b09a3a8fbd9470158a3c6
9edaad5d894ff050bafc8835286cfd60f93087d6
187075a20acaf980bbae259f0f84ca684643337b
8c79522960bedcfb18fe53d07c7ef4fa8f9ce269
05a7b49774f978d7e5694d5eb8aeb82c2c9c3790
beeccbe387d57e254f17cbbb8fadb05406bc5c65
f71bc784c23bc08caafb3757b9b250a124558290
6b115080891205769e6533d202e7a5631709e72a
10bb1ae61d5653841f8146b3bb389230781ab4cb
779b5e8348e7e2603823cbd748dd90a95fcc1a73
f954278d35e683070dcc1b18bcf2d4e666665abd
64c0357eeac671c0b7e4b16797851075b0664f19
```

The 19 commits were enumerated from the GitHub commit API. Branches corresponding to earlier phase work point at these same reviewed changes; no tags were present. The repository still has a private visibility setting. No visibility change, release, tag push, or history rewrite was performed.

## Recommended history-cleanup procedure

If the author email or historical Drive paths should not become public, an authorized maintainer should:

1. Decide on an approved replacement identity/email (for example, a GitHub noreply address) and preserve attribution as intended.
2. Clone the repository with all branches and tags, create an offline backup, and use `git filter-repo` with an explicit mailmap/callback to rewrite commit author/committer metadata and, if desired, historical documentation paths.
3. Inspect rewritten commits, trees, and diffs; run secret and artifact scans again.
4. Coordinate the force push of all affected branches and any pull-request refs with collaborators. Existing clones/forks may retain the old history.
5. Check GitHub cached views and discuss cache removal with GitHub Support if the old metadata remains accessible.

Do not rewrite history solely as part of documentation review. If any actual credential is later discovered, revoke/rotate it first, then follow the credential-provider's incident procedure in addition to history cleanup.

## License and citation evidence

- The repository's `LICENSE` blob at `main` is the standard Apache License 2.0 text; it is not modified in this documentation branch.
- The [official MaizeField3D Hugging Face dataset card](https://huggingface.co/datasets/BGLab/MaizeField3D) states `CC-BY-NC-4.0` and describes the 520 segmented 100k PLY subset.
- The associated article is [Kimara et al., Plant Phenomics 8(1), 100108 (2026)](https://doi.org/10.1016/j.plaphe.2025.100108). Repository citation metadata contains no DOI or publication claim.
