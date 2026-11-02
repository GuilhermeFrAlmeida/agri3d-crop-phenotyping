# Public-release audit

## Scope and decision

**Review date: 2026-11-02. Basis:** `main` at `3a50d684c944682b5f5c80054f3c034315824296` plus `chore/public-portfolio-hardening`. This audit is a scoped engineering review, not a guarantee that pattern scanning finds every possible secret.

**Release-candidate decision: READY WITH MANUAL ACTIONS, provided the branch's required CI check is green.** Repository visibility was verified as private at review time and is not changed here. No release, package, tag, model hosting, history rewrite or remote branch deletion is performed.

## Safety and presentation checks

| Check | Status | Evidence / notes | Action required |
|---|---|---|---|
| Credentials / secrets | PASS, scoped | Current text and all available patches in the 27 commits reachable from main scanned for credential/private-key patterns; no matching credential values found | Re-scan if additional refs/content are introduced |
| Private paths / infrastructure | PASS, current tree | Public guides/notebooks use placeholders, not account/Drive/host paths; no private infrastructure endpoint found | Sanitize future generated output provenance before sharing |
| Personal identifiers | MANUAL ACTION | Intended public author/repository attribution remains; personal author email metadata exists in Git ancestry | Decide whether existing history exposure is acceptable; do not assume branch deletion removes it |
| Corporate / proprietary references | PASS, scoped | No corporate source/data or proprietary infrastructure reference identified in reviewed project content | Review future imported files |
| Raw datasets / private manifests | PASS | No raw MaizeField3D PLY/archive or research manifest committed; data stays external | Preserve ignore rules and external acquisition |
| Models / experiment bundles | PASS | No `.pt`, `.pth`, `.onnx` or raw val78 bundle committed; existing synthetic benchmark JSON is explicitly labeled | Keep scientific artifacts external |
| Large files | PASS | Only newly approved canonical PDF is about 1.34 MB; no new dataset/model binary | Review future binary additions |
| Notebooks | PASS | Three curated wrappers, no outputs/attachments, synthetic execution checks and explicit external inputs | Never commit run outputs or checkpoint/data payloads |
| Report metadata / contents | PASS, scoped | Supplied 23-page PDF retained byte-for-byte; public professional author attribution, no detected private paths/credentials, no PDF JavaScript or embedded dataset | Maintainer decides explicit report/figure reuse terms |
| Dataset redistribution | PASS, boundary documented | No dataset bundled; report documents results and attributed figures, not raw data | Respect external CC BY-NC 4.0; code licensing does not relicense data |
| License consistency | PASS, with ownership decision | Canonical Apache-2.0 LICENSE; package metadata includes its expression/file; appendix placeholder remains an example | Add a project-specific notice only with verified ownership; no separate report license invented |
| Metric / stale-state review | PASS | Run 1–4 values preserved; approved val78 aggregates and report corroborate 156 executions / zero failures; current docs updated | Keep historical selection and later sampled/full-cloud evaluations distinct |
| Local links / command references | PASS after validation | Referenced files/configs and current CLI interfaces checked; no removed-plan links remain | Check PR verification summary and CI before public release |
| CI / installation | REQUIRED GATE | Missing Open3D EGL runtime addressed with `libegl1`, real import smoke and unchanged geometry tests; Python 3.11 / Ubuntu 24.04 canonical | Require green branch CI; do not waive known environment/test failures |
| Git history review | MANUAL ACTION | 27 reachable commits, 472 file-change records, 458 text patches; 14 no-patch records are empty package initializers | Review email and removed planning history; unmerged refs are not exhaustively certified |

## History findings and boundaries

The main ancestry was enumerated through Git commit records, including merged histories down to `64c0357eeac671c0b7e4b16797851075b0664f19`. All nonempty changed-file records supplied text patches. Historical notebook changes were reviewed; no committed dataset/checkpoint archive was identified. No credential value is reproduced in this audit.

Personal author email metadata occurs in the reviewed ancestry, including the current base commit. The address is deliberately not repeated here. Removing it from a current document would not remove commit metadata. Similarly, the removed planning documents remain reachable in older commits. **Deleting old branches does not remove commits already merged into main.**

No real secret was discovered, so no secret-driven history rewrite is proposed as an automatic action. If a future credential is found, first revoke/rotate it, identify every affected path/commit/ref, then plan a reviewed rewrite with a tool such as `git filter-repo`, coordinate collaborators, remove affected hosted refs/caches as applicable, and verify from a fresh clone. Do not silently force-push or treat tip deletion as eradication. Any privacy-driven rewrite of email/planning history is a separate maintainer decision.

This review covers current main ancestry, not every unmerged remote branch, fork, hosting cache or future commit. The maintainer's separate branch-cleanup plan remains outside this task.

## Document classification

No unnecessary path moves were made. The documentation index supplies navigation instead.

| Classification | Documents | Rationale |
|---|---|---|
| CURRENT PUBLIC DOCUMENT | Root README; `docs/README.md`, `getting_started.md`, `architecture.md`, `dataset.md` | Public entry points, installation and actual system contracts |
| CURRENT PUBLIC DOCUMENT | `results.md`, `experiments.md`, `performance.md`, `reproducibility.md` | Current evaluation scopes, execution and measured evidence |
| CURRENT PUBLIC DOCUMENT | `limitations.md`, `roadmap.md`, `dataset_and_licensing.md`, `model_interoperability.md` | Qualification boundaries, future work and rights/contracts |
| CURRENT PUBLIC DOCUMENT | `phases4_5_phenotyping_pipeline.md`, this audit, `reports/README.md`, canonical report PDF | Current pipeline, release review and long-form presentation |
| HISTORICAL TECHNICAL EVIDENCE | `phase2_audit.md`, `phase3_pointnet2.md`, `phase3a_validation_analysis.md` | Data/training/analysis protocols, labeled historical |
| HISTORICAL TECHNICAL EVIDENCE | `phase3b_run2_augmentation.md`, `phase3c_run3_scheduler.md`, `phase3d_run4_normals.md` | Negative experiments and preserved controlled contracts |
| HISTORICAL TECHNICAL EVIDENCE | `phase3p_training_performance.md`, `phase4a_instance_diagnostics.md`, `benchmarks/*.json` | Runtime evidence and decision-time freeze; old pending wording is historical |
| REDUNDANT / INTERNAL — removed from current tree | `devlopment_plan.md`, `project_specification.md` | Internal planning/aspirational specification; relevant public directions are now in architecture/roadmap |
| REDUNDANT — replaced | `notebooks/colab_data_pipeline.ipynb` | Three-cell setup stub superseded by curated runnable wrappers and getting-started guide; no scientific outputs discarded |

The surviving phase records are scientific evidence, not installation/status entry points. Current claims should be read from the report, results and roadmap.

## Result consistency

- Historical Run 1 selection mIoU **0.836603**, epoch **76**; rejected alternatives **0.829574**, **0.831723**, **0.817336**.
- Later val78 sampled mIoU **0.83662**, projected mIoU **0.83967**; **156** successful executions, **0** failures.
- Structural means **15.94 / 17.92**, reported paired gap **+1.99**; predicted-greater/equal/lower counts **63 / 7 / 8**.
- The slight historical-versus-later sampled metric difference is retained explicitly rather than silently reconciled. No unverified cause, statistical significance or improved generalization is inferred.
- The report and approved aggregate summary are the public evidence source; raw validation bundles are intentionally external. No scientific evaluation is recomputed by this release preparation.

## Report integrity and licensing

Canonical PDF SHA256: `013b80939ae32d46ba84a0321962bbc8632427aed1cdbdd0bd8f87c40b67009f` (1,340,809 bytes). Content is unchanged from the supplied report. No separate DOCX copy or unreviewed point-cloud render is added. The canonical report may contain its approved attributed figures; it does not bundle the dataset.

The official [MaizeField3D dataset card](https://huggingface.co/datasets/BGLab/MaizeField3D) identifies CC BY-NC 4.0, separately from Apache-2.0 source code. The report index does not invent additional document licensing or ownership.

## Manual actions before changing visibility

1. Require the branch's test workflow to pass; inspect the required check rather than assuming a local run proves clean Ubuntu compatibility.
2. Decide whether personal email and removed planning content in existing Git history are acceptable. No history rewrite is performed here.
3. Decide/report the reuse terms for the technical report/figures and any verified project-specific copyright notice.
4. Review unmerged refs and perform any desired remote branch cleanup separately; deletion is not ancestry removal.
5. Optionally update repository description/topics, then explicitly change visibility when ready. No release/package publication is implied.
