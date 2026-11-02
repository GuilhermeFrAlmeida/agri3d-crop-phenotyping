"""Fixed val5 geometry-only threshold experiment; never promotes production config."""
import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from agri3d.phenotyping.instances import InstanceExtractor
from agri3d.pipeline.artifacts import write_json
from agri3d.pipeline.config import InstanceConfig, PipelineConfig
from agri3d.pipeline.provenance import provenance, sha256_file

from .instance_artifacts import PLANTS, SOURCES, load_val5, verify_unchanged
from .instance_diagnostics import (
    DIAGNOSTIC_POLICY,
    attachment_context,
    diagnose,
    partition_comparison,
)

BASELINE = "r015_k16"


@dataclass(frozen=True)
class SweepConfig:
    radii_m: tuple = (.010, .015, .020, .025)
    max_neighbors: tuple = (16, 24, 32)
    min_leaf_points: int = 100
    attachment_radius_m: float = .03

    def __post_init__(self):
        if self.radii_m != (.010, .015, .020, .025) or self.max_neighbors != (16, 24, 32):
            raise ValueError("Phase 4A requires the exact ordered 4x3 grid; no extra factors")
        if (type(self.min_leaf_points) is not int or self.min_leaf_points != 100
                or self.attachment_radius_m != .03
                or any(type(k) is not int for k in self.max_neighbors)
                or any(isinstance(r, bool) for r in self.radii_m)):
            raise ValueError("Primary sweep fixes minimum size=100 and attachment radius=0.03 m")

    @classmethod
    def load(cls, path):
        with open(path) as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict) or set(data) != {"radii_m", "max_neighbors", "fixed"}:
            raise ValueError("Sweep YAML requires radii_m,max_neighbors,fixed only")
        if not isinstance(data["fixed"], dict) or set(data["fixed"]) != {"min_leaf_points", "attachment_radius_m"}:
            raise ValueError("Unexpected fixed sweep factor")
        return cls(tuple(data["radii_m"]), tuple(data["max_neighbors"]), **data["fixed"])

    def grid(self):
        return [(f"r{round(r*1000):03d}_k{k}", InstanceConfig(
            neighborhood_radius_m=r, max_neighbors=k, min_leaf_points=self.min_leaf_points,
            attachment_radius_m=self.attachment_radius_m))
            for r, k in itertools.product(self.radii_m, self.max_neighbors)]

    def to_dict(self):
        return {"radii_m": list(self.radii_m), "max_neighbors": list(self.max_neighbors),
                "fixed": {"min_leaf_points": self.min_leaf_points,
                          "attachment_radius_m": self.attachment_radius_m}}


def stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return None
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max()), "min": float(values.min())}


def write_csv(path, rows):
    import csv
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, stability):
    metrics = ["raw_component_count", "accepted_leaf_instances", "unassigned_fraction",
               "near_threshold_accepted_fraction", "heuristic_merge_risk_components",
               "extreme_size_components", "largest_to_accepted_median_ratio",
               "instance_runtime_ms", "graph_undirected_edges"]
    groups = {}
    for source in SOURCES.values():
        groups[source] = {}
        for key in dict.fromkeys(row["config_id"] for row in rows):
            selected = [r for r in rows if r["semantic_source"] == source and r["config_id"] == key]
            if selected:
                groups[source][key] = {metric: stats([r[metric] for r in selected]) for metric in metrics}
    return {"scope": "fixed val5 diagnostic experiment; component count is not verified leaf count",
            "rows": len(rows), "by_semantic_source": groups,
            "collapse_events": [r for r in stability if r["heuristic_count_collapse"]],
            "baseline_reproduced_exactly": all(r["baseline_exact_partition"] for r in rows if r["config_id"] == BASELINE),
            "timing_scope": "instance extractor wall ms, single measurement per plant/config; diagnostics/I/O/plots excluded",
            "instance_gt_status": "unverified; no instance accuracy metrics"}


def stability_analysis(records, rows, labels, sweep):
    by_key = {(r["sample_id"], r["semantic_source"], r["config_id"]): r for r in rows}
    grid = {(c.neighborhood_radius_m, c.max_neighbors): key for key, c in sweep.grid()}
    results = []
    for record in records:
        prefix = (record.sample_id, record.source)
        for r, k in itertools.product(sweep.radii_m, sweep.max_neighbors):
            current = grid[(r, k)]
            previous = []
            if r != sweep.radii_m[0]:
                previous.append(("radius", grid[(sweep.radii_m[sweep.radii_m.index(r)-1], k)]))
            if k != sweep.max_neighbors[0]:
                previous.append(("neighbors", grid[(r, sweep.max_neighbors[sweep.max_neighbors.index(k)-1])]))
            for factor, prior in previous:
                if (*prefix, current) not in by_key:
                    continue
                a, b = by_key[(*prefix, prior)], by_key[(*prefix, current)]
                count_a, count_b = a["accepted_leaf_instances"], b["accepted_leaf_instances"]
                drop = count_a - count_b
                fraction = drop / count_a if count_a else 0.
                results.append({"sample_id": record.sample_id, "semantic_source": record.source,
                    "factor": factor, "previous_config": prior, "current_config": current,
                    "previous_count": count_a, "current_count": count_b, "count_change": count_b-count_a,
                    "relative_count_drop": fraction,
                    "heuristic_count_collapse": bool(drop >= 3 and fraction >= .25),
                    **partition_comparison(labels[(*prefix, prior)], labels[(*prefix, current)], record.semantic == 0)})
    return results


def comparison_rows(rows):
    indexed = {(r["sample_id"], r["semantic_source"], r["config_id"]): r for r in rows}
    result = []
    for plant in PLANTS:
        for config_id in dict.fromkeys(r["config_id"] for r in rows):
            oracle = indexed.get((plant, SOURCES["oracle"], config_id))
            predicted = indexed.get((plant, SOURCES["predicted"], config_id))
            if oracle and predicted:
                result.append({"sample_id": plant, "config_id": config_id,
                    "oracle_components": oracle["accepted_leaf_instances"],
                    "predicted_components": predicted["accepted_leaf_instances"],
                    "delta_components": predicted["accepted_leaf_instances"]-oracle["accepted_leaf_instances"],
                    "oracle_unassigned_fraction": oracle["unassigned_fraction"],
                    "predicted_unassigned_fraction": predicted["unassigned_fraction"],
                    "oracle_runtime_ms": oracle["instance_runtime_ms"], "predicted_runtime_ms": predicted["instance_runtime_ms"]})
    return result


def recommendation(source_prov, summary, sweep, mode, review_path=None):
    grid = dict(sweep.grid())
    result = {"recommended_config": vars(grid[BASELINE]), "recommended_config_id": BASELINE,
              "decision": "KEEP CURRENT DEFAULT", "phase4_readiness": "NOT READY FOR VAL78",
              "reasoning": ["No reviewed multi-criteria decision supplied; do not promote from a scalar score."],
              "limitations": ["Component count is not verified leaf count.",
                              "Five development plants only; no test or val78 evaluation.",
                              "Heuristic flags are not definitive merge/fragment labels."],
              "production_config_modified": False, "review": None}
    if review_path:
        review = json.loads(Path(review_path).read_text())
        if review.get("input_artifact_fingerprint") != source_prov["input_artifact_fingerprint"]:
            raise ValueError("Visual-review evidence belongs to different input artifacts")
        if (set(review.get("plant_observations", {})) != set(PLANTS)
                or not all(isinstance(v, str) and v.strip() for v in review["plant_observations"].values())):
            raise ValueError("Review must cover all five plants, including 0022")
        candidate = review.get("recommended_config_id")
        if candidate not in grid or not review.get("reasoning") or not review.get("reviewer"):
            raise ValueError("Invalid/incomplete multi-criteria visual review")
        decision = review.get("decision")
        readiness = review.get("phase4_readiness")
        if decision not in ("KEEP CURRENT DEFAULT", "PROMOTE NEW CONFIG", "NO CONFIG READY — NEED ALGORITHM CHANGE"):
            raise ValueError("Invalid qualification decision")
        if readiness not in ("READY FOR VAL78", "NOT READY FOR VAL78"):
            raise ValueError("Invalid val78 readiness")
        if (decision == "KEEP CURRENT DEFAULT" and candidate != BASELINE
                or decision == "PROMOTE NEW CONFIG" and candidate == BASELINE):
            raise ValueError("Decision and recommended configuration disagree")
        if decision == "NO CONFIG READY — NEED ALGORITHM CHANGE" and readiness != "NOT READY FOR VAL78":
            raise ValueError("An unready algorithm cannot be qualified for val78")
        if mode != "both" or not summary["baseline_reproduced_exactly"]:
            raise ValueError("A final review requires both modes and exact archived baseline reproduction")
        result.update(recommended_config=vars(grid[candidate]), recommended_config_id=candidate,
                      decision=decision, phase4_readiness=readiness, reasoning=review["reasoning"],
                      review=review, review_sha256=sha256_file(review_path))
    return result


def render_charts(out, rows, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for source, short in ((SOURCES["oracle"], "oracle"), (SOURCES["predicted"], "predicted")):
        selected = [r for r in rows if r["semantic_source"] == source]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), constrained_layout=True)
        lo, hi = min(r["accepted_leaf_instances"] for r in selected), max(r["accepted_leaf_instances"] for r in selected)
        for ax, plant in zip(axes, PLANTS):
            matrix = np.array([[next(r["accepted_leaf_instances"] for r in selected if r["sample_id"] == plant
                               and r["radius_m"] == radius and r["max_neighbors"] == k)
                               for k in sweep.max_neighbors] for radius in sweep.radii_m])
            ax.imshow(matrix, vmin=lo, vmax=hi, cmap="viridis", aspect="auto")
            for (i, j), val in np.ndenumerate(matrix):
                ax.text(j, i, str(val), ha="center", va="center", color="white", fontsize=11)
            ax.set(xticks=range(3), xticklabels=sweep.max_neighbors, yticks=range(4),
                   yticklabels=sweep.radii_m, xlabel="max neighbors", ylabel="radius (m)", title=plant)
        fig.suptitle(short + ": accepted geometric components (not verified leaves)")
        fig.savefig(out / (short + "_component_counts.png"), dpi=150)
        plt.close(fig)
    for metric, title in (("instance_runtime_ms", "Instance runtime (ms; extractor only)"),
                          ("unassigned_fraction", "Unassigned fraction of leaf points")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, source in zip(axes, SOURCES.values()):
            selected = [r for r in rows if r["semantic_source"] == source]
            if not selected:
                ax.axis("off")
                continue
            for k in sweep.max_neighbors:
                values = [np.mean([r[metric] for r in selected if r["radius_m"] == radius and r["max_neighbors"] == k])
                          for radius in sweep.radii_m]
                ax.plot(sweep.radii_m, values, "o-", label=f"k={k}")
            ax.set(title=source, xlabel="radius (m)", ylabel=title, xticks=sweep.radii_m)
            ax.legend()
        fig.savefig(out / (metric + ".png"), dpi=150)
        plt.close(fig)


def render_visuals(out, records, labels, rows, selected_ids):
    import colorsys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    indexed = {(r["sample_id"], r["semantic_source"], r["config_id"]): r for r in rows}
    for record in records:
        sample = np.linspace(0, len(record.xyz)-1, min(6500, len(record.xyz)), dtype=int)
        xyz = record.xyz[sample]
        keys = [k for k in selected_ids if (record.sample_id, record.source, k) in labels]
        fig, axes = plt.subplots(2, len(keys), figsize=(4*len(keys), 7), squeeze=False, constrained_layout=True)
        html = make_subplots(rows=1, cols=len(keys), specs=[[{"type": "scene"}]*len(keys)], subplot_titles=keys)
        for j, key in enumerate(keys):
            partition = labels[(record.sample_id, record.source, key)]
            def color(i):
                return (.2, .2, .2) if i == 0 else (.7, .7, .7) if i < 0 else colorsys.hsv_to_rgb((i*.381966)%1, .75, .8)
            colors = [color(int(i)) for i in partition[sample]]
            count = indexed[(record.sample_id, record.source, key)]["accepted_leaf_instances"]
            for ax, horizontal in zip(axes[:, j], (0, 1)):
                ax.scatter(xyz[:, horizontal], xyz[:, 2], c=colors, s=1, rasterized=True)
                for instance_id in np.unique(partition[partition > 0]):
                    center = record.xyz[partition == instance_id].mean(0)
                    ax.text(center[horizontal], center[2], str(instance_id), fontsize=6,
                            bbox={"facecolor": "white", "alpha": .65, "pad": .1, "edgecolor": "none"})
                ax.set(xlabel=("X" if horizontal == 0 else "Y") + " (m)", ylabel="Z (m)",
                       title=f"{key}: {count} components", xlim=(record.xyz[:, horizontal].min(), record.xyz[:, horizontal].max()),
                       ylim=(record.xyz[:, 2].min(), record.xyz[:, 2].max()), aspect="equal")
            rgb = [f"rgb({round(r*255)},{round(g*255)},{round(b*255)})" for r, g, b in colors]
            html.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                marker={"size": 1.5, "color": rgb}, text=[f"instance {i}" for i in partition[sample]], showlegend=False), row=1, col=j+1)
            html.update_scenes(aspectmode="data", row=1, col=j+1)
        fig.suptitle(f"{record.sample_id} / {record.source}\nDark stalk; grey unassigned; colors/IDs are components, not GT leaves")
        short = "oracle" if record.source == SOURCES["oracle"] else "predicted"
        stem = record.sample_id + "_" + short
        fig.savefig(out / (stem + ".png"), dpi=180)
        plt.close(fig)
        html.update_layout(title=f"{record.sample_id} / {record.source}; component IDs are config-local", height=650)
        html.write_html(str(out / (stem + ".html")), include_plotlyjs="directory")


def report_text(rows, summary, recommended, comparison):
    lines = ["# Phase 4A instance sweep", "", "Component count is not verified biological leaf count.", "",
             f"Decision: **{recommended['decision']}**", f"Readiness: **{recommended['phase4_readiness']}**", "",
             "Oracle is primary; predicted-mode component deltas are robustness diagnostics only.", "",
             "| Plant | Baseline oracle | Recommended oracle | Baseline predicted | Recommended predicted |",
             "|---|---:|---:|---:|---:|"]
    key = recommended["recommended_config_id"]
    indexed = {(r["sample_id"], r["semantic_source"], r["config_id"]): r for r in rows}
    for plant in PLANTS:
        values = [indexed.get((plant, source, cfg), {}).get("accepted_leaf_instances", "not run")
                  for source, cfg in [(SOURCES["oracle"], BASELINE), (SOURCES["oracle"], key),
                                      (SOURCES["predicted"], BASELINE), (SOURCES["predicted"], key)]]
        lines.append(f"| {plant} | " + " | ".join(map(str, values)) + " |")
    lines.extend(["", "## Decision reasoning", "", *["- " + x for x in recommended["reasoning"]], "",
                  "## Runtime and stability", "", "See runtime.csv, stability.csv and charts/. One timing per cell; P95 is descriptive across five plants, not a repeated-benchmark uncertainty estimate.",
                  f"Archived baseline reproduced exactly: {summary['baseline_reproduced_exactly']}. Flagged adjacent count-collapse events: {len(summary['collapse_events'])}.",
                  "", "## Limitations", "", *["- " + x for x in recommended["limitations"]], "",
                  "No production config is modified. Use recommendation.json and an explicit reviewed config only after approval."])
    if recommended.get("review"):
        lines.extend(["", "## Visual findings", ""])
        for plant, note in recommended["review"]["plant_observations"].items():
            lines.append(f"- {plant}: {note}")
    lines.extend(["", "## Complete count grid", "", "Counts are geometric components, not verified leaves."])
    for source in SOURCES.values():
        lines.extend(["", "### " + source, "", "| Config | " + " | ".join(PLANTS) + " |",
                      "|---|" + "---:|"*len(PLANTS)])
        for config_id in dict.fromkeys(row["config_id"] for row in rows):
            values = [indexed.get((plant, source, config_id), {}).get("accepted_leaf_instances", "not run") for plant in PLANTS]
            lines.append("| " + config_id + " | " + " | ".join(map(str, values)) + " |")
    lines.extend(["", "## All runtime groups (extractor wall ms)", "",
                  "| Mode | Config | Mean | Median | P95 | Max |", "|---|---|---:|---:|---:|---:|"])
    for source, groups in summary["by_semantic_source"].items():
        for config_id, metrics in groups.items():
            timing = metrics["instance_runtime_ms"]
            lines.append("| " + source + " | " + config_id + " | " + " | ".join(f"{timing[k]:.1f}" for k in ("mean", "median", "p95", "max")) + " |")
    lines.extend(["", "## Diagnostic interpretation", "",
        "Near-threshold means 100–200 accepted points inclusive; very small rejected means <=25 points. Unassigned fraction uses all semantic leaf points as denominator. Median size uses accepted components; raw median is separate.",
        "Heuristic risk flags use size >3x accepted median, vertical span >60% plant height, contact-height span >0.15 m, or multiple stalk contact-height groups. See configuration.json for exact frozen thresholds. No flag is a definitive merge label.",
        "See per_plant_config.csv for every fragmentation/attachment/coverage diagnostic and component_details/ for size distributions. No raw RGB palette has been verified."])
    return "\n".join(lines) + "\n"


def run_sweep(input_artifacts, output_dir, config=None, sweep=None, mode="both", plants=None,
              split="val", review_path=None, visualize=True, code_commit=None):
    if split != "val":
        raise ValueError("Only fixed val5 validation artifacts permitted; no test access")
    if mode not in ("both", "oracle", "predicted"):
        raise ValueError("Invalid semantic mode")
    config, sweep = config or PipelineConfig(), sweep or SweepConfig()
    records, source_prov = load_val5(input_artifacts, config, plants, split)
    records = [r for r in records if mode == "both" or r.source == SOURCES[mode]]
    out = Path(output_dir).resolve()
    source_root = Path(input_artifacts).resolve()
    if out.exists() or out.is_relative_to(source_root) or source_root.is_relative_to(out):
        raise ValueError("Output must be new and separate from immutable source artifacts")
    out.mkdir(parents=True)
    configuration = {"sweep": sweep.to_dict(), "pipeline_config": config.to_dict(), "mode": mode,
                     "diagnostic_policy": DIAGNOSTIC_POLICY, "plants": list(PLANTS)}
    write_json(out / "configuration.json", configuration)
    prov = {**provenance(code_commit), **source_prov,
            "instance_implementation_sha256": sha256_file(Path(__file__).parents[1] / "phenotyping" / "instances.py"),
            "diagnostic_implementation_sha256": {name: sha256_file(Path(__file__).with_name(name))
                for name in ("instance_artifacts.py", "instance_diagnostics.py", "instance_sweep.py")},
            "controlled_factors": ["neighborhood_radius_m", "max_neighbors"],
            "source_input_directory": str(source_root), "inference_calls": 0,
            "graph_edge_instrumentation": "metadata-only graph.nnz//2; extraction algorithm unchanged"}
    write_json(out / "provenance.json", prov)
    rows, labels = [], {}
    total = len(records)*len(sweep.grid())
    try:
        for record in records:  # load_val5 orders all oracle plants before predicted plants
            context = attachment_context(record)
            for key, instance_config in sweep.grid():
                start = time.perf_counter()
                structure = InstanceExtractor(instance_config).extract(record.xyz, record.semantic)
                elapsed = (time.perf_counter() - start)*1000
                partition = structure.instance_labels(len(record.xyz))
                before = time.perf_counter()
                diagnostics, details = diagnose(record, structure, context, instance_config, elapsed)
                diagnostics_ms = (time.perf_counter() - before)*1000
                base_exact = bool(np.array_equal(partition, record.baseline_instances)) if key == BASELINE else None
                row = {"sample_id": record.sample_id, "semantic_source": record.source, "config_id": key,
                       "radius_m": instance_config.neighborhood_radius_m, "max_neighbors": instance_config.max_neighbors,
                       **diagnostics, "diagnostics_runtime_ms": diagnostics_ms,
                       "baseline_exact_partition": base_exact,
                       "baseline_saved_component_count": record.report["structure"]["accepted_leaf_instances"],
                       "partition_sha256": hashlib.sha256(partition.astype("<i4").tobytes()).hexdigest()}
                rows.append(row)
                labels[(record.sample_id, record.source, key)] = partition
                folder = out / "partitions" / record.sample_id / record.source
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / (key + ".npz"), instance_labels=partition)
                detail_folder = out / "component_details" / record.sample_id / record.source
                detail_folder.mkdir(parents=True, exist_ok=True)
                write_json(detail_folder / (key + ".json"), details)
                write_json(out / "progress.json", {"status": "running", "completed": len(rows), "total": total,
                                                   "last": [record.sample_id, record.source, key]})
                print(f"{len(rows)}/{total} {record.sample_id} {record.source} {key} components={len(structure.leaves)} runtime_ms={elapsed:.1f}", flush=True)
        stability = stability_analysis(records, rows, labels, sweep)
        summary = summarize(rows, stability)
        compare = comparison_rows(rows)
        selected = recommendation(source_prov, summary, sweep, mode, review_path)
        for row in rows:
            base = labels[(row["sample_id"], row["semantic_source"], BASELINE)]
            record = next(r for r in records if r.sample_id == row["sample_id"] and r.source == row["semantic_source"])
            row.update({"vs_baseline_" + k: v for k, v in partition_comparison(base,
                       labels[(row["sample_id"], row["semantic_source"], row["config_id"])], record.semantic == 0).items()})
        write_csv(out / "per_plant_config.csv", rows)
        write_csv(out / "oracle_comparison.csv", [r for r in rows if r["semantic_source"] == SOURCES["oracle"]])
        write_csv(out / "predicted_comparison.csv", compare)
        write_csv(out / "runtime.csv", [{k: r[k] for k in ("sample_id", "semantic_source", "config_id", "radius_m", "max_neighbors",
                    "leaf_point_count", "graph_undirected_edges", "instance_runtime_ms", "diagnostics_runtime_ms")} for r in rows])
        write_csv(out / "stability.csv", stability)
        write_json(out / "summary.json", summary)
        write_json(out / "recommendation.json", selected)
        approved_copy = config.to_dict()
        approved_copy["instances"] = selected["recommended_config"]
        (out / "recommended_pipeline.yaml").write_text("# UNPROMOTED review candidate; use only after approval.\n" + yaml.safe_dump(approved_copy, sort_keys=False))
        (out / "report.md").write_text(report_text(rows, summary, selected, compare))
        if visualize:
            (out / "charts").mkdir()
            render_charts(out / "charts", rows, sweep)
            (out / "visualizations").mkdir()
            visual_ids = list(dict.fromkeys([BASELINE, "r010_k16", "r025_k32", selected["recommended_config_id"]]))
            render_visuals(out / "visualizations", records, labels, rows, visual_ids)
        verify_unchanged(input_artifacts, source_prov)
        write_json(out / "progress.json", {"status": "complete", "completed": total, "total": total})
        files = {p.relative_to(out).as_posix(): {"size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
                 for p in sorted(out.rglob("*")) if p.is_file()}
        write_json(out / "artifacts.json", {"schema_version": 1, "files": files})
        return summary, selected
    except Exception as exc:
        write_json(out / "progress.json", {"status": "failed", "completed": len(rows), "total": total,
                                           "error_type": type(exc).__name__, "error": str(exc)})
        raise


def finalize_review(input_artifacts, output_dir, review_path, visualize=True):
    """Attach a reviewed decision to this new sweep; never rerun extraction or inference.

    Validates the sealed sweep and immutable val5 source before updating only the
    decision, report, unpromoted config copy and selected visualizations.
    """
    import csv
    out = Path(output_dir).resolve()
    source = Path(input_artifacts).resolve()
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Review output must remain separate from original val5 artifacts")
    inventory = json.loads((out / "artifacts.json").read_text())["files"]
    for name, info in inventory.items():
        path = (out / name).resolve()
        if (not path.is_relative_to(out) or path.stat().st_size != info["size_bytes"]
                or sha256_file(path) != info["sha256"]):
            raise ValueError("Completed sweep inventory mismatch: " + name)
    if json.loads((out / "progress.json").read_text())["status"] != "complete":
        raise ValueError("Cannot review an incomplete sweep")
    configuration = json.loads((out / "configuration.json").read_text())
    config_data = configuration["pipeline_config"]
    config = PipelineConfig.load()  # this protocol freezes the original pipeline
    if config.to_dict() != config_data:
        raise ValueError("Unsupported original pipeline contract")
    records, source_prov = load_val5(source, config)
    recorded_prov = json.loads((out / "provenance.json").read_text())
    if source_prov["input_artifact_fingerprint"] != recorded_prov["input_artifact_fingerprint"]:
        raise ValueError("Review source differs from measured sweep")
    sweep_data = configuration["sweep"]
    sweep = SweepConfig(tuple(sweep_data["radii_m"]), tuple(sweep_data["max_neighbors"]), **sweep_data["fixed"])
    summary = json.loads((out / "summary.json").read_text())
    selected = recommendation(source_prov, summary, sweep, configuration["mode"], review_path)
    rows = []
    text_fields = {"sample_id", "semantic_source", "config_id", "partition_sha256"}
    with open(out / "per_plant_config.csv", newline="") as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                if key not in text_fields:
                    row[key] = None if value == "" else (value == "True" if value in ("True", "False") else json.loads(value))
            rows.append(row)
    labels = {}
    selected_ids = list(dict.fromkeys([BASELINE, "r010_k16", "r025_k32", selected["recommended_config_id"]]))
    if visualize:
        for record in records:
            for key in selected_ids:
                path = out / "partitions" / record.sample_id / record.source / (key + ".npz")
                with np.load(path, allow_pickle=False) as data:
                    labels[(record.sample_id, record.source, key)] = data["instance_labels"]
        folder = out / "visualizations"
        folder.mkdir(exist_ok=True)
        render_visuals(folder, records, labels, rows, selected_ids)
    verify_unchanged(source, source_prov)
    write_json(out / "recommendation.json", selected)
    write_json(out / "review.json", selected["review"])
    config_data["instances"] = selected["recommended_config"]
    (out / "recommended_pipeline.yaml").write_text("# UNPROMOTED review candidate; use only after approval.\n" + yaml.safe_dump(config_data, sort_keys=False))
    (out / "report.md").write_text(report_text(rows, summary, selected, comparison_rows(rows)))
    recorded_prov["qualification"] = selected["phase4_readiness"]
    recorded_prov["decision"] = selected["decision"]
    recorded_prov["review_finalization"] = {"extraction_calls": 0, "inference_calls": 0,
                                           "review_sha256": selected["review_sha256"]}
    write_json(out / "provenance.json", recorded_prov)
    files = {p.relative_to(out).as_posix(): {"size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(out.rglob("*")) if p.is_file() and p.name != "artifacts.json"}
    write_json(out / "artifacts.json", {"schema_version": 1, "files": files})
    return summary, selected
