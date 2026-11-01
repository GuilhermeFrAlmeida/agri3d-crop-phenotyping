"""One-plant analysis with metric/model coordinates explicitly separated."""
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agri3d.inference.predictor import SemanticPrediction, SemanticPredictor
from agri3d.inference.projection import project_nearest
from agri3d.io.ply import PLYReader, PointCloudData
from agri3d.phenotyping.instances import InstanceExtractor
from agri3d.phenotyping.measurements import PhenotypeExtractor
from agri3d.phenotyping.schemas import PlantStructure
from agri3d.preprocessing.filtering import CloudFilter
from agri3d.preprocessing.labels import decode_maizefield3d_labels
from agri3d.preprocessing.normalization import CloudNormalizer
from agri3d.preprocessing.sampling import FixedSizeSampler
from agri3d.training.evaluate import SegmentationEvaluator

from .config import PipelineConfig
from .provenance import provenance, sha256_file


@dataclass
class AnalysisResult:
    report: dict
    cloud: PointCloudData
    sampled: PointCloudData
    prediction: SemanticPrediction | None
    semantic_labels: np.ndarray
    instance_labels: np.ndarray
    target_labels: np.ndarray | None
    structure: PlantStructure
    started: float


def analyze_plant(path, predictor: SemanticPredictor | None = None, config=None, sample_id=None,
                  semantic_source="run1_projected_prediction", evaluate_semantics=False,
                  code_commit=None):
    if semantic_source not in ("run1_projected_prediction", "ground_truth"):
        raise ValueError("Unsupported semantic source")
    config = config or PipelineConfig()
    if predictor is None and semantic_source != "ground_truth":
        raise ValueError("Predicted semantic mode requires a predictor")
    sample_id = sample_id or Path(path).stem
    started = time.perf_counter()
    timings = {}

    def stage(name, operation):
        start = time.perf_counter()
        result = operation()
        timings[name] = (time.perf_counter() - start) * 1000
        return result

    before_hash = sha256_file(path)
    raw = stage("load", lambda: PLYReader(path).load_point_cloud())
    metric = stage("filtering", lambda: CloudFilter().remove_nan_inf(raw))
    target = None
    if evaluate_semantics or semantic_source == "ground_truth":
        target = stage("semantic_decode", lambda: decode_maizefield3d_labels(metric)).attributes["label"]
    normalized = stage("normalization", lambda: CloudNormalizer(
        center=True, scale_to_unit_sphere=True).process(metric))
    digest = hashlib.sha256(f"{config.seed}:0:{sample_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    input_points = predictor.input_points if predictor else 10000
    sampled = stage("sampling", lambda: FixedSizeSampler(input_points).sample(normalized, rng))
    valid = sampled.attributes["valid_mask"]
    raw_ids = sampled.attributes["source_index"]
    clean_ids = np.searchsorted(metric.attributes["source_index"], raw_ids[valid])
    prediction = None
    if semantic_source == "ground_truth":
        labels = target.copy()
        timings.update(semantic_inference=0., projection=0.)
    else:
        prediction = stage("semantic_inference", lambda: predictor.predict(sampled.xyz))
        prediction.validate(len(sampled.xyz))
        if "cuda_forward_softmax" in prediction.timing_ms:
            timings["semantic_cuda_device_nested"] = prediction.timing_ms["cuda_forward_softmax"]
        labels, _ = stage("projection", lambda: project_nearest(
            metric.xyz, metric.xyz[clean_ids], prediction.labels[valid], clean_ids,
            prediction.probabilities[valid]))
    structure = stage("instances", lambda: InstanceExtractor(config.instances).extract(metric.xyz, labels))
    traits = stage("phenotypes", lambda: PhenotypeExtractor(config.phenotypes).extract(metric.xyz, structure))
    metrics = {}
    if target is not None and prediction is not None:
        sampled_target = target[clean_ids]
        metrics = {
            "sampled_10k": SegmentationEvaluator().calculate_metrics(prediction.labels[valid], sampled_target),
            "projected_full_cloud": SegmentationEvaluator().calculate_metrics(labels, target),
        }
    if sha256_file(path) != before_hash:
        raise ValueError("Input changed during analysis")
    report = {
        "schema_version": 1, "plant_id": sample_id, "source_file": str(Path(path).resolve()),
        "source_point_count": len(raw.xyz), "clean_point_count": len(metric.xyz),
        "semantic_source": semantic_source,
        "semantic_model": predictor.metadata if prediction is not None else None,
        "semantic": {"leaf_points": int((labels == 0).sum()), "stalk_points": int((labels == 1).sum()),
                     "projection_method": config.projection if prediction is not None else None,
                     "sampled_points": input_points, "valid_sampled_points": int(valid.sum())},
        "structure": structure.metadata, **traits.to_dict(), "semantic_metrics": metrics,
        "timing_ms": timings,
        "provenance": {**provenance(code_commit), "input_sha256": before_hash,
                       "configuration": config.to_dict(), "sampling_epoch": 0,
                       "coordinate_spaces": {"measurements": "clean full metric XYZ (meters)",
                                             "model": "full-cloud centroid/unit-sphere normalized float32 XYZ"},
                       "normalization": normalized.metadata,
                       "instance_gt_status": "unverified unless explicit verified palette supplied",
                       "phenotype_reference_status": "unavailable unless verified reference adapter supplied"},
    }
    return AnalysisResult(report, metric, sampled, prediction, labels,
                          structure.instance_labels(len(metric.xyz)), target, structure, started)
