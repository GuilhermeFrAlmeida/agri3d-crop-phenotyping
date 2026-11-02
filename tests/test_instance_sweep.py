"""CPU-only Phase 4A protocol tests using synthetic, explicitly val-scoped artifacts."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from agri3d.analysis.instance_artifacts import (
    PLANTS,
    SOURCES,
    load_val5,
    verify_unchanged,
)
from agri3d.analysis.instance_diagnostics import (
    attachment_context,
    diagnose,
    partition_comparison,
)
from agri3d.analysis.instance_sweep import (
    BASELINE,
    SweepConfig,
    comparison_rows,
    recommendation,
    run_sweep,
    stats,
)
from agri3d.phenotyping.instances import InstanceExtractor
from agri3d.pipeline.artifacts import write_json
from agri3d.pipeline.config import PipelineConfig
from agri3d.pipeline.provenance import sha256_file


@pytest.fixture
def artifacts(tmp_path):
    root = tmp_path / 'val5'
    root.mkdir()
    cfg = PipelineConfig().to_dict()
    model = {'model_family': 'pointnet2_ssg', 'feature_mode': 'xyz', 'checkpoint_sha256': 'c'*64,
                 'manifest_sha256': 'm'*64, 'input_points': 10000, 'selected_weights_verified': True,
                 'repository_metadata': {'frozen_run1_policy_enforced': True, 'compatible_experiment_policy': 'run1_xyz'}}
    write_json(root / 'configuration.json', cfg)
    write_json(root / 'provenance.json', {'selected_sample_ids': list(PLANTS), 'split': 'val',
        'instance_gt_status': 'unverified', 'semantic_model': model, 'manifest_sha256': 'm'*64})
    rng = np.random.default_rng(42)
    xyz = np.concatenate([rng.normal([.1, 0, .4], .001, (110, 3)),
                          rng.normal([-.1, 0, .8], .001, (110, 3)),
                          rng.normal([0, 0, .5], .001, (20, 3))]).astype('float32')
    rows = []
    for plant in PLANTS:
        for source in SOURCES.values():
            folder = root / plant / source
            folder.mkdir(parents=True)
            semantic = np.r_[np.zeros(220, dtype='int64'), np.ones(20, dtype='int64')]
            if source == SOURCES['predicted']:
                semantic[0] = 1
            partition = InstanceExtractor().extract(xyz, semantic).instance_labels(len(xyz))
            report = {'schema_version': 1, 'plant_id': plant, 'semantic_source': source, 'split': 'val',
                'clean_point_count': len(xyz), 'semantic_model': model,
                'structure': {'accepted_leaf_instances': len(np.unique(partition[partition > 0]))},
                'provenance': {'configuration': cfg, 'manifest_sha256': 'm'*64, 'input_sha256': 'i'*64}}
            write_json(folder / 'phenotype.json', report)
            vertices = np.empty(len(xyz), dtype=[(a, 'f4') for a in 'xyz'] +
                [('semantic_label', 'i4'), ('instance_id', 'i4'), ('source_index', 'i4')])
            for i, a in enumerate('xyz'):
                vertices[a] = xyz[:, i]
            vertices['semantic_label'] = semantic
            vertices['instance_id'] = partition
            vertices['source_index'] = np.arange(len(xyz))*2
            PlyData([PlyElement.describe(vertices, 'vertex')]).write(str(folder / 'structure.ply'))
            np.save(folder / 'semantic_labels.npy', semantic)
            np.save(folder / 'instance_labels.npy', partition)
            write_json(folder / 'artifacts.json', {'files': {p.name: {'size_bytes': p.stat().st_size,
                'sha256': sha256_file(p)} for p in folder.iterdir()}})
            rows.append({'sample_id': plant, 'semantic_source': source, 'artifact_dir': folder.relative_to(root)})
    with open(root / 'per_plant.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_exact_grid_and_validation():
    sweep = SweepConfig.load('configs/phase4a_instance_sweep.yaml')
    assert len(sweep.grid()) == 12
    assert [key for key, _ in sweep.grid()] == [f'r{r:03d}_k{k}' for r in [10, 15, 20, 25] for k in [16, 24, 32]]
    assert dict(sweep.grid())[BASELINE] == PipelineConfig().instances
    for kwargs in [{'radii_m': (.01,)}, {'max_neighbors': (True, 24, 32)},
                   {'min_leaf_points': 200}, {'attachment_radius_m': .04}]:
        with pytest.raises(ValueError):
            SweepConfig(**kwargs)


def test_readonly_artifact_reuse_and_exact_modes(artifacts):
    records, prov = load_val5(artifacts)
    assert [(r.sample_id, r.source) for r in records] == [(p, s) for s in SOURCES.values() for p in PLANTS]
    assert not records[0].xyz.flags.writeable
    assert not np.array_equal(records[0].semantic, records[5].semantic)
    assert records[0].geometry_sha256 == records[5].geometry_sha256
    verify_unchanged(artifacts, prov)
    with open(artifacts / '0006' / SOURCES['oracle'] / 'semantic_labels.npy', 'ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ValueError, match='changed'):
        verify_unchanged(artifacts, prov)
    with pytest.raises(ValueError, match='checksum'):
        load_val5(artifacts)


def test_no_test_access_or_cohort_substitution(tmp_path):
    missing = tmp_path / 'MUST_NOT_OPEN'
    with pytest.raises(ValueError, match='test'):
        load_val5(missing, split='test')
    with pytest.raises(ValueError, match='exactly'):
        load_val5(missing, plants=['0006'])
    with pytest.raises(ValueError, match='test'):
        run_sweep(missing, tmp_path / 'out', split='test')


def test_diagnostics_aggregation_and_partition_determinism(artifacts):
    records, _ = load_val5(artifacts)
    rec = records[0]
    one = InstanceExtractor().extract(rec.xyz, rec.semantic)
    two = InstanceExtractor().extract(rec.xyz, rec.semantic)
    np.testing.assert_array_equal(one.instance_labels(len(rec.xyz)), two.instance_labels(len(rec.xyz)))
    row, detail = diagnose(rec, one, attachment_context(rec), PipelineConfig().instances, 12.)
    assert row['accepted_leaf_instances'] == 2
    assert row['near_threshold_accepted_fraction'] == 1
    assert row['unassigned_fraction'] == 0
    assert row['graph_undirected_edges'] > 0
    assert len(detail['raw_component_sizes']) == 2
    assert stats([1, 2, 3, 4, 5]) == {'n': 5, 'mean': 3., 'median': 3., 'p95': 4.8, 'max': 5., 'min': 1.}
    labels = one.instance_labels(len(rec.xyz))
    assert partition_comparison(labels, labels, rec.semantic == 0)['leaf_adjusted_rand_index'] == 1
    merged = np.where(labels > 0, 1, labels)
    assert partition_comparison(labels, merged, rec.semantic == 0)['absorbed_previous_components'] == 1


def test_recommendation_requires_bound_complete_review(tmp_path):
    prov = {'input_artifact_fingerprint': 'fingerprint'}
    summary = {'baseline_reproduced_exactly': True}
    review = {'input_artifact_fingerprint': 'fingerprint', 'reviewer': 'synthetic test',
        'plant_observations': {p: 'checked' for p in PLANTS}, 'recommended_config_id': 'r020_k24',
        'decision': 'PROMOTE NEW CONFIG', 'phase4_readiness': 'READY FOR VAL78', 'reasoning': ['synthetic only']}
    path = tmp_path / 'review.json'
    write_json(path, review)
    result = recommendation(prov, summary, SweepConfig(), 'both', path)
    assert json.loads(json.dumps(result))['recommended_config']['max_neighbors'] == 24
    assert not result['production_config_modified']
    with pytest.raises(ValueError, match='both modes'):
        recommendation(prov, summary, SweepConfig(), 'oracle', path)
    review['input_artifact_fingerprint'] = 'other'
    write_json(path, review)
    with pytest.raises(ValueError, match='different input'):
        recommendation(prov, summary, SweepConfig(), 'both', path)
    assert recommendation(prov, summary, SweepConfig(), 'both')['phase4_readiness'] == 'NOT READY FOR VAL78'


@pytest.mark.parametrize('mode,expected_rows', [('oracle', 60), ('predicted', 60), ('both', 120)])
def test_sweep_never_infers_and_preserves_production_config(artifacts, tmp_path, monkeypatch, mode, expected_rows):
    import builtins
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(('agri3d.models', 'agri3d.training', 'agri3d.inference.pointnet2_backend')):
            raise AssertionError('Geometry-only sweep must not import model inference')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    production = Path('configs/phenotyping_pipeline.yaml')
    before = production.read_bytes()
    out = tmp_path / f'out-{mode}'
    summary, selected = run_sweep(artifacts, out, mode=mode, visualize=False)
    assert summary['rows'] == expected_rows
    assert summary['baseline_reproduced_exactly']
    assert selected['decision'] == 'KEEP CURRENT DEFAULT'
    assert production.read_bytes() == before
    assert json.loads((out / 'provenance.json').read_text())['inference_calls'] == 0
    rows = list(csv.DictReader((out / 'per_plant_config.csv').open()))
    assert len(rows) == expected_rows
    assert {r['semantic_source'] for r in rows} == (set(SOURCES.values()) if mode == 'both' else {SOURCES[mode]})
    assert len(comparison_rows([{'sample_id': r['sample_id'], 'semantic_source': r['semantic_source'],
        'config_id': r['config_id'], 'accepted_leaf_instances': int(r['accepted_leaf_instances']),
        'unassigned_fraction': float(r['unassigned_fraction']), 'instance_runtime_ms': float(r['instance_runtime_ms'])}
        for r in rows])) == (60 if mode == 'both' else 0)
    with pytest.raises(ValueError, match='new and separate'):
        run_sweep(artifacts, out, visualize=False)


def test_review_finalization_reuses_partitions_without_extraction(artifacts, tmp_path, monkeypatch):
    from agri3d.analysis.instance_sweep import finalize_review
    out = tmp_path / 'completed'
    run_sweep(artifacts, out, visualize=False)
    original_csv = (out / 'per_plant_config.csv').read_bytes()
    source = json.loads((out / 'provenance.json').read_text())
    review = {'input_artifact_fingerprint': source['input_artifact_fingerprint'],
        'reviewer': 'synthetic test', 'plant_observations': {p: 'checked' for p in PLANTS},
        'recommended_config_id': BASELINE, 'decision': 'KEEP CURRENT DEFAULT',
        'phase4_readiness': 'READY FOR VAL78', 'reasoning': ['Synthetic fixture only']}
    path = tmp_path / 'review.json'
    write_json(path, review)

    def forbidden(*args, **kwargs):
        raise AssertionError('Review must reuse saved partitions, never run extraction')

    monkeypatch.setattr(InstanceExtractor, 'extract', forbidden)
    _, selected = finalize_review(artifacts, out, path, visualize=False)
    assert selected['phase4_readiness'] == 'READY FOR VAL78'
    assert (out / 'per_plant_config.csv').read_bytes() == original_csv
    assert json.loads((out / 'provenance.json').read_text())['review_finalization']['extraction_calls'] == 0
    (out / 'per_plant_config.csv').write_text('corrupt')
    with pytest.raises(ValueError, match='inventory'):
        finalize_review(artifacts, out, path, visualize=False)


@pytest.mark.parametrize('mutation', ['test', 'cohort', 'wrong_model', 'unqualified_model'])
def test_rejects_artifact_scope_and_policy_mismatch(artifacts, mutation):
    path = artifacts / 'provenance.json'
    value = json.loads(path.read_text())
    if mutation == 'test':
        value['split'] = 'test'
    elif mutation == 'cohort':
        value['selected_sample_ids'][-1] = '9999'
    elif mutation == 'wrong_model':
        value['semantic_model']['feature_mode'] = 'xyz_normals'
    else:
        value['semantic_model']['repository_metadata']['frozen_run1_policy_enforced'] = False
    write_json(path, value)
    with pytest.raises(ValueError):
        load_val5(artifacts)
