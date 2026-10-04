import numpy as np

def validate_manifest(records):
    if not records:
        raise ValueError("Empty manifest")
    paths, ids, groups = set(), set(), {}
    for r in records:
        if not all(isinstance(r.get(k), str) and r[k] for k in ('path','sample_id','group_id','split')):
            raise ValueError("Manifest requires path, sample_id, group_id, split")
        if r['split'] not in ('train','val','test') or r['path'] in paths or r['sample_id'] in ids:
            raise ValueError("Invalid split or duplicate sample")
        if r['group_id'] in groups and groups[r['group_id']] != r['split']:
            raise ValueError("Group leakage between splits")
        paths.add(r['path']); ids.add(r['sample_id']); groups[r['group_id']] = r['split']
    return records

def make_group_split(records, seed=42):
    """Assign explicit metadata groups, never infer plant/genotype from filenames."""
    groups = sorted({r['group_id'] for r in records})
    if len(groups) < 3:
        raise ValueError("At least three groups required")
    np.random.default_rng(seed).shuffle(groups)
    n_val = max(1, int(round(len(groups)*0.15)))
    n_test = max(1, int(round(len(groups)*0.15)))
    n_train = len(groups)-n_val-n_test
    assignment = {g: ('train' if i < n_train else 'val' if i < n_train+n_val else 'test') for i,g in enumerate(groups)}
    result = [dict(r, split=assignment[r['group_id']]) for r in sorted(records, key=lambda r:r['sample_id'])]
    return validate_manifest(result)
