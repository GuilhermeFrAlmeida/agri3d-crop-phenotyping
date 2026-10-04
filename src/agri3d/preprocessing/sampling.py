import numpy as np

def voxel_indices(xyz, voxel_size):
    if voxel_size is None or not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    keys = np.floor((xyz.astype(np.float64) - xyz.min(axis=0)) / voxel_size)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return np.sort(indices)

class FixedSizeSampler:
    """Keep original points and attributes. Pad by repetition with a validity mask.

    Voxel representatives are original points, never averaged discrete labels.
    FPS is O(N*K); use voxel mode for large CPU clouds.
    """
    def __init__(self, num_points=10000, method='voxel', voxel_size=0.005):
        if not isinstance(num_points, int) or num_points <= 0 or method not in ('random','voxel','fps'):
            raise ValueError("Invalid sampling configuration")
        self.num_points, self.method, self.voxel_size = num_points, method, voxel_size

    def sample(self, pcd, rng=None):
        pcd.validate(finite=True)
        rng = np.random.default_rng() if rng is None else rng
        n, k = len(pcd.xyz), self.num_points
        candidates = voxel_indices(pcd.xyz, self.voxel_size) if self.method == 'voxel' else np.arange(n)
        if self.method == 'fps' and n > k:
            xyz = pcd.xyz.astype(np.float64)
            distance = np.full(n, np.inf)
            chosen = np.zeros(n, dtype=bool)
            indices = np.empty(k, dtype=np.int64)
            current = int(rng.integers(n))
            for j in range(k):
                indices[j] = current
                chosen[current] = True
                distance = np.minimum(distance, np.sum((xyz - xyz[current])**2, axis=1))
                distance[chosen] = -1
                current = int(np.argmax(distance))
        elif len(candidates) >= k:
            indices = rng.choice(candidates, k, replace=False)
        else:
            # Fill from unused original points before introducing any duplicates.
            rest = np.setdiff1d(np.arange(n), candidates)
            indices = np.concatenate([candidates, rng.choice(rest, min(k-len(candidates), len(rest)), replace=False)])
        valid = np.zeros(k, dtype=bool)
        valid[:len(indices)] = True
        if len(indices) < k:
            indices = np.concatenate([indices, rng.choice(indices, k-len(indices), replace=True)])
        result = pcd.select(indices)
        result.attributes['valid_mask'] = valid
        result.metadata.update(sampling_method=self.method, sampled_point_count=k)
        return result
