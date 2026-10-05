import numpy as np

def validate_labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in 'iu' or not np.isin(labels, [0, 1]).all():
        raise ValueError("Semantic labels must be integer (N,), 0=leaf, 1=stalk")
    return labels.astype(np.int64)

def decode_labels(pcd, color_map=None, label_map=None):
    """Require verified RGB->class mapping or explicitly declared integer label encoding."""
    mapping = color_map if color_map is not None else label_map
    if mapping is not None and any(not isinstance(v, (int, np.integer)) or v not in (0,1) for v in mapping.values()):
        raise ValueError("Mapping targets must be integer semantic class IDs")
    if color_map is not None:
        if 'colors' not in pcd.attributes:
            raise ValueError("RGB annotations missing")
        colors = pcd.attributes['colors']
        labels = np.full(len(colors), -1, dtype=np.int64)
        for rgb, cls in color_map.items():
            if len(rgb) != 3 or any(v < 0 or v > 255 for v in rgb):
                raise ValueError("Invalid RGB palette")
            labels[np.all(colors == rgb, axis=1)] = cls
        validate_labels(labels)  # Reject unknown colors, never default to leaf.
    elif label_map is not None and 'label' in pcd.attributes:
        raw = pcd.attributes['label']
        if raw.dtype.kind not in 'iu':
            raise ValueError("Raw labels must be integers")
        labels = np.full(len(raw), -1, dtype=np.int64)
        for source, target in label_map.items():
            labels[raw == source] = target
        validate_labels(labels)
    else:
        raise ValueError("Verified color_map or explicit label_map required")
    result = pcd.select(np.arange(len(pcd.xyz)))
    result.attributes['label'] = labels
    result.metadata['label_encoding'] = '0=leaf,1=stalk;explicit_mapping'
    return result


def decode_maizefield3d_labels(pcd):
    """Decode annotated MaizeField3D: black=stalk (1), non-black=leaf (0).

    Apply only to segmented exports, not natural-color raw clouds. Original
    RGB values remain unchanged for subsequent instance-level validation.
    """
    if 'colors' not in pcd.attributes:
        raise ValueError("MaizeField3D requires RGB annotations (red, green, blue)")
    colors = np.asarray(pcd.attributes['colors'])
    if colors.shape != (len(pcd.xyz), 3):
        raise ValueError("MaizeField3D RGB annotations must have shape (N, 3)")
    if colors.dtype.kind not in 'iu' or np.any(colors < 0) or np.any(colors > 255):
        raise ValueError("MaizeField3D RGB annotations must be integer values in [0, 255]")
    pcd.validate()
    result = pcd.select(np.arange(len(pcd.xyz)))
    result.attributes['label'] = np.all(colors == 0, axis=1).astype(np.int64)
    result.metadata['label_encoding'] = 'maizefield3d:black=stalk(1),nonblack=leaf(0)'
    return result
