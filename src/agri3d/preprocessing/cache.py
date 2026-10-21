"""Opt-in local disk cache before sampling/augmentation, with byte provenance."""

import hashlib
import inspect
import json
import os
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from agri3d.io.ply import PLYReader, PointCloudData
from agri3d.preprocessing.filtering import CloudFilter
from agri3d.preprocessing.labels import decode_maizefield3d_labels
from agri3d.preprocessing.normalization import CloudNormalizer

CACHE_VERSION = 2


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline_fingerprint():
    sources = [
        inspect.getsource(item)
        for item in (
            PLYReader,
            PointCloudData,
            CloudFilter,
            inspect.getmodule(decode_maizefield3d_labels),
            CloudNormalizer,
        )
    ]
    return hashlib.sha256("\n".join(sources).encode()).hexdigest()


def payload_hash(xyz, attributes, metadata):
    digest = hashlib.sha256(metadata.encode())
    for name, array in [("xyz", xyz), *sorted(attributes.items())]:
        array = np.ascontiguousarray(array)
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class NormalizedCloudCache:
    def __init__(
        self,
        directory,
        boundary_fingerprint=None,
        feature_provenance=None,
        feature_contract="xyz",
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pipeline_hash = pipeline_fingerprint()
        self.boundary_fingerprint = boundary_fingerprint
        self.feature_provenance = feature_provenance
        self.feature_contract = feature_contract

    def identity(self, source, sample_identity=None):
        return {
            "cache_version": CACHE_VERSION,
            "features": self.feature_contract,
            "feature_provenance": self.feature_provenance,
            "sample_identity": sample_identity,
            "storage": "uncompressed_npz",
            "source_file": str(Path(source).resolve()),
            "source_sha256": sha256_file(source),
            "pipeline_sha256": self.pipeline_hash,
            "boundary_sha256": self.boundary_fingerprint,
            "numpy_version": np.__version__,
            "plyfile_version": version("plyfile"),
            "annotation_mode": "maizefield3d",
            "sor": False,
            "normalization": "centroid_unit_sphere_float32",
        }

    def get(self, source, produce, sample_identity=None):
        identity = self.identity(
            source, sample_identity
        )  # Full bytes checked every access, not only mtime/size.
        encoded = json.dumps(identity, sort_keys=True)
        key = hashlib.sha256(encoded.encode()).hexdigest()
        destination = self.directory / f"{key}.npz"
        self.last_hit = destination.exists()
        if self.last_hit:
            try:
                with np.load(destination, allow_pickle=False) as data:
                    if str(data["identity"]) != encoded:
                        raise ValueError("Cache identity mismatch")
                    xyz = data["xyz"].copy()
                    attrs = {
                        k[6:]: data[k].copy()
                        for k in data.files
                        if k.startswith("attr__")
                    }
                    metadata = str(data["metadata"])
                    if str(data["payload_sha256"]) != payload_hash(
                        xyz, attrs, metadata
                    ):
                        raise ValueError("Cache payload checksum mismatch")
                    return PointCloudData(xyz, attrs, json.loads(metadata)).validate(
                        finite=True
                    )
            except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
                raise ValueError(
                    f"Invalid normalized cache entry {destination}; remove/rebuild it"
                ) from exc
        cloud = produce().validate(finite=True)
        # Reject a source edited during cache population rather than pairing stale identity/data.
        if sha256_file(source) != identity["source_sha256"]:
            raise ValueError("Source changed while building cache")
        metadata = json.dumps(cloud.metadata, sort_keys=True, allow_nan=False)
        arrays = {
            "xyz": cloud.xyz,
            "identity": np.asarray(encoded),
            "metadata": np.asarray(metadata),
            "payload_sha256": np.asarray(
                payload_hash(cloud.xyz, cloud.attributes, metadata)
            ),
            **{"attr__" + key: value for key, value in cloud.attributes.items()},
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.directory, suffix=".npz", delete=False
            ) as stream:
                temporary = Path(stream.name)
                np.savez(stream, **arrays)
            os.replace(
                temporary, destination
            )  # Concurrent workers may populate the same identical key.
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return cloud
