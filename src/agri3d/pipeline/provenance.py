"""Hashes and provenance without importing model/training implementations."""
import hashlib
import importlib.metadata
import platform
import re
import subprocess
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(code_commit=None):
    if code_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ValueError("code_commit must be a full lowercase commit SHA")
    source = "user_supplied" if code_commit else "git"
    if code_commit is None:
        try:
            code_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
                stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            source, code_commit = "unavailable", None
    versions = {}
    for package in ("numpy", "scipy", "torch", "plyfile", "onnx", "onnxruntime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"code_commit": code_commit, "code_commit_source": source,
            "python": platform.python_version(), "versions": versions,
            "qualification": "IMPLEMENTED — NOT YET EXPERIMENTALLY QUALIFIED"}
