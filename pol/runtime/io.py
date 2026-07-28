"""Byte-stable JSON and atomic file-writing helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    """Durably replace a regular file without exposing partial bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing to replace symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_json_bytes(value: object) -> bytes:
    """Canonical finite JSON bytes used by content identities."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def write_strict_json(path: Path, value: object) -> None:
    """Write sorted UTF-8 JSON with finite values and a trailing newline."""
    _atomic_replace_bytes(
        path,
        (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    )


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> None:
    """Atomically write a deterministic CSV table."""
    import csv
    import io

    materialized = list(rows)
    fields = list(fieldnames or ())
    if not fields and materialized:
        fields = list(materialized[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    if fields:
        writer.writeheader()
        writer.writerows(materialized)
    _atomic_replace_bytes(path, stream.getvalue().encode("utf-8"))


def atomic_torch_save(path: Path, value: object) -> None:
    """Atomically publish a PyTorch archive in the destination directory."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing to replace symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a regular file."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()
