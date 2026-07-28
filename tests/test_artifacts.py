from __future__ import annotations

import json
from pathlib import Path

import pytest

from pol.artifacts import ArtifactStore, verify_artifact


def test_content_addressed_artifact_is_atomic_and_tamper_evident(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    identity = {"schema_version": "test-v1", "value": 3}
    ref = store.reference("example", identity)

    def writer(root: Path):
        (root / "payload.txt").write_text("stable\n", encoding="utf-8")
        return ("payload.txt",)

    store.publish(ref, identity=identity, writer=writer)
    manifest = verify_artifact(ref.path)
    assert manifest["artifact_id"] == ref.artifact_id
    (ref.path / "payload.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes do not match"):
        verify_artifact(ref.path)



def test_artifact_manifest_identity_is_self_authenticating(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    identity = {"schema_version": "test-v1", "value": 3}
    ref = store.reference("example", identity)

    def writer(root: Path):
        (root / "payload.txt").write_text("stable\n", encoding="utf-8")
        return ("payload.txt",)

    store.publish(ref, identity=identity, writer=writer)
    manifest_path = ref.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["value"] = 4
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identity hash"):
        verify_artifact(ref.path)


def test_artifact_writer_cannot_publish_extra_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.reference("example", {"value": 1})

    def writer(root: Path):
        (root / "declared.txt").write_text("a", encoding="utf-8")
        (root / "extra.txt").write_text("b", encoding="utf-8")
        return ("declared.txt",)

    with pytest.raises(ValueError, match="artifact tree mismatch"):
        store.publish(ref, identity={"value": 1}, writer=writer)
    assert not ref.path.exists()


def test_store_does_not_reuse_a_valid_artifact_under_the_wrong_identity(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    original = store.reference("example", {"value": 1})

    def writer(root: Path):
        (root / "payload.txt").write_text("one", encoding="utf-8")
        return ("payload.txt",)

    store.publish(original, identity={"value": 1}, writer=writer)
    requested = store.reference("example", {"value": 2})
    requested.path.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copytree(original.path, requested.path)
    assert not store.exists(requested)
