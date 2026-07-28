from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping

from pol.runtime.artifacts import RunTransaction, exact_artifact_tree, manifest_records
from pol.runtime.hashing import stable_object_hash
from pol.runtime.io import write_strict_json


@dataclass(frozen=True)
class ArtifactRef:
    kind: str
    artifact_id: str
    path: Path

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"


class ArtifactStore:
    """A small content-addressed store rooted at ``root``.

    Artifact identities are hashes of canonical JSON-compatible scientific
    inputs.  Files are staged in a sibling directory and atomically published
    only after an exact-tree validation succeeds.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    def reference(self, kind: str, identity: Mapping[str, object]) -> ArtifactRef:
        if not kind or "/" in kind or "\\" in kind or kind.startswith("."):
            raise ValueError(f"unsafe artifact kind: {kind!r}")
        artifact_id = stable_object_hash(dict(identity))
        return ArtifactRef(kind=kind, artifact_id=artifact_id, path=self.root / kind / artifact_id)

    def exists(self, ref: ArtifactRef) -> bool:
        try:
            manifest = verify_artifact(ref.path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return (
            manifest.get("artifact_id") == ref.artifact_id
            and manifest.get("kind") == ref.kind
        )

    def publish(
        self,
        ref: ArtifactRef,
        *,
        identity: Mapping[str, object],
        writer: Callable[[Path], Iterable[str]],
        force: bool = False,
    ) -> ArtifactRef:
        if self.exists(ref) and not force:
            return ref
        transaction = RunTransaction(ref.path)
        staging = transaction.begin()
        try:
            names = sorted(set(writer(staging)))
            if "manifest.json" in names:
                raise ValueError("artifact writer must not create manifest.json")
            records = manifest_records(staging, names)
            write_strict_json(
                staging / "manifest.json",
                {
                    "schema_version": "pol-artifact-manifest-v1",
                    "kind": ref.kind,
                    "artifact_id": ref.artifact_id,
                    "identity": dict(identity),
                    "files": records,
                },
            )
            expected = [*names, "manifest.json"]
            transaction.publish(lambda root: _validate_expected(root, expected, ref))
        except BaseException:
            transaction.cleanup()
            raise
        return ref


def _validate_expected(root: Path, expected: Iterable[str], ref: ArtifactRef) -> None:
    exact_artifact_tree(root, expected)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_id") != ref.artifact_id or manifest.get("kind") != ref.kind:
        raise ValueError("artifact manifest identity mismatch")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or stable_object_hash(identity) != ref.artifact_id:
        raise ValueError("artifact manifest identity hash mismatch")
    _verify_records(root, manifest)


def _verify_records(root: Path, manifest: Mapping[str, object]) -> None:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("manifest files must be a list")
    expected: list[str] = []
    actual_records = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("manifest file record must be an object")
        name = item.get("relative_path")
        if not isinstance(name, str):
            raise ValueError("manifest relative_path must be a string")
        expected.append(name)
        actual_records.extend(manifest_records(root, [name]))
    if actual_records != files:
        raise ValueError("artifact bytes do not match manifest")
    exact_artifact_tree(root, [*expected, "manifest.json"])


def verify_artifact(path: Path | str) -> dict[str, object]:
    root = Path(path).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"not a safe artifact directory: {root}")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("artifact has no regular manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "pol-artifact-manifest-v1":
        raise ValueError("unsupported artifact manifest schema")
    artifact_id = manifest.get("artifact_id")
    kind = manifest.get("kind")
    identity = manifest.get("identity")
    if not isinstance(artifact_id, str) or not isinstance(kind, str):
        raise ValueError("artifact manifest identity fields are invalid")
    if not isinstance(identity, dict):
        raise ValueError("artifact manifest identity must be an object")
    if stable_object_hash(identity) != artifact_id:
        raise ValueError("artifact identity hash does not match artifact_id")
    _verify_records(root, manifest)
    return dict(manifest)
