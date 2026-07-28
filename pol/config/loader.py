from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .models import DatasetSpec, StudySpec, ValidationSpec


T = TypeVar("T", bound=BaseModel)


def _format_validation_error(exc: ValidationError) -> ValueError:
    item = exc.errors(include_url=False)[0]
    location = "$"
    for part in item.get("loc", ()):
        location += f"[{part}]" if isinstance(part, int) else f".{part}"
    return ValueError(f"invalid configuration at {location}: {item['msg']}")


def _read(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be an object: {source}")
    return raw


def _resolve_path(value: Any, *, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    # Repository-relative paths are stable under copied run specs.  A leading
    # './' is intentionally treated the same way.
    return (repo_root / path).resolve()


def _prepare_paths(
    raw: dict[str, Any], *, repo_root: Path, kind: str
) -> dict[str, Any]:
    result = copy.deepcopy(raw)
    if "artifact_root" in result:
        result["artifact_root"] = _resolve_path(
            result["artifact_root"], repo_root=repo_root
        )
    if kind == "dataset":
        result["validation_spec"] = _resolve_path(
            result["validation_spec"], repo_root=repo_root
        )
    elif kind == "study":
        result["output_root"] = _resolve_path(
            result.get("output_root", "outputs/studies"),
            repo_root=repo_root,
        )
        result["dataset_spec"] = _resolve_path(
            result["dataset_spec"], repo_root=repo_root
        )
    return result


def _load(model: type[T], path: str | Path, *, repo_root: Path, kind: str) -> T:
    raw = _read(path)
    schema_version = raw.get("schema_version")
    if kind == "validation" and schema_version == "pol-validation-v1":
        raise ValueError(
            "unsupported legacy validation schema pol-validation-v1; "
            "migrate to pol-validation-v2 and regenerate the certificate"
        )
    if kind == "dataset" and schema_version == "pol-dataset-v1":
        raise ValueError(
            "unsupported legacy dataset schema pol-dataset-v1; migrate to "
            "pol-dataset-v2 and add an explicit validated_reference or "
            "foundation_only binding"
        )
    prepared = _prepare_paths(raw, repo_root=repo_root, kind=kind)
    try:
        return model.model_validate(prepared)
    except ValidationError as exc:
        raise _format_validation_error(exc) from exc


def load_validation_spec(path: str | Path, *, repo_root: Path) -> ValidationSpec:
    return _load(ValidationSpec, path, repo_root=repo_root, kind="validation")


def load_dataset_spec(path: str | Path, *, repo_root: Path) -> DatasetSpec:
    return _load(DatasetSpec, path, repo_root=repo_root, kind="dataset")


def load_study_spec(path: str | Path, *, repo_root: Path) -> StudySpec:
    return _load(StudySpec, path, repo_root=repo_root, kind="study")


def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: Any = root
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"override path does not exist: {path}")
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise ValueError(f"override path does not exist: {path}")
    current[parts[-1]] = value


def load_study_with_overrides(
    path: str | Path,
    *,
    repo_root: Path,
    overrides: list[str],
) -> StudySpec:
    raw = _read(path)
    for item in overrides:
        if "=" not in item:
            raise ValueError("--set must use dotted.path=JSON_VALUE")
        dotted, encoded = item.split("=", 1)
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError:
            value = encoded
        _set_path(raw, dotted, value)
    prepared = _prepare_paths(raw, repo_root=repo_root, kind="study")
    try:
        return StudySpec.model_validate(prepared)
    except ValidationError as exc:
        raise _format_validation_error(exc) from exc
