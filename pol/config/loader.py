from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .models import DatasetSpec, StudySpec, ValidationSpec
from .report_models import ReportSpec
from pol.digital_baselines.protocol import DigitalBaselineSpec


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
        for variant in result.get("variants", []):
            if not isinstance(variant, dict):
                continue
            source = variant.get("selection_source")
            if isinstance(source, dict) and "source_study_spec" in source:
                source["source_study_spec"] = _resolve_path(
                    source["source_study_spec"],
                    repo_root=repo_root,
                )
    elif kind == "report":
        result["output_root"] = _resolve_path(
            result.get("output_root", "outputs/reports"),
            repo_root=repo_root,
        )
        for source in result.get("sources", []):
            if isinstance(source, dict) and "study_spec" in source:
                source["study_spec"] = _resolve_path(
                    source["study_spec"],
                    repo_root=repo_root,
                )
    elif kind == "digital_baseline":
        result["output_root"] = _resolve_path(
            result.get("output_root", "outputs/digital_baselines"),
            repo_root=repo_root,
        )
        result["dataset_spec"] = _resolve_path(
            result["dataset_spec"],
            repo_root=repo_root,
        )
        comparison = result.get("physical_comparison")
        if isinstance(comparison, dict) and "source_study_spec" in comparison:
            comparison["source_study_spec"] = _resolve_path(
                comparison["source_study_spec"],
                repo_root=repo_root,
            )
    return result


def _load(model: type[T], path: str | Path, *, repo_root: Path, kind: str) -> T:
    raw = _read(path)
    schema_version = raw.get("schema_version")
    if kind == "validation" and schema_version in {
        "pol-validation-v1",
        "pol-validation-v2",
        "pol-validation-v3",
        "pol-validation-v4",
    }:
        raise ValueError(
            f"unsupported legacy validation schema {schema_version}; "
            "migrate to pol-validation-v6 with a refinement-validated "
            "target_reference specification and regenerate the certificate"
        )
    if kind == "dataset" and schema_version in {
        "pol-dataset-v1",
        "pol-dataset-v2",
    }:
        raise ValueError(
            f"unsupported legacy dataset schema {schema_version}; migrate to "
            "pol-dataset-v3 and regenerate the target-reference binding proof"
        )
    if kind == "study" and schema_version in {
        "pol-study-v1",
        "pol-study-v2",
    }:
        raise ValueError(
            f"unsupported legacy study schema {schema_version}; migrate to "
            "pol-study-v3 so completed-study selection provenance cannot be "
            "silently omitted"
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


def load_report_spec(path: str | Path, *, repo_root: Path) -> ReportSpec:
    return _load(ReportSpec, path, repo_root=repo_root, kind="report")


def load_digital_baseline_spec(
    path: str | Path,
    *,
    repo_root: Path,
) -> DigitalBaselineSpec:
    return _load(
        DigitalBaselineSpec,
        path,
        repo_root=repo_root,
        kind="digital_baseline",
    )


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
