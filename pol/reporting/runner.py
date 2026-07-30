from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pol.config.loader import load_study_spec
from pol.config.report_models import (
    BaselineSummaryTableSpec,
    PhaseDiagramReportSpec,
    ReportSpec,
)
from pol.plotting.reporters import MetricMapData, build_metric_map_data
from pol.runtime.artifacts import RunTransaction, manifest_records
from pol.runtime.environment import numerical_environment_fingerprint
from pol.runtime.hashing import stable_object_hash
from pol.runtime.io import file_sha256, write_csv, write_strict_json
from pol.study.selection_source import (
    VerifiedCompletedRun,
    resolve_verified_completed_run,
)


REPORT_MANIFEST_SCHEMA_VERSION = "pol-report-manifest-v1"
REPORT_IDENTITY_SCHEMA_VERSION = "pol-report-identity-v1"
REPORT_SUMMARY_SCHEMA_VERSION = "pol-report-summary-v1"
SOURCE_REFERENCES_SCHEMA_VERSION = "pol-report-source-references-v1"
PHASE_TABLE_SCHEMA_VERSION = "pol-phase-diagram-table-v1"
BASELINE_TABLE_SCHEMA_VERSION = "pol-baseline-summary-table-v1"


@dataclass(frozen=True)
class ReportResult:
    path: Path
    report_id: str
    summary: dict[str, Any]
    reused: bool


@dataclass(frozen=True)
class _ResolvedSource:
    id: str
    completed: VerifiedCompletedRun
    reference: dict[str, Any]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _source_reference(
    source_id: str,
    completed: VerifiedCompletedRun,
) -> dict[str, Any]:
    summary = json.loads(
        (completed.path / "run_summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (completed.path / "manifest.json").read_text(encoding="utf-8")
    )
    return {
        "source_id": source_id,
        "study_name": completed.identity["study"]["name"],
        "profile": completed.identity["study"]["profile"],
        "run_hash": completed.run_hash,
        "study_scientific_identity_hash": completed.scientific_identity_hash,
        "study_manifest_schema_version": manifest["schema_version"],
        "study_manifest_sha256": file_sha256(
            completed.path / "manifest.json"
        ),
        "selection_record_hash": summary["selection_record_hash"],
        "frozen_plan_hash": summary["frozen_plan_hash"],
        "dataset_artifact_id": summary["dataset_artifact_id"],
        "dataset_split_hash": completed.dataset.split_hash,
    }


def _resolve_sources(
    spec: ReportSpec,
    *,
    repo_root: Path,
) -> tuple[_ResolvedSource, ...]:
    resolved: list[_ResolvedSource] = []
    for source in spec.sources:
        if not source.study_spec.is_file():
            raise ValueError(
                f"report source study spec does not exist: {source.study_spec}"
            )
        study_spec = load_study_spec(
            source.study_spec,
            repo_root=repo_root,
        )
        if study_spec.profile != spec.profile:
            raise ValueError(
                "report/source profile mismatch: "
                f"report={spec.profile}, source={study_spec.profile}"
            )
        completed = resolve_verified_completed_run(
            study_spec,
            repo_root=repo_root,
        )
        resolved.append(
            _ResolvedSource(
                id=source.id,
                completed=completed,
                reference=_source_reference(source.id, completed),
            )
        )
    return tuple(resolved)


def _report_environment() -> dict[str, Any]:
    return {
        "schema_version": "pol-report-environment-v1",
        "numerical_environment": numerical_environment_fingerprint(),
        "matplotlib_version": str(matplotlib.__version__),
        "render_backend": str(matplotlib.get_backend()),
    }


def _resolved_report_spec(spec: ReportSpec) -> dict[str, Any]:
    return {
        "schema_version": spec.schema_version,
        "name": spec.name,
        "profile": spec.profile,
        "source_ids": [source.id for source in spec.sources],
        "reporters": [
            reporter.model_dump(mode="json") for reporter in spec.reporters
        ],
        "storage_locations_excluded": True,
    }


def _report_identity(
    spec: ReportSpec,
    sources: Iterable[_ResolvedSource],
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_IDENTITY_SCHEMA_VERSION,
        "report_spec": _resolved_report_spec(spec),
        "sources": [source.reference for source in sources],
        "software_environment": _report_environment(),
    }


def _finite_float(value: object, *, field: str) -> float:
    if value in (None, ""):
        raise ValueError(f"report source row has no {field}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"report source row has invalid {field}") from exc
    if not math.isfinite(result):
        raise ValueError(f"report source row has non-finite {field}")
    return result


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"invalid boolean table value: {value!r}")


def _axis_edges(values: tuple[float, ...], scale: str) -> np.ndarray:
    ordered = np.asarray(values, dtype=float)
    transformed = np.log(ordered) if scale == "log" else ordered
    if len(transformed) == 1:
        half_width = (
            math.log(10.0) / 2.0
            if scale == "log"
            else max(abs(float(transformed[0])) * 0.05, 0.5)
        )
        edges = np.asarray(
            [transformed[0] - half_width, transformed[0] + half_width]
        )
    else:
        inner = (transformed[:-1] + transformed[1:]) / 2.0
        edges = np.concatenate(
            (
                [transformed[0] - (inner[0] - transformed[0])],
                inner,
                [transformed[-1] + (transformed[-1] - inner[-1])],
            )
        )
    return np.exp(edges) if scale == "log" else edges


def _save_figure(
    figure: Any,
    root: Path,
    filename: str,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    names: list[str] = []
    for extension in formats:
        relative = f"figures/{filename}.{extension}"
        metadata = (
            {
                "Creator": "pol read-only report",
                "CreationDate": None,
                "ModDate": None,
            }
            if extension == "pdf"
            else {"Software": "pol read-only report"}
        )
        figure.savefig(
            root / relative,
            dpi=dpi,
            bbox_inches="tight",
            metadata=metadata,
        )
        names.append(relative)
    plt.close(figure)
    return names


def _phase_figure(
    spec: PhaseDiagramReportSpec,
    data: MetricMapData,
    *,
    source: _ResolvedSource,
    root: Path,
) -> list[str]:
    figure, axis = plt.subplots()
    x_edges = _axis_edges(data.x_values, spec.xscale)
    y_edges = _axis_edges(data.y_values, spec.yscale)
    image = axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(data.matrix),
        shading="flat",
    )
    axis.set_xscale(spec.xscale)
    axis.set_yscale(spec.yscale)
    axis.set_xticks(data.x_values)
    axis.set_yticks(data.y_values)
    axis.set_xlabel(spec.x_label)
    axis.set_ylabel(spec.y_label)
    axis.set_title(
        f"{spec.split}: {spec.variant_id}/{spec.readout_id}\n"
        f"source={source.completed.run_hash[:12]}"
    )
    if spec.mark_selected:
        for x_index, y_index in data.selected_cells:
            axis.scatter(
                [data.x_values[x_index]],
                [data.y_values[y_index]],
                marker="*",
                s=150,
                facecolors="none",
                edgecolors="white",
                linewidths=1.5,
                label="validation-selected cell",
            )
    if data.missing_cells:
        axis.scatter(
            [data.x_values[x] for x, _ in data.missing_cells],
            [data.y_values[y] for _, y in data.missing_cells],
            marker="s",
            s=70,
            facecolors="none",
            edgecolors="gray",
            label="missing verified row",
        )
    if data.invalid_cells:
        axis.scatter(
            [data.x_values[x] for x, _, _ in data.invalid_cells],
            [data.y_values[y] for _, y, _ in data.invalid_cells],
            marker="x",
            s=70,
            color="red",
            label="invalid declared cell",
        )
        for x_index, y_index, reason in data.invalid_cells:
            axis.annotate(
                reason,
                (data.x_values[x_index], data.y_values[y_index]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
                color="red",
            )
    if spec.mark_selected or data.missing_cells or data.invalid_cells:
        axis.legend(fontsize=7)
    figure.colorbar(image, ax=axis, label=spec.metric_label)
    axis.grid(True, which="both", alpha=0.2)
    return _save_figure(
        figure,
        root,
        spec.filename,
        spec.formats,
        spec.dpi,
    )


def _phase_table_rows(
    spec: PhaseDiagramReportSpec,
    data: MetricMapData,
    source_rows: list[dict[str, str]],
    *,
    source: _ResolvedSource,
) -> list[dict[str, Any]]:
    valid: dict[tuple[float, float], dict[str, str]] = {}
    for row in source_rows:
        if (
            row.get("variant_id") == spec.variant_id
            and row.get("readout_id") == spec.readout_id
            and row.get(spec.x) not in (None, "")
            and row.get(spec.y) not in (None, "")
            and row.get(spec.metric) not in (None, "")
        ):
            coordinate = (
                _finite_float(row[spec.x], field=spec.x),
                _finite_float(row[spec.y], field=spec.y),
            )
            if coordinate in valid:
                raise ValueError(
                    "phase diagram has duplicate source coordinates"
                )
            _finite_float(row[spec.metric], field=spec.metric)
            valid[coordinate] = row
    invalid = {
        (data.x_values[x], data.y_values[y]): reason
        for x, y, reason in data.invalid_cells
    }
    missing = {
        (data.x_values[x], data.y_values[y])
        for x, y in data.missing_cells
    }
    table: list[dict[str, Any]] = []
    for y in data.y_values:
        for x in data.x_values:
            coordinate = (x, y)
            source_row = valid.get(coordinate)
            if source_row is not None:
                status = "valid"
                metric_value: float | None = _finite_float(
                    source_row[spec.metric],
                    field=spec.metric,
                )
                selected = _bool(source_row.get("selected", "false"))
                reason = ""
            elif coordinate in invalid:
                status = "invalid"
                metric_value = None
                selected = False
                reason = invalid[coordinate]
            elif coordinate in missing:
                status = "missing"
                metric_value = None
                selected = False
                reason = "no verified row in the declared source table"
            else:
                raise ValueError("phase diagram cell classification mismatch")
            table.append(
                {
                    "table_schema_version": PHASE_TABLE_SCHEMA_VERSION,
                    "source_id": source.id,
                    "source_run_hash": source.completed.run_hash,
                    "source_study_scientific_identity_hash": (
                        source.completed.scientific_identity_hash
                    ),
                    "split": spec.split,
                    "metric": spec.metric,
                    "variant_id": spec.variant_id,
                    "readout_id": spec.readout_id,
                    "x_name": spec.x,
                    "x_value": x,
                    "y_name": spec.y,
                    "y_value": y,
                    "cell_status": status,
                    "metric_value": metric_value,
                    "validation_selected": selected,
                    "cell_reason": reason,
                    "candidate_id": (
                        source_row.get("candidate_id", "")
                        if source_row is not None
                        else ""
                    ),
                    "feature_system_condition_hash": (
                        source_row.get("feature_system_condition_hash", "")
                        if source_row is not None
                        else ""
                    ),
                    "selected_condition_source_kind": (
                        source_row.get(
                            "selected_condition_source_kind",
                            "",
                        )
                        if source_row is not None
                        else ""
                    ),
                    "selected_condition_source_study_run_hash": (
                        source_row.get(
                            "selected_condition_source_study_run_hash",
                            "",
                        )
                        if source_row is not None
                        else ""
                    ),
                }
            )
    return table


def _render_phase_diagram(
    spec: PhaseDiagramReportSpec,
    source: _ResolvedSource,
    root: Path,
) -> tuple[list[str], list[str], int]:
    table_name = (
        "validation_trials.csv"
        if spec.split == "validation"
        else "test_metrics.csv"
    )
    rows = _read_csv(source.completed.path / table_name)
    skipped = json.loads(
        (source.completed.path / "skipped_trials.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(skipped, list):
        raise ValueError("source skipped_trials.json must contain a list")
    data = build_metric_map_data(spec, rows, skipped_rows=skipped)
    if data is None:
        raise ValueError("phase diagram has no declared or skipped cells")
    table_rows = _phase_table_rows(
        spec,
        data,
        rows,
        source=source,
    )
    table_relative = f"machine_readable_tables/{spec.filename}.csv"
    write_csv(root / table_relative, table_rows)
    figures = _phase_figure(spec, data, source=source, root=root)
    return [table_relative], figures, len(table_rows)


def _optional_finite(row: Mapping[str, str], field: str) -> float | None:
    value = row.get(field)
    return None if value in (None, "") else _finite_float(value, field=field)


def _baseline_machine_row(
    spec: BaselineSummaryTableSpec,
    declared: Any,
    row: dict[str, str],
    *,
    source: _ResolvedSource,
) -> dict[str, Any]:
    result_kind = row.get("test_result_kind")
    readout_kind = row.get("readout_kind")
    if result_kind not in {
        "single_model",
        "independent_seed_metric_summary",
    }:
        raise ValueError(
            "baseline primary table cannot consume an ensemble/member row"
        )
    is_random = readout_kind == "random_feature_ridge"
    if is_random != (result_kind == "independent_seed_metric_summary"):
        raise ValueError(
            "random-feature baseline must use the independent-seed primary row"
        )
    field_value = _finite_float(row.get(spec.field_metric), field=spec.field_metric)
    data_value = _finite_float(row.get(spec.data_metric), field=spec.data_metric)
    result: dict[str, Any] = {
        "table_schema_version": BASELINE_TABLE_SCHEMA_VERSION,
        "row_id": declared.id,
        "label": declared.label,
        "source_id": source.id,
        "source_run_hash": source.completed.run_hash,
        "source_study_scientific_identity_hash": (
            source.completed.scientific_identity_hash
        ),
        "source_row_hash": stable_object_hash(row),
        "case_id": row.get("case_id", ""),
        "variant_id": row["variant_id"],
        "readout_id": row["readout_id"],
        "readout_kind": readout_kind,
        "primary_result_kind": result_kind,
        "prediction_ensemble_in_primary": False,
        "field_metric_name": spec.field_metric,
        "field_metric_value": field_value,
        "data_metric_name": spec.data_metric,
        "data_metric_value": data_value,
        "field_representation_floor_metric_name": (
            spec.field_representation_floor_metric
        ),
        "field_representation_floor_value": _finite_float(
            row.get(spec.field_representation_floor_metric),
            field=spec.field_representation_floor_metric,
        ),
        "data_representation_floor_metric_name": (
            spec.data_representation_floor_metric
        ),
        "data_representation_floor_value": _finite_float(
            row.get(spec.data_representation_floor_metric),
            field=spec.data_representation_floor_metric,
        ),
        "n_tar": int(row["n_tar"]),
        "n_sur": int(row["n_sur"]),
        "J": int(row["J"]),
        "q": int(row["q"]),
        "selected_ridge_zeta": _optional_finite(
            row,
            "selected_ridge_zeta",
        ),
        "selected_random_feature_width": (
            int(row["selected_random_feature_width"])
            if row.get("selected_random_feature_width") not in (None, "")
            else None
        ),
        "selected_random_feature_weight_scale": _optional_finite(
            row,
            "selected_random_feature_weight_scale",
        ),
        "selected_random_feature_bias_scale": _optional_finite(
            row,
            "selected_random_feature_bias_scale",
        ),
        "selected_condition_source_kind": row.get(
            "selected_condition_source_kind",
            "",
        ),
        "selected_condition_source_marker": row.get(
            "selected_condition_source_marker",
            "",
        ),
        "selected_condition_source_study_run_hash": row.get(
            "selected_condition_source_study_run_hash",
            "",
        ),
        "selected_condition_source_candidate_id": row.get(
            "selected_condition_source_candidate_id",
            "",
        ),
        "feature_system_condition_hash": row.get(
            "feature_system_condition_hash",
            "",
        ),
        "seed_count": None,
        "confidence_interval_method": "",
        "confidence_level": None,
        "field_seed_std": None,
        "field_seed_ci95_low": None,
        "field_seed_ci95_high": None,
        "data_seed_std": None,
        "data_seed_ci95_low": None,
        "data_seed_ci95_high": None,
    }
    if is_random:
        seed_fields = {
            "field_seed_std": f"{spec.field_metric}_seed_std",
            "field_seed_ci95_low": f"{spec.field_metric}_seed_ci95_low",
            "field_seed_ci95_high": f"{spec.field_metric}_seed_ci95_high",
            "data_seed_std": f"{spec.data_metric}_seed_std",
            "data_seed_ci95_low": f"{spec.data_metric}_seed_ci95_low",
            "data_seed_ci95_high": f"{spec.data_metric}_seed_ci95_high",
        }
        field_seed_mean = _finite_float(
            row.get(f"{spec.field_metric}_seed_mean"),
            field=f"{spec.field_metric}_seed_mean",
        )
        data_seed_mean = _finite_float(
            row.get(f"{spec.data_metric}_seed_mean"),
            field=f"{spec.data_metric}_seed_mean",
        )
        if not math.isclose(field_value, field_seed_mean, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("random-feature primary field value is not seed mean")
        if not math.isclose(data_value, data_seed_mean, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("random-feature primary data value is not seed mean")
        result.update(
            {
                "seed_count": int(row["test_seed_count"]),
                "confidence_interval_method": row[
                    "test_confidence_interval_method"
                ],
                "confidence_level": _finite_float(
                    row["test_confidence_level"],
                    field="test_confidence_level",
                ),
                **{
                    output: _finite_float(row.get(field), field=field)
                    for output, field in seed_fields.items()
                },
            }
        )
    return result


def _format_number(value: object, digits: int) -> str:
    return "" if value in (None, "") else f"{float(value):.{digits}g}"


def _metric_cell(
    row: Mapping[str, Any],
    prefix: str,
    digits: int,
) -> str:
    mean = _format_number(row[f"{prefix}_metric_value"], digits)
    standard_deviation = row.get(f"{prefix}_seed_std")
    if standard_deviation is None:
        return mean
    return (
        f"{mean} ± {_format_number(standard_deviation, digits)} "
        f"[{_format_number(row[f'{prefix}_seed_ci95_low'], digits)}, "
        f"{_format_number(row[f'{prefix}_seed_ci95_high'], digits)}]"
    )


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    digits: int,
) -> None:
    lines = [
        "| Model | Readout | Reference-field error | Finite-data error | "
        "Reference floor | Data floor | n_tar | n_sur | J | q |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = str(row["label"]).replace("|", "\\|")
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    str(row["readout_id"]),
                    _metric_cell(row, "field", digits),
                    _metric_cell(row, "data", digits),
                    _format_number(
                        row["field_representation_floor_value"],
                        digits,
                    ),
                    _format_number(
                        row["data_representation_floor_value"],
                        digits,
                    ),
                    str(row["n_tar"]),
                    str(row["n_sur"]),
                    str(row["J"]),
                    str(row["q"]),
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _write_latex(
    path: Path,
    rows: list[dict[str, Any]],
    digits: int,
) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\hline",
        (
            r"Model & Readout & Field & Data & Field floor & Data floor & "
            r"$n_{\rm tar}$ & $n_{\rm sur}$ & $J$ & $q$ \\"
        ),
        r"\hline",
    ]
    for row in rows:
        cells = (
            _latex_escape(str(row["label"])),
            _latex_escape(str(row["readout_id"])),
            _latex_escape(_metric_cell(row, "field", digits)),
            _latex_escape(_metric_cell(row, "data", digits)),
            _format_number(row["field_representation_floor_value"], digits),
            _format_number(row["data_representation_floor_value"], digits),
            str(row["n_tar"]),
            str(row["n_sur"]),
            str(row["J"]),
            str(row["q"]),
        )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend((r"\hline", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_baseline_table(
    spec: BaselineSummaryTableSpec,
    source: _ResolvedSource,
    root: Path,
) -> tuple[list[str], int]:
    source_rows = _read_csv(source.completed.path / "test_metrics.csv")
    machine_rows: list[dict[str, Any]] = []
    for declared in spec.rows:
        matches = [
            row
            for row in source_rows
            if row.get("variant_id") == declared.variant_id
            and row.get("readout_id") == declared.readout_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "baseline predeclared coordinate must match exactly one "
                f"primary row: {declared.variant_id}/{declared.readout_id}"
            )
        machine_rows.append(
            _baseline_machine_row(
                spec,
                declared,
                matches[0],
                source=source,
            )
        )
    names = [f"machine_readable_tables/{spec.filename}.csv"]
    write_csv(root / names[0], machine_rows)
    for output in spec.formatted_outputs:
        if output == "markdown":
            relative = f"formatted_tables/{spec.filename}.md"
            _write_markdown(
                root / relative,
                machine_rows,
                spec.significant_digits,
            )
        else:
            relative = f"formatted_tables/{spec.filename}.tex"
            _write_latex(
                root / relative,
                machine_rows,
                spec.significant_digits,
            )
        names.append(relative)
    return names, len(machine_rows)


def _exact_report_tree(root: Path, expected_files: Iterable[str]) -> None:
    expected = set(expected_files)
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"report artifact contains symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError(f"unsafe report artifact entry: {path}")
    if actual != expected:
        raise ValueError(
            "report artifact tree mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def verify_report(path: Path | str) -> dict[str, Any]:
    root = Path(path).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"not a safe report directory: {root}")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("report has no regular manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != REPORT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported report manifest schema")
    report_id = manifest.get("report_id")
    identity = manifest.get("identity")
    if not isinstance(report_id, str) or not isinstance(identity, dict):
        raise ValueError("report manifest identity is invalid")
    if stable_object_hash(identity) != report_id:
        raise ValueError("report identity hash mismatch")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("report manifest files must be a list")
    names: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("report manifest file record is invalid")
        name = record.get("relative_path")
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or name == "manifest.json"
        ):
            raise ValueError("unsafe report manifest relative path")
        names.append(name)
    if manifest_records(root, names) != records:
        raise ValueError("report artifact bytes do not match manifest")
    _exact_report_tree(root, [*names, "manifest.json"])
    required = {
        "resolved_report_spec.json",
        "source_references.json",
        "report_summary.json",
    }
    if not required <= set(names):
        raise ValueError("report artifact is missing required metadata")
    sources = json.loads(
        (root / "source_references.json").read_text(encoding="utf-8")
    )
    if (
        sources.get("schema_version")
        != SOURCE_REFERENCES_SCHEMA_VERSION
        or sources.get("sources") != identity.get("sources")
    ):
        raise ValueError("report source references disagree with identity")
    summary = json.loads(
        (root / "report_summary.json").read_text(encoding="utf-8")
    )
    if (
        summary.get("schema_version") != REPORT_SUMMARY_SCHEMA_VERSION
        or summary.get("status") != "complete"
        or summary.get("report_id") != report_id
        or summary.get("source_count") != len(identity["sources"])
    ):
        raise ValueError("report summary disagrees with manifest identity")
    figure_names = sorted(
        name for name in names if name.startswith("figures/")
    )
    table_names = sorted(
        name
        for name in names
        if name.startswith("machine_readable_tables/")
    )
    if (
        summary.get("figures") != figure_names
        or summary.get("machine_readable_tables") != table_names
    ):
        raise ValueError("report summary output inventory mismatch")
    return manifest


def run_report(
    spec: ReportSpec,
    *,
    repo_root: Path,
    force: bool = False,
) -> ReportResult:
    sources = _resolve_sources(spec, repo_root=repo_root)
    source_by_id = {source.id: source for source in sources}
    identity = _report_identity(spec, sources)
    report_id = stable_object_hash(identity)
    final_dir = (
        spec.output_root
        / spec.name
        / f"{spec.profile}-{report_id[:12]}"
    )
    if final_dir.is_dir() and not force:
        manifest = verify_report(final_dir)
        if manifest.get("identity") != identity:
            raise ValueError("existing report identity mismatch")
        summary = json.loads(
            (final_dir / "report_summary.json").read_text(encoding="utf-8")
        )
        return ReportResult(
            path=final_dir,
            report_id=report_id,
            summary=summary,
            reused=True,
        )

    transaction = RunTransaction(final_dir)
    staging = transaction.begin()
    try:
        (staging / "figures").mkdir()
        (staging / "machine_readable_tables").mkdir()
        (staging / "formatted_tables").mkdir()
        resolved_spec = _resolved_report_spec(spec)
        write_strict_json(
            staging / "resolved_report_spec.json",
            resolved_spec,
        )
        write_strict_json(
            staging / "source_references.json",
            {
                "schema_version": SOURCE_REFERENCES_SCHEMA_VERSION,
                "sources": [source.reference for source in sources],
            },
        )
        generated: list[str] = [
            "resolved_report_spec.json",
            "source_references.json",
        ]
        figures: list[str] = []
        machine_tables: list[str] = []
        table_row_counts: dict[str, int] = {}
        for reporter in spec.reporters:
            source = source_by_id[reporter.source_id]
            if isinstance(reporter, PhaseDiagramReportSpec):
                tables, rendered, row_count = _render_phase_diagram(
                    reporter,
                    source,
                    staging,
                )
                generated.extend(tables)
                generated.extend(rendered)
                machine_tables.extend(tables)
                figures.extend(rendered)
                table_row_counts[reporter.filename] = row_count
            elif isinstance(reporter, BaselineSummaryTableSpec):
                tables, row_count = _render_baseline_table(
                    reporter,
                    source,
                    staging,
                )
                generated.extend(tables)
                machine_tables.append(tables[0])
                table_row_counts[reporter.filename] = row_count
            else:
                raise AssertionError("unreachable report kind")
        summary = {
            "schema_version": REPORT_SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "report_id": report_id,
            "name": spec.name,
            "profile": spec.profile,
            "source_count": len(sources),
            "source_run_hashes": [
                source.completed.run_hash for source in sources
            ],
            "reporter_count": len(spec.reporters),
            "table_row_counts": table_row_counts,
            "machine_readable_tables": sorted(machine_tables),
            "figures": sorted(figures),
            "source_runs_verified_before_reporting": True,
            "upstream_study_execution": False,
            "feature_solve": False,
            "readout_fit": False,
            "test_inference": False,
            "storage_locations_excluded_from_identity": True,
        }
        write_strict_json(staging / "report_summary.json", summary)
        generated.append("report_summary.json")
        records = manifest_records(staging, generated)
        write_strict_json(
            staging / "manifest.json",
            {
                "schema_version": REPORT_MANIFEST_SCHEMA_VERSION,
                "report_id": report_id,
                "identity": identity,
                "files": records,
            },
        )
        transaction.publish(lambda root: verify_report(root))
    except BaseException:
        transaction.cleanup()
        raise
    return ReportResult(
        path=final_dir,
        report_id=report_id,
        summary=summary,
        reused=False,
    )


__all__ = [
    "BASELINE_TABLE_SCHEMA_VERSION",
    "PHASE_TABLE_SCHEMA_VERSION",
    "REPORT_MANIFEST_SCHEMA_VERSION",
    "ReportResult",
    "run_report",
    "verify_report",
]
