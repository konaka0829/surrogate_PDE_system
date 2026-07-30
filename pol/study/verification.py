from __future__ import annotations

from importlib import import_module
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from pol.config.models import TrialSpec
from pol.learning.direct import verify_fixed_fourier_decoder_diagnostic
from pol.runtime.artifacts import manifest_records
from pol.runtime.device import require_cpu_tensors, verify_execution_device_policy
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from pol.runtime.io import file_sha256
from .diagnostics import (
    HEAT_MULTIPLIER_COEFFICIENT_SCHEMA_VERSION,
    HEAT_MULTIPLIER_SUMMARY_SCHEMA_VERSION,
    READOUT_STABILITY_MODEL_SCHEMA_VERSION,
    READOUT_STABILITY_REPEAT_SCHEMA_VERSION,
    READOUT_STABILITY_SUMMARY_SCHEMA_VERSION,
    summarize_repeated_metrics,
)
from .evaluation import (
    READOUT_SELECTION_FIELDS,
    feature_system_condition_hash,
    random_feature_member_parameter_hash,
    random_feature_member_result_fields,
    selected_readout_parameter_fields,
    summarize_independent_seed_metrics,
)
from .protocol import (
    assert_selection_record_safe,
    test_evaluation_contract,
    verify_frozen_decoder_bindings,
    verify_no_decoder_diagnostic,
    verify_representative_feature_bindings,
    verify_selection_source_provenance_bindings,
)
from .prediction_capture import (
    PREDICTION_CAPTURE_FILENAME,
    load_prediction_capture,
    verify_prediction_capture_payload,
)
from .results import (
    RESULT_ROW_SCHEMA_VERSION,
    SELECTION_SOURCE_RESULT_FIELDS,
    build_selected_comparison_rows,
    load_rows,
    selection_source_result_fields,
)
from .training_subsets import (
    TRAINING_SUBSET_POLICY,
    TRAINING_SUBSET_SCHEMA_VERSION,
    training_subset_result_fields,
)


def _row_binding(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (
        row.get("case_id"),
        row.get("readout_id"),
        row.get("candidate_id"),
    )


def _has_csv_value(row: Mapping[str, Any], key: str) -> bool:
    return key in row and row.get(key) not in ("", None)


def _require_close(actual: Any, expected: float, *, label: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{label} does not match the expected value")


def _csv_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{label} is not a boolean")


def _require_csv_equivalent(
    actual: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    if isinstance(expected, bool):
        if _csv_bool(actual, label=label) is not expected:
            raise ValueError(f"{label} does not match the expected value")
        return
    if isinstance(expected, int):
        try:
            value = int(actual)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} is not an integer") from exc
        if value != expected:
            raise ValueError(f"{label} does not match the expected value")
        return
    if isinstance(expected, float):
        _require_close(actual, expected, label=label)
        return
    if actual != expected:
        raise ValueError(f"{label} does not match the expected value")


def _verify_result_row_contract(
    row: Mapping[str, Any],
    *,
    table: str,
    provenance_by_variant: Mapping[str, Mapping[str, Any]],
    model: Mapping[str, Any] | None = None,
    expected_schema: str = RESULT_ROW_SCHEMA_VERSION,
) -> None:
    if row.get("result_row_schema_version") != expected_schema:
        raise ValueError(f"{table} row has an unsupported result-row schema")
    try:
        n_tar = int(row["n_tar"])
        n_sur = int(row["n_sur"])
        J = int(row["J"])
        q = int(row["q"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{table} row has invalid dimensions") from exc
    if J > n_sur or q > n_tar or q <= 0 or q % 2 == 0:
        raise ValueError(f"{table} row violates the dimension contract")
    try:
        system_parameters = json.loads(row["feature_system_parameters"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{table} row has invalid feature-system parameters"
        ) from exc
    if not isinstance(system_parameters, Mapping):
        raise ValueError(
            f"{table} row feature-system parameters are not an object"
        )
    system_hash = stable_object_hash(dict(system_parameters))
    if row.get("feature_system_condition_hash") != system_hash:
        raise ValueError(f"{table} row feature-system hash mismatch")
    expected_dynamic = row.get("feature_family") != "static_input"
    if _csv_bool(
        row.get("feature_is_dynamic"),
        label=f"{table} row feature_is_dynamic",
    ) is not expected_dynamic:
        raise ValueError(f"{table} row has inconsistent feature semantics")

    variant_id = row.get("variant_id")
    provenance = provenance_by_variant.get(str(variant_id))
    expected_source = selection_source_result_fields(
        provenance,
        feature_family=str(row.get("feature_family")),
    )
    for field in SELECTION_SOURCE_RESULT_FIELDS:
        if field in expected_source:
            _require_csv_equivalent(
                row.get(field),
                expected_source[field],
                label=f"{table} row {field}",
            )
        elif _has_csv_value(row, field):
            raise ValueError(
                f"{table} row has false selection-source field {field}"
            )
    expected_source_system_hash = expected_source.get(
        "selected_condition_source_feature_system_hash"
    )
    if expected_source_system_hash is not None and (
        row.get("feature_system_condition_hash")
        != expected_source_system_hash
    ):
        raise ValueError(
            f"{table} row feature system differs from its selected source"
        )

    kind = row.get("readout_kind")
    if kind not in {
        "direct_fourier_decoder",
        "affine_ridge",
        "random_feature_ridge",
    }:
        raise ValueError(f"{table} row has an unknown readout kind")
    if model is not None:
        if kind != model.get("kind"):
            raise ValueError(f"{table} row readout kind mismatch")
        expected_settings = selected_readout_parameter_fields(model)
        for field in READOUT_SELECTION_FIELDS:
            if field in expected_settings:
                _require_csv_equivalent(
                    row.get(field),
                    expected_settings[field],
                    label=f"{table} row {field}",
                )
            elif _has_csv_value(row, field):
                raise ValueError(
                    f"{table} row has false selected setting {field}"
                )
    elif kind == "direct_fourier_decoder":
        if any(_has_csv_value(row, field) for field in READOUT_SELECTION_FIELDS):
            raise ValueError(
                f"{table} direct row has false learned-readout settings"
            )
    elif not _has_csv_value(row, "selected_ridge_zeta"):
        raise ValueError(f"{table} learned row has no selected ridge setting")

    metric_prefix = (
        "validation"
        if table == "validation"
        else "test_ensemble"
        if table == "random-feature ensemble"
        else "test"
    )
    required_metrics = (
        f"{metric_prefix}_field_relative_l2_mean",
        f"{metric_prefix}_representation_floor_relative_l2_mean",
        f"{metric_prefix}_data_representation_floor_relative_l2_mean",
    )
    for field in required_metrics:
        if not _has_csv_value(row, field):
            raise ValueError(f"{table} row is missing required metric {field}")
        try:
            value = float(row[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{table} row metric {field} is not numeric"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"{table} row metric {field} is not finite"
            )


def _nested_mapping_value(
    payload: Mapping[str, Any],
    path: str,
) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"resolved study has no declared path {path}")
        value = value[part]
    return value


def _declared_varying_paths(
    resolved_study: Mapping[str, Any],
) -> set[str]:
    varying: set[str] = set()
    global_axes = resolved_study.get("global_axes", [])
    variants = resolved_study.get("variants", [])
    if not isinstance(global_axes, list) or not isinstance(variants, list):
        raise ValueError("resolved study has invalid declared axes")
    for axis in global_axes:
        if not isinstance(axis, Mapping) or not isinstance(
            axis.get("path"), str
        ):
            raise ValueError("resolved study global axis is malformed")
        varying.add(str(axis["path"]))
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ValueError("resolved study variant is malformed")
        overrides = variant.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("resolved study variant overrides are malformed")
        varying.update(str(path) for path in overrides)
        search = variant.get("search", {})
        if not isinstance(search, Mapping):
            raise ValueError("resolved study variant search is malformed")
        axes = search.get("axes", [])
        if not isinstance(axes, list):
            raise ValueError("resolved study search axes are malformed")
        for axis in axes:
            if not isinstance(axis, Mapping) or not isinstance(
                axis.get("path"), str
            ):
                raise ValueError("resolved study search axis is malformed")
            varying.add(str(axis["path"]))
    return varying


def _verify_fixed_dimension_rows(
    resolved_study: Mapping[str, Any],
    tables: Mapping[str, list[dict[str, Any]]],
) -> None:
    """Bind row dimensions not declared as axes to the configured base trial."""
    base_trial = resolved_study.get("base_trial")
    if not isinstance(base_trial, Mapping):
        raise ValueError("resolved study base trial is missing")
    varying = _declared_varying_paths(resolved_study)
    dimensions = {
        "input.n_tar": "n_tar",
        "feature.n_sur": "n_sur",
        "feature.observation.J": "J",
        "output.q": "q",
    }
    for path, field in dimensions.items():
        if path in varying:
            continue
        expected = _nested_mapping_value(base_trial, path)
        for table, rows in tables.items():
            for row in rows:
                _require_csv_equivalent(
                    row.get(field),
                    expected,
                    label=f"{table} row fixed {field}",
                )


def _variant_specs_by_id(
    resolved_study: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    variants = resolved_study.get("variants")
    if not isinstance(variants, list):
        raise ValueError("resolved study variants are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ValueError("resolved study variant is not an object")
        variant_id = variant.get("id")
        if not isinstance(variant_id, str) or variant_id in result:
            raise ValueError("resolved study variant ids are invalid")
        result[variant_id] = variant
    return result


def _expected_grid_axis_values(
    search: Mapping[str, Any],
) -> list[dict[str, Any]]:
    axes = search.get("axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("grid search axes are missing")
    paths: list[str] = []
    values: list[list[Any]] = []
    for axis in axes:
        if not isinstance(axis, Mapping):
            raise ValueError("grid search axis is not an object")
        path = axis.get("path")
        axis_values = axis.get("values")
        if not isinstance(path, str) or not isinstance(axis_values, list):
            raise ValueError("grid search axis is malformed")
        paths.append(path)
        values.append(axis_values)
    return [
        dict(zip(paths, combination, strict=True))
        for combination in itertools.product(*values)
    ]


def _verify_selection_search_contract(
    *,
    resolved_study: Mapping[str, Any],
    selection: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    selection_cases = selection.get("cases")
    plan_cases = plan.get("cases")
    if not isinstance(selection_cases, Mapping) or not isinstance(
        plan_cases,
        Mapping,
    ):
        raise ValueError("selection/frozen plan search cases are missing")
    if set(selection_cases) != set(plan_cases):
        raise ValueError("selection/frozen plan search case sets differ")
    variants = _variant_specs_by_id(resolved_study)
    binding_fields = (
        "selected_by_readout",
        "search_kind",
        "declared_candidate_count",
        "planned_cartesian_cell_count",
        "evaluated_candidate_count",
        "skipped_candidate_count",
        "candidate_order",
        "selection_order_by_readout",
        "representative_readout",
        "representative_candidate_id",
        "representative_feature_condition",
        "training_subsets_by_readout",
    )
    for case_id, selection_case in selection_cases.items():
        if not isinstance(selection_case, Mapping):
            raise ValueError("selection search case is not an object")
        plan_case = plan_cases[case_id]
        if not isinstance(plan_case, Mapping):
            raise ValueError("frozen-plan search case is not an object")
        if any(
            selection_case.get(field) != plan_case.get(field)
            for field in binding_fields
        ):
            raise ValueError(
                "selection/frozen plan representative/search binding mismatch"
            )
        candidate_order = selection_case.get("candidate_order")
        if (
            not isinstance(candidate_order, list)
            or len(candidate_order) != len(set(candidate_order))
        ):
            raise ValueError("candidate order must be a unique list")
        selected_by_readout = selection_case.get("selected_by_readout")
        selection_order_by_readout = selection_case.get(
            "selection_order_by_readout"
        )
        if not isinstance(selected_by_readout, Mapping) or any(
            candidate_id not in candidate_order
            for candidate_id in selected_by_readout.values()
        ):
            raise ValueError("selected candidates are outside candidate order")
        if (
            not isinstance(selection_order_by_readout, Mapping)
            or set(selection_order_by_readout) != set(selected_by_readout)
            or any(
                not isinstance(readout_order, list)
                or not readout_order
                or len(readout_order) != len(set(readout_order))
                or any(
                    candidate_id not in candidate_order
                    for candidate_id in readout_order
                )
                or selected_by_readout[readout_id] not in readout_order
                for readout_id, readout_order in (
                    selection_order_by_readout.items()
                )
            )
        ):
            raise ValueError("per-readout selection order is invalid")
        representative_readout = selection_case.get(
            "representative_readout"
        )
        representative_id = selection_case.get(
            "representative_candidate_id"
        )
        if selected_by_readout.get(representative_readout) != representative_id:
            raise ValueError("representative candidate is not readout-selected")
        condition = selection_case.get("representative_feature_condition")
        if (
            not isinstance(condition, Mapping)
            or condition.get("selection_split") != "validation"
            or condition.get("selection_metric")
            != selection.get("selection_metric")
            or condition.get("representative_readout")
            != representative_readout
            or condition.get("candidate_id") != representative_id
        ):
            raise ValueError("representative feature condition is malformed")
        variant_id = selection_case.get("variant_id")
        variant = variants.get(str(variant_id))
        if variant is None:
            raise ValueError("selection case refers to an unknown variant")
        search = variant.get("search")
        if not isinstance(search, Mapping):
            raise ValueError("variant search is missing")
        search_kind = search.get("kind")
        if selection_case.get("search_kind") != search_kind:
            raise ValueError("selection search kind differs from study")
        grid_cells = selection_case.get("grid_cells")
        skipped_candidates = selection_case.get("skipped_candidates")
        if not isinstance(grid_cells, list) or not isinstance(
            skipped_candidates,
            list,
        ):
            raise ValueError("selection search evidence is missing")
        if search_kind == "grid":
            expected_axes = _expected_grid_axis_values(search)
            planned_count = len(expected_axes)
            if (
                selection_case.get("declared_candidate_count")
                != planned_count
                or selection_case.get("planned_cartesian_cell_count")
                != planned_count
                or len(grid_cells) != planned_count
            ):
                raise ValueError("planned Cartesian cell count mismatch")
            if [cell.get("cell_index") for cell in grid_cells] != list(
                range(planned_count)
            ):
                raise ValueError("grid cell indices do not cover the plan")
            if [cell.get("axis_values") for cell in grid_cells] != expected_axes:
                raise ValueError("grid cells differ from the declared product")
            evaluated_cells = [
                cell for cell in grid_cells
                if cell.get("status") == "evaluated"
            ]
            skipped_cells = [
                cell for cell in grid_cells
                if cell.get("status") == "skipped"
            ]
            if len(evaluated_cells) + len(skipped_cells) != planned_count:
                raise ValueError("grid cell status is unsupported")
            if any(
                cell.get("candidate_id") is not None
                or not isinstance(cell.get("reason"), str)
                or not cell.get("reason")
                for cell in skipped_cells
            ):
                raise ValueError("skipped grid cell has no explicit reason")
            if [cell.get("candidate_id") for cell in evaluated_cells] != (
                candidate_order
            ):
                raise ValueError("grid candidate order differs from config order")
            if any(
                readout_order != candidate_order
                for readout_order in selection_order_by_readout.values()
            ):
                raise ValueError(
                    "grid per-readout order differs from config order"
                )
            if (
                selection_case.get("evaluated_candidate_count")
                != len(evaluated_cells)
                or selection_case.get("skipped_candidate_count")
                != len(skipped_cells)
                or len(skipped_candidates) != len(skipped_cells)
            ):
                raise ValueError("grid evaluated/skipped counts mismatch")
            expected_skipped = [
                {
                    "stage": f"grid:{cell['cell_index']}",
                    "overrides": cell["axis_values"],
                    "reason": cell["reason"],
                }
                for cell in skipped_cells
            ]
            if skipped_candidates != expected_skipped:
                raise ValueError("skipped grid reasons differ from cell evidence")
        else:
            if (
                selection_case.get("planned_cartesian_cell_count") is not None
                or grid_cells
            ):
                raise ValueError(
                    "non-grid search falsely claims Cartesian grid evidence"
                )


def _verify_validation_selection_order(
    *,
    resolved_study: Mapping[str, Any],
    selection: Mapping[str, Any],
    validation_rows: list[dict[str, Any]],
) -> None:
    selection_spec = resolved_study.get("selection")
    cases = selection.get("cases")
    if not isinstance(selection_spec, Mapping) or not isinstance(cases, Mapping):
        raise ValueError("selection-order contract is missing")
    metric = selection_spec.get("metric")
    tolerance = float(selection_spec.get("tie_tolerance", 0.0))
    for case_id, case in cases.items():
        if not isinstance(case, Mapping):
            raise ValueError("selection-order case is not an object")
        order = case.get("candidate_order")
        per_readout_order = case.get("selection_order_by_readout")
        selected = case.get("selected_by_readout")
        if (
            not isinstance(order, list)
            or not isinstance(per_readout_order, Mapping)
            or not isinstance(selected, Mapping)
        ):
            raise ValueError("selection-order evidence is missing")
        for readout_id, selected_id in selected.items():
            active_order = per_readout_order[readout_id]
            relevant_rows = [
                row
                for row in validation_rows
                if row.get("case_id") == case_id
                and row.get("readout_id") == readout_id
            ]
            by_candidate = {
                row.get("candidate_id"): row
                for row in relevant_rows
            }
            if (
                len(relevant_rows) != len(order)
                or set(by_candidate) != set(order)
            ):
                raise ValueError(
                    "validation rows do not match candidate order"
                )
            best = min(
                float(by_candidate[item][metric]) for item in active_order
            )
            expected = next(
                item
                for item in active_order
                if float(by_candidate[item][metric]) <= best + tolerance
            )
            if selected_id != expected:
                raise ValueError(
                    "selected candidate violates validation config-order tie rule"
                )


def _verify_heat_multiplier_diagnostics(
    root: Path,
    *,
    resolved_study: Mapping[str, Any],
    expected_bindings: set[tuple[Any, Any, Any]],
    entry_by_binding: Mapping[tuple[Any, Any, Any], Mapping[str, Any]],
    model_by_binding: Mapping[tuple[Any, Any, Any], Mapping[str, Any]],
    dataset_binding_proof: Mapping[str, Any],
    coefficient_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    diagnostics = resolved_study.get("diagnostics", [])
    heat_configured = any(
        isinstance(item, Mapping) and item.get("kind") == "heat_multiplier"
        for item in diagnostics
    )
    coefficient_path = root / "heat_multiplier.csv"
    summary_path = root / "heat_multiplier_summary.csv"
    if not heat_configured:
        if coefficient_path.exists() or summary_path.exists():
            raise ValueError(
                "study without heat-multiplier diagnostic published heat tables"
            )
        if coefficient_rows or summary_rows:
            raise ValueError("unexpected heat-multiplier diagnostic rows")
        return
    if not summary_path.is_file():
        raise ValueError("heat-multiplier summary table is missing")

    summary_by_binding: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for row in summary_rows:
        if row.get("schema_version") != (
            HEAT_MULTIPLIER_SUMMARY_SCHEMA_VERSION
        ):
            raise ValueError("unsupported heat-multiplier summary schema")
        binding = _row_binding(row)
        if binding in summary_by_binding:
            raise ValueError("duplicate heat-multiplier summary binding")
        summary_by_binding[binding] = row
    if set(summary_by_binding) != expected_bindings:
        raise ValueError(
            "heat-multiplier summaries do not match frozen candidates"
        )

    coefficients_by_binding: dict[
        tuple[Any, Any, Any],
        list[dict[str, Any]],
    ] = {}
    for row in coefficient_rows:
        if row.get("schema_version") != (
            HEAT_MULTIPLIER_COEFFICIENT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported heat-multiplier coefficient schema")
        binding = _row_binding(row)
        if binding not in expected_bindings:
            raise ValueError(
                "heat-multiplier coefficient row is not frozen-selected"
            )
        coefficients_by_binding.setdefault(binding, []).append(row)

    dataset_condition = dataset_binding_proof.get("dataset_condition")
    if not isinstance(dataset_condition, Mapping):
        raise ValueError("dataset binding has no diagnostic target condition")
    target = dataset_condition.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("dataset binding has no diagnostic target evolution")
    target_system = target.get("system")
    if not isinstance(target_system, Mapping):
        raise ValueError("dataset binding has no diagnostic target system")
    target_is_heat = target_system.get("kind") == "heat"
    domain_length = float(dataset_condition["domain_length"])

    for binding in expected_bindings:
        summary_row = summary_by_binding[binding]
        entry = entry_by_binding[binding]
        model = model_by_binding[binding]
        trial = TrialSpec.model_validate(entry["trial"])
        q = int(trial.output.q)
        J = int(trial.feature.observation.J)
        rows = coefficients_by_binding.get(binding, [])
        if summary_row.get("variant_id") != entry.get("variant_id"):
            raise ValueError("heat-multiplier variant binding mismatch")
        if summary_row.get("readout_kind") != model.get("kind"):
            raise ValueError("heat-multiplier readout-kind binding mismatch")
        if int(summary_row.get("q", -1)) != q:
            raise ValueError("heat-multiplier summary q mismatch")
        if int(summary_row.get("J", -1)) != J:
            raise ValueError("heat-multiplier summary J mismatch")

        expected_zeta = model.get("zeta")
        stored_zeta = summary_row.get("selected_zeta")
        if expected_zeta is None:
            if stored_zeta not in ("", None):
                raise ValueError("heat-multiplier summary has a false zeta")
        else:
            _require_close(
                stored_zeta,
                float(expected_zeta),
                label="heat-multiplier selected zeta",
            )

        evolution = trial.feature.evolution
        feature_is_heat = (
            trial.feature.kind == "pde_dynamics"
            and evolution is not None
            and evolution.system.kind == "heat"
        )
        linear_readout = model.get("kind") in {
            "direct_fourier_decoder",
            "affine_ridge",
        }
        applicable = target_is_heat and feature_is_heat and linear_readout
        if _csv_bool(
            summary_row.get("applicable"),
            label="heat-multiplier applicable",
        ) is not applicable:
            raise ValueError("heat-multiplier applicability mismatch")
        if not applicable:
            if rows:
                raise ValueError(
                    "non-applicable heat-multiplier summary has coefficient rows"
                )
            if int(summary_row.get("coefficient_row_count", -1)) != 0:
                raise ValueError(
                    "non-applicable heat-multiplier coefficient count is nonzero"
                )
            if (
                target_is_heat
                and feature_is_heat
                and model.get("kind") == "random_feature_ridge"
                and (
                    summary_row.get("diagnostic_status")
                    != "not_applicable_nonlinear_readout"
                )
            ):
                raise ValueError(
                    "random-feature multiplier diagnostic is not explicitly N/A"
                )
            continue

        if len(rows) != q or int(
            summary_row.get("coefficient_row_count", -1)
        ) != q:
            raise ValueError(
                "heat-multiplier coefficient rows do not cover frozen q"
            )
        indices = sorted(int(row["coefficient_index"]) for row in rows)
        if indices != list(range(q)):
            raise ValueError(
                "heat-multiplier coefficient indices are not canonical"
            )
        ordered = sorted(rows, key=lambda row: int(row["coefficient_index"]))
        identifiable_modes: set[int] = set()
        diagonal_errors: list[float] = []
        off_diagonal_squared = 0.0
        amplification_magnitudes: list[float] = []
        for index, row in enumerate(ordered):
            if int(row.get("q", -1)) != q or int(row.get("J", -1)) != J:
                raise ValueError("heat-multiplier coefficient J/q mismatch")
            expected_mode = 0 if index == 0 else (index + 1) // 2
            expected_kind = (
                "dc"
                if index == 0
                else ("cosine" if index % 2 else "sine")
            )
            if int(row.get("mode_index", -1)) != expected_mode:
                raise ValueError("heat-multiplier mode index mismatch")
            if row.get("coefficient_kind") != expected_kind:
                raise ValueError("heat-multiplier coefficient kind mismatch")
            if row.get("variant_id") != entry.get("variant_id"):
                raise ValueError(
                    "heat-multiplier coefficient variant mismatch"
                )
            for physical_field in (
                "target_nu",
                "target_time",
                "target_diffusion_time",
                "surrogate_nu",
                "surrogate_time",
                "surrogate_diffusion_time",
                "diffusion_condition",
                "inverse_amplification_required",
                "inverse_condition_note",
                "domain_length",
            ):
                if row.get(physical_field) != summary_row.get(
                    physical_field
                ):
                    raise ValueError(
                        "heat-multiplier coefficient physical condition "
                        f"mismatch: {physical_field}"
                    )
            identifiable = _csv_bool(
                row.get("identifiable"),
                label="heat-multiplier identifiable",
            )
            if identifiable:
                identifiable_modes.add(expected_mode)
                try:
                    diagonal_error = float(row["absolute_diagonal_error"])
                    ideal = float(row["ideal_readout_multiplier"])
                    diagonal = float(row["effective_learned_diagonal"])
                    amplification = _csv_bool(
                        row.get("amplification"),
                        label="heat-multiplier amplification",
                    )
                    amplification_magnitude = float(
                        row["amplification_magnitude"]
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "identifiable heat-multiplier row is incomplete"
                    ) from exc
                _require_close(
                    diagonal_error,
                    abs(diagonal - ideal),
                    label="heat-multiplier absolute diagonal error",
                )
                expected_relative_error = diagonal_error / abs(ideal)
                if math.isfinite(expected_relative_error):
                    _require_close(
                        row.get("relative_diagonal_error"),
                        expected_relative_error,
                        label="heat-multiplier relative diagonal error",
                    )
                    if row.get("relative_diagonal_error_status") != "available":
                        raise ValueError(
                            "heat-multiplier relative-error status mismatch"
                        )
                elif (
                    row.get("relative_diagonal_error") not in ("", None)
                    or row.get("relative_diagonal_error_status") != "overflow"
                ):
                    raise ValueError(
                        "heat-multiplier relative-error overflow mismatch"
                    )
                if amplification is not (ideal > 1.0):
                    raise ValueError(
                        "heat-multiplier amplification flag mismatch"
                    )
                _require_close(
                    amplification_magnitude,
                    max(1.0, abs(ideal)),
                    label="heat-multiplier amplification magnitude",
                )
                diagonal_errors.append(diagonal_error)
                amplification_magnitudes.append(amplification_magnitude)
            off_diagonal = float(row["off_diagonal_l2_contribution"])
            if not math.isfinite(off_diagonal) or off_diagonal < 0:
                raise ValueError(
                    "heat-multiplier off-diagonal contribution is invalid"
                )
            off_diagonal_squared += off_diagonal * off_diagonal

        if int(summary_row.get("identifiable_mode_count", -1)) != len(
            identifiable_modes
        ):
            raise ValueError("heat-multiplier identifiable-mode count mismatch")
        if int(summary_row.get("identifiable_coefficient_count", -1)) != len(
            diagonal_errors
        ):
            raise ValueError(
                "heat-multiplier identifiable-coefficient count mismatch"
            )
        if not diagonal_errors:
            raise ValueError(
                "applicable heat-multiplier diagnostic has no identifiable row"
            )
        _require_close(
            summary_row.get("diagonal_rmse"),
            math.sqrt(
                math.fsum(value * value for value in diagonal_errors)
                / len(diagonal_errors)
            ),
            label="heat-multiplier diagonal RMSE",
        )
        _require_close(
            summary_row.get("diagonal_max_error"),
            max(diagonal_errors),
            label="heat-multiplier diagonal maximum error",
        )
        _require_close(
            summary_row.get("off_diagonal_frobenius_norm"),
            math.sqrt(off_diagonal_squared),
            label="heat-multiplier off-diagonal Frobenius norm",
        )
        _require_close(
            summary_row.get("max_ideal_amplification"),
            max(amplification_magnitudes),
            label="heat-multiplier maximum ideal amplification",
        )

        if feature_is_heat and evolution is not None:
            target_nu = float(target_system["nu"])
            target_time = float(target["time"])
            surrogate_nu = float(evolution.system.nu)
            surrogate_time = float(evolution.time)
            physical_values = {
                "target_nu": target_nu,
                "target_time": target_time,
                "target_diffusion_time": target_nu * target_time,
                "surrogate_nu": surrogate_nu,
                "surrogate_time": surrogate_time,
                "surrogate_diffusion_time": surrogate_nu * surrogate_time,
                "domain_length": domain_length,
            }
            for field, expected_value in physical_values.items():
                _require_close(
                    summary_row.get(field),
                    expected_value,
                    label=f"heat-multiplier physical condition {field}",
                )
            target_diffusion_time = target_nu * target_time
            surrogate_diffusion_time = surrogate_nu * surrogate_time
            if math.isclose(
                target_diffusion_time,
                surrogate_diffusion_time,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                diffusion_condition = "matched"
            elif surrogate_diffusion_time < target_diffusion_time:
                diffusion_condition = "under_diffusive"
            else:
                diffusion_condition = "more_diffusive"
            if summary_row.get("diffusion_condition") != diffusion_condition:
                raise ValueError(
                    "heat-multiplier diffusion-condition mismatch"
                )
            amplification_required = (
                surrogate_diffusion_time > target_diffusion_time
            )
            if _csv_bool(
                summary_row.get("inverse_amplification_required"),
                label="heat-multiplier inverse amplification",
            ) is not amplification_required:
                raise ValueError(
                    "heat-multiplier inverse-amplification condition mismatch"
                )


_PREDICTION_METRIC_FIELDS = (
    "coefficient_mse",
    "coefficient_relative_l2_mean",
    "coefficient_relative_l2_median",
    "coefficient_relative_l2_max",
    "field_absolute_l2_mean",
    "field_absolute_l2_median",
    "field_absolute_l2_max",
    "field_relative_l2_mean",
    "field_relative_l2_median",
    "field_relative_l2_max",
    "data_field_absolute_l2_mean",
    "data_field_absolute_l2_median",
    "data_field_absolute_l2_max",
    "data_field_relative_l2_mean",
    "data_field_relative_l2_median",
    "data_field_relative_l2_max",
)


def _optional_csv_int(value: Any, *, label: str) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an integer") from exc


def _metric_values(
    row: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in _PREDICTION_METRIC_FIELDS:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} has an invalid {field}") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} has a non-finite or negative {field}")
        result[field] = value
    return result


def _verify_repeated_summary(
    row: Mapping[str, Any],
    metrics: list[dict[str, float]],
    *,
    dimension: str,
    label: str,
) -> None:
    expected = summarize_repeated_metrics(metrics, dimension=dimension)
    for field, value in expected.items():
        _require_close(row.get(field), value, label=f"{label} {field}")
    if int(row.get("repeat_count", -1)) < 2:
        raise ValueError(f"{label} has an invalid repeat count")
    _require_close(
        row.get("confidence_level"),
        0.95,
        label=f"{label} confidence level",
    )
    if row.get("confidence_interval_method") != "student_t":
        raise ValueError(f"{label} confidence interval is not Student-t")


def _verify_training_subset_record(
    record: Mapping[str, Any],
    *,
    split_binding: Mapping[str, Any],
    label: str,
) -> None:
    expected_fields = {
        "schema_version",
        "kind",
        "policy",
        "policy_version",
        "n_train",
        "subset_ids",
        "subset_ids_hash",
        "parent_train_ids_hash",
        "parent_train_count",
        "validation_ids_hash",
        "training_subset_hash",
    }
    if set(record) != expected_fields:
        raise ValueError(f"{label} has unknown or missing fields")
    if (
        record.get("schema_version") != TRAINING_SUBSET_SCHEMA_VERSION
        or record.get("kind") != "nested_train_prefix"
        or record.get("policy") != TRAINING_SUBSET_POLICY
        or record.get("policy_version") != 1
    ):
        raise ValueError(f"{label} policy/schema mismatch")
    ids = record.get("subset_ids")
    try:
        n_train = int(record["n_train"])
        parent_count = int(record["parent_train_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} has invalid counts") from exc
    if (
        not isinstance(ids, list)
        or n_train < 1
        or len(ids) != n_train
        or n_train > parent_count
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in ids
        )
        or len(ids) != len(set(ids))
    ):
        raise ValueError(f"{label} has invalid prefix IDs")
    ids_tensor = torch.tensor(ids, dtype=torch.long)
    if tensor_sha256(ids_tensor) != record.get("subset_ids_hash"):
        raise ValueError(f"{label} subset-ID hash mismatch")
    if record.get("parent_train_ids_hash") != split_binding.get(
        "train_ids_hash"
    ):
        raise ValueError(f"{label} parent-train hash mismatch")
    if record.get("validation_ids_hash") != split_binding.get(
        "validation_ids_hash"
    ):
        raise ValueError(f"{label} validation hash mismatch")
    unsigned = dict(record)
    stored_hash = unsigned.pop("training_subset_hash")
    if stable_object_hash(unsigned) != stored_hash:
        raise ValueError(f"{label} content hash mismatch")


def _verify_training_subset_contract(
    *,
    resolved_study: Mapping[str, Any],
    selection: Mapping[str, Any],
    plan: Mapping[str, Any],
    models: Mapping[str, Any],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    ensemble_rows: list[dict[str, Any]],
) -> None:
    split_binding = selection.get("split_binding")
    selection_cases = selection.get("cases")
    plan_cases = plan.get("cases")
    if (
        not isinstance(split_binding, Mapping)
        or not isinstance(selection_cases, Mapping)
        or not isinstance(plan_cases, Mapping)
    ):
        raise ValueError("training-subset split/case binding is missing")
    records_by_binding: dict[
        tuple[Any, Any, Any],
        Mapping[str, Any],
    ] = {}
    case_records: dict[str, Mapping[str, Any]] = {}
    for case_id, case in selection_cases.items():
        if not isinstance(case, Mapping):
            raise ValueError("training-subset selection case is invalid")
        selected = case.get("selected_by_readout")
        subsets = case.get("training_subsets_by_readout")
        plan_case = plan_cases.get(case_id)
        if (
            not isinstance(selected, Mapping)
            or not isinstance(subsets, Mapping)
            or not isinstance(plan_case, Mapping)
            or plan_case.get("training_subsets_by_readout") != subsets
            or set(subsets) != set(selected)
        ):
            raise ValueError("training-subset freeze binding mismatch")
        unique_case_records: dict[str, Mapping[str, Any]] = {}
        for readout_id, candidate_id in selected.items():
            record = subsets[readout_id]
            if not isinstance(record, Mapping):
                raise ValueError("training-subset record is not an object")
            _verify_training_subset_record(
                record,
                split_binding=split_binding,
                label="selection training subset",
            )
            records_by_binding[
                (case_id, readout_id, candidate_id)
            ] = record
            unique_case_records[str(record["training_subset_hash"])] = record
        if len(unique_case_records) != 1:
            raise ValueError(
                "readouts within one case use different training subsets"
            )
        case_records[str(case_id)] = next(iter(unique_case_records.values()))

    model_keys_by_binding = {
        (
            entry.get("case_id"),
            entry.get("readout_id"),
            entry.get("candidate_id"),
        ): (model_key, entry)
        for model_key, entry in models.items()
        if isinstance(entry, Mapping)
    }
    if set(model_keys_by_binding) != set(records_by_binding):
        raise ValueError("training subsets do not cover frozen models")
    for binding, record in records_by_binding.items():
        _, entry = model_keys_by_binding[binding]
        if entry.get("training_subset") != record:
            raise ValueError("frozen model training-subset binding mismatch")

    for table, rows in (
        ("validation", validation_rows),
        ("test", test_rows),
        ("random-feature seed", seed_rows),
        ("random-feature ensemble", ensemble_rows),
    ):
        for row in rows:
            binding = _row_binding(row)
            record = records_by_binding.get(binding)
            if record is None:
                if table == "validation" and str(
                    row.get("selected", "")
                ).lower() != "true":
                    continue
                raise ValueError(f"{table} row has no training-subset binding")
            for field, expected in training_subset_result_fields(record).items():
                _require_csv_equivalent(
                    row.get(field),
                    expected,
                    label=f"{table} row {field}",
                )

    learning = resolved_study.get("learning_curve")
    if learning is None:
        return
    if (
        not isinstance(learning, Mapping)
        or learning.get("kind") != "learning_curve"
        or learning.get("subset_policy") != TRAINING_SUBSET_POLICY
    ):
        raise ValueError("learning-curve contract is invalid")
    by_variant: dict[str, list[Mapping[str, Any]]] = {}
    for case_id, case in selection_cases.items():
        by_variant.setdefault(str(case["variant_id"]), []).append(
            case_records[str(case_id)]
        )
    for records in by_variant.values():
        ordered = sorted(records, key=lambda item: int(item["n_train"]))
        sizes = [int(item["n_train"]) for item in ordered]
        if len(sizes) != len(set(sizes)):
            raise ValueError("learning curve repeats a training size")
        for smaller, larger in zip(ordered, ordered[1:]):
            smaller_ids = list(smaller["subset_ids"])
            larger_ids = list(larger["subset_ids"])
            if larger_ids[: len(smaller_ids)] != smaller_ids:
                raise ValueError("learning-curve subsets are not nested prefixes")

    for table, rows in (("validation", validation_rows), ("test", test_rows)):
        groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
        for row in rows:
            if row.get("readout_kind") != "direct_fourier_decoder":
                continue
            groups.setdefault(
                (
                    row.get("variant_id"),
                    row.get("readout_id"),
                    row.get("feature_system_condition_hash"),
                ),
                [],
            ).append(row)
        for group in groups.values():
            selected_group = [
                row
                for row in group
                if table != "validation"
                or str(row.get("selected", "")).lower() == "true"
            ]
            if len(selected_group) < 2:
                continue
            cache_field = (
                "selection_feature_cache_id"
                if table == "validation"
                else "feature_cache_id"
            )
            if len({row.get(cache_field) for row in selected_group}) != 1:
                raise ValueError(
                    "learning-curve feature cache was not reused across sizes"
                )
            if table == "test":
                reference = selected_group[0]
                metric_fields = [
                    field
                    for field, value in reference.items()
                    if field.startswith("test_")
                    and field != "test_result_kind"
                    and value not in ("", None)
                ]
                for row in selected_group[1:]:
                    for field in metric_fields:
                        _require_close(
                            row.get(field),
                            float(reference[field]),
                            label=f"direct learning-curve invariant {field}",
                        )


def _verify_readout_stability_diagnostics(
    *,
    resolved_study: Mapping[str, Any],
    models: Mapping[str, Any],
    selection_hash: str,
    frozen_plan_hash: str,
    model_rows: list[dict[str, Any]],
    repeat_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    ensemble_repeat_rows: list[dict[str, Any]],
    ensemble_summary_rows: list[dict[str, Any]],
) -> None:
    configured = [
        item
        for item in resolved_study.get("diagnostics", [])
        if isinstance(item, Mapping)
        and item.get("kind") == "readout_stability_noise"
    ]
    all_rows = (
        model_rows
        + repeat_rows
        + summary_rows
        + ensemble_repeat_rows
        + ensemble_summary_rows
    )
    if not configured:
        if all_rows:
            raise ValueError(
                "study without readout-stability diagnostic published "
                "readout-stability tables"
            )
        return
    if len(configured) != 1:
        raise ValueError("study must configure at most one stability diagnostic")
    diagnostic = configured[0]
    try:
        levels = [float(value) for value in diagnostic["levels"]]
        repeats = int(diagnostic["repeats"])
        noise_seed_base = int(diagnostic["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid readout-stability diagnostic config") from exc
    if not model_rows or not repeat_rows or not summary_rows:
        raise ValueError("readout-stability diagnostic tables are incomplete")

    def entry_for(row: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
        model_key = row.get("model_key")
        entry = models.get(model_key)
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label} does not bind to a frozen model")
        expected_binding = (
            entry.get("case_id"),
            entry.get("readout_id"),
            entry.get("candidate_id"),
        )
        if _row_binding(row) != expected_binding:
            raise ValueError(f"{label} frozen-model binding mismatch")
        if row.get("variant_id") != entry.get("variant_id"):
            raise ValueError(f"{label} variant binding mismatch")
        model = entry.get("model")
        if (
            not isinstance(model, Mapping)
            or row.get("readout_kind") != model.get("kind")
        ):
            raise ValueError(f"{label} readout-kind binding mismatch")
        if row.get("selection_record_hash") != selection_hash:
            raise ValueError(f"{label} selection-record binding mismatch")
        if row.get("frozen_plan_hash") != frozen_plan_hash:
            raise ValueError(f"{label} frozen-plan binding mismatch")
        if row.get("diagnostic_kind") != "readout_stability_noise":
            raise ValueError(f"{label} diagnostic kind mismatch")
        if (
            row.get("noise_scaling_kind")
            != "relative_global_feature_rms"
        ):
            raise ValueError(f"{label} noise-scaling contract mismatch")
        if _csv_bool(
            row.get("common_random_numbers"),
            label=f"{label} common-random-numbers flag",
        ) is not True:
            raise ValueError(f"{label} does not use common random numbers")
        try:
            feature_rms = float(row["feature_rms"])
            shape = json.loads(str(row["sample_shape"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} feature metadata is invalid") from exc
        if (
            not math.isfinite(feature_rms)
            or feature_rms <= 0.0
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(not isinstance(value, int) or value < 1 for value in shape)
        ):
            raise ValueError(f"{label} feature metadata is invalid")
        return entry

    expected_model_members: dict[str, dict[int | None, Mapping[str, Any]]] = {}
    for model_key, entry in models.items():
        if not isinstance(entry, Mapping):
            raise ValueError("frozen model entry is invalid")
        model = entry.get("model")
        if not isinstance(model, Mapping):
            raise ValueError("frozen model payload is invalid")
        if model.get("kind") == "random_feature_ridge":
            members = model.get("members")
            if not isinstance(members, list):
                raise ValueError("frozen random-feature members are invalid")
            expected_model_members[str(model_key)] = {
                int(member["seed"]): member for member in members
            }
        else:
            expected_model_members[str(model_key)] = {None: model}

    actual_model_members: dict[str, dict[int | None, dict[str, Any]]] = {}
    covariance_fields = (
        "covariance_singular_values",
        "covariance_rank",
        "covariance_dimension",
        "covariance_rank_cutoff",
        "covariance_raw_condition",
        "covariance_retained_rank_condition",
    )
    for row in model_rows:
        if row.get("schema_version") != READOUT_STABILITY_MODEL_SCHEMA_VERSION:
            raise ValueError("readout-stability model schema mismatch")
        entry = entry_for(row, label="readout-stability model row")
        model_key = str(row["model_key"])
        seed = _optional_csv_int(
            row.get("seed"),
            label="readout-stability model seed",
        )
        member = expected_model_members.get(model_key, {}).get(seed)
        if not isinstance(member, Mapping):
            raise ValueError(
                "readout-stability model row has an unknown member seed"
            )
        bucket = actual_model_members.setdefault(model_key, {})
        if seed in bucket:
            raise ValueError("duplicate readout-stability model row")
        bucket[seed] = row
        model = entry["model"]
        if model.get("kind") == "direct_fourier_decoder":
            if row.get("norm_status") != "not_applicable_fixed_decoder":
                raise ValueError("fixed decoder has a learned-model norm")
            for field in (
                "weight_frobenius_norm",
                "weight_operator_norm",
                "bias_norm",
                "selected_ridge_zeta",
            ):
                if _has_csv_value(row, field):
                    raise ValueError("fixed decoder has false norm metadata")
        else:
            if row.get("norm_status") != "available":
                raise ValueError("learned readout is missing norm metadata")
            W = member.get("W")
            b = member.get("b")
            if not isinstance(W, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise ValueError("frozen learned readout has invalid tensors")
            for field, value in {
                "weight_frobenius_norm": float(
                    torch.linalg.matrix_norm(W, ord="fro")
                ),
                "weight_operator_norm": float(
                    torch.linalg.matrix_norm(W, ord=2)
                ),
                "bias_norm": float(torch.linalg.vector_norm(b)),
                "selected_ridge_zeta": float(member["zeta"]),
            }.items():
                _require_close(
                    row.get(field),
                    value,
                    label=f"readout-stability model {field}",
                )
        if model.get("kind") == "random_feature_ridge":
            expected_map_hash = stable_object_hash(
                {
                    "seed": seed,
                    "A_sha256": tensor_sha256(member["A"]),
                    "c_sha256": tensor_sha256(member["c"]),
                    "activation": model["activation"],
                    "weight_scale": model["weight_scale"],
                    "bias_scale": model["bias_scale"],
                }
            )
            if row.get("random_map_parameter_hash") != expected_map_hash:
                raise ValueError("random-feature map hash mismatch")
        elif _has_csv_value(row, "random_map_parameter_hash"):
            raise ValueError("deterministic model has random-map metadata")
        for prefix in ("base_feature_", "readout_design_"):
            try:
                singular = json.loads(
                    str(row[prefix + "covariance_singular_values"])
                )
                rank = int(row[prefix + "covariance_rank"])
                dimension = int(row[prefix + "covariance_dimension"])
                cutoff = float(row[prefix + "covariance_rank_cutoff"])
                raw_condition = float(
                    row[prefix + "covariance_raw_condition"]
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise ValueError("invalid covariance diagnostics") from exc
            if (
                not isinstance(singular, list)
                or len(singular) != dimension
                or not 0 <= rank <= dimension
                or not math.isfinite(cutoff)
                or cutoff < 0.0
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    for value in singular
                )
                or (not math.isfinite(raw_condition) and not math.isinf(raw_condition))
            ):
                raise ValueError("invalid covariance diagnostics")
            retained = row.get(prefix + "covariance_retained_rank_condition")
            if retained not in ("", None):
                retained_value = float(retained)
                if not math.isfinite(retained_value) or retained_value < 1.0:
                    raise ValueError("invalid retained-rank condition number")
            for field in covariance_fields:
                if prefix + field not in row:
                    raise ValueError("incomplete covariance diagnostics")
    if set(actual_model_members) != set(expected_model_members) or any(
        set(actual_model_members.get(key, {})) != set(members)
        for key, members in expected_model_members.items()
    ):
        raise ValueError(
            "readout-stability model rows do not match frozen models"
        )

    repeats_by_member_level: dict[
        tuple[str, int | None, float],
        list[dict[str, Any]],
    ] = {}
    seen_repeats: set[tuple[str, int | None, float, int]] = set()
    for row in repeat_rows:
        if row.get("schema_version") != READOUT_STABILITY_REPEAT_SCHEMA_VERSION:
            raise ValueError("readout-stability repeat schema mismatch")
        entry_for(row, label="readout-stability repeat row")
        model_key = str(row["model_key"])
        seed = _optional_csv_int(
            row.get("seed"),
            label="readout-stability repeat seed",
        )
        if seed not in expected_model_members.get(model_key, {}):
            raise ValueError("readout-stability repeat has unknown member seed")
        try:
            level = float(row["noise_level"])
            repeat = int(row["repeat"])
            row_noise_seed = int(row["noise_seed"])
            feature_rms = float(row["feature_rms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("readout-stability repeat metadata is invalid") from exc
        if level not in levels or not 0 <= repeat < repeats:
            raise ValueError("readout-stability repeat coordinate is invalid")
        expected_noise_seed = (
            noise_seed_base + 1009 * levels.index(level) + repeat
        )
        if row_noise_seed != expected_noise_seed:
            raise ValueError("readout-stability common noise seed mismatch")
        _require_close(
            row.get("noise_rms"),
            level * feature_rms,
            label="readout-stability noise RMS",
        )
        expected_kind = (
            "independent_seed_realization"
            if seed is not None
            else "single_model"
        )
        if row.get("result_kind") != expected_kind:
            raise ValueError("readout-stability repeat result kind mismatch")
        _metric_values(row, label="readout-stability repeat row")
        coordinate = (model_key, seed, level, repeat)
        if coordinate in seen_repeats:
            raise ValueError("duplicate readout-stability repeat")
        seen_repeats.add(coordinate)
        repeats_by_member_level.setdefault(
            (model_key, seed, level),
            [],
        ).append(row)
    expected_repeat_coordinates = {
        (model_key, seed, level, repeat)
        for model_key, members in expected_model_members.items()
        for seed in members
        for level in levels
        for repeat in range(repeats)
    }
    if seen_repeats != expected_repeat_coordinates:
        raise ValueError(
            "readout-stability repeats do not cover the configured design"
        )

    summary_by_coordinate: dict[
        tuple[str, int | None, float, str],
        dict[str, Any],
    ] = {}
    for row in summary_rows:
        if row.get("schema_version") != READOUT_STABILITY_SUMMARY_SCHEMA_VERSION:
            raise ValueError("readout-stability summary schema mismatch")
        entry_for(row, label="readout-stability summary row")
        model_key = str(row["model_key"])
        seed = _optional_csv_int(
            row.get("seed"),
            label="readout-stability summary seed",
        )
        try:
            level = float(row["noise_level"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("readout-stability summary level is invalid") from exc
        kind = str(row.get("result_kind"))
        coordinate = (model_key, seed, level, kind)
        if coordinate in summary_by_coordinate:
            raise ValueError("duplicate readout-stability summary")
        summary_by_coordinate[coordinate] = row
    for model_key, members in expected_model_members.items():
        random_model = all(seed is not None for seed in members)
        for level in levels:
            seed_means: list[dict[str, float]] = []
            for seed in members:
                kind = (
                    "independent_seed_repeat_summary"
                    if seed is not None
                    else "single_model_repeat_summary"
                )
                row = summary_by_coordinate.pop(
                    (model_key, seed, level, kind),
                    None,
                )
                if row is None:
                    raise ValueError("missing per-model repeat summary")
                source = sorted(
                    repeats_by_member_level[(model_key, seed, level)],
                    key=lambda item: int(item["repeat"]),
                )
                metrics = [
                    _metric_values(
                        item,
                        label="readout-stability repeat row",
                    )
                    for item in source
                ]
                if int(row.get("repeat_count", -1)) != repeats:
                    raise ValueError("repeat-summary count mismatch")
                _verify_repeated_summary(
                    row,
                    metrics,
                    dimension="repeat",
                    label="readout-stability repeat summary",
                )
                seed_means.append(
                    {
                        field: math.fsum(
                            metric[field] for metric in metrics
                        )
                        / repeats
                        for field in _PREDICTION_METRIC_FIELDS
                    }
                )
            if random_model:
                row = summary_by_coordinate.pop(
                    (
                        model_key,
                        None,
                        level,
                        "independent_seed_primary_summary",
                    ),
                    None,
                )
                if row is None:
                    raise ValueError("missing independent-seed primary summary")
                if int(row.get("seed_count", -1)) != len(members):
                    raise ValueError("independent-seed summary count mismatch")
                if int(row.get("repeat_count", -1)) != repeats:
                    raise ValueError("independent-seed repeat count mismatch")
                _verify_repeated_summary(
                    row,
                    seed_means,
                    dimension="seed",
                    label="independent-seed primary summary",
                )
    if summary_by_coordinate:
        raise ValueError("unexpected readout-stability summary rows")

    include_ensemble = bool(
        diagnostic.get("include_prediction_ensemble", True)
    )
    random_model_keys = {
        key
        for key, members in expected_model_members.items()
        if all(seed is not None for seed in members)
    }
    expected_ensemble_coordinates = (
        {
            (model_key, level, repeat)
            for model_key in random_model_keys
            for level in levels
            for repeat in range(repeats)
        }
        if include_ensemble
        else set()
    )
    ensemble_by_level: dict[
        tuple[str, float],
        list[dict[str, Any]],
    ] = {}
    actual_ensemble_coordinates: set[tuple[str, float, int]] = set()
    for row in ensemble_repeat_rows:
        if row.get("schema_version") != READOUT_STABILITY_REPEAT_SCHEMA_VERSION:
            raise ValueError("stability ensemble-repeat schema mismatch")
        entry_for(row, label="stability ensemble-repeat row")
        model_key = str(row["model_key"])
        try:
            level = float(row["noise_level"])
            repeat = int(row["repeat"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid stability ensemble coordinate") from exc
        coordinate = (model_key, level, repeat)
        if coordinate in actual_ensemble_coordinates:
            raise ValueError("duplicate stability ensemble repeat")
        actual_ensemble_coordinates.add(coordinate)
        if row.get("result_kind") != "noise_prediction_ensemble":
            raise ValueError("stability ensemble result kind mismatch")
        if int(row.get("ensemble_member_count", -1)) != len(
            expected_model_members.get(model_key, {})
        ):
            raise ValueError("stability ensemble member count mismatch")
        expected_noise_seed = (
            noise_seed_base + 1009 * levels.index(level) + repeat
            if level in levels and 0 <= repeat < repeats
            else None
        )
        if expected_noise_seed is None or int(row["noise_seed"]) != expected_noise_seed:
            raise ValueError("stability ensemble common noise seed mismatch")
        _metric_values(row, label="stability ensemble-repeat row")
        ensemble_by_level.setdefault((model_key, level), []).append(row)
    if actual_ensemble_coordinates != expected_ensemble_coordinates:
        raise ValueError(
            "stability ensemble repeats do not cover the configured design"
        )
    actual_ensemble_summaries: set[tuple[str, float]] = set()
    for row in ensemble_summary_rows:
        if row.get("schema_version") != READOUT_STABILITY_SUMMARY_SCHEMA_VERSION:
            raise ValueError("stability ensemble-summary schema mismatch")
        entry_for(row, label="stability ensemble-summary row")
        model_key = str(row["model_key"])
        try:
            level = float(row["noise_level"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid stability ensemble-summary level") from exc
        coordinate = (model_key, level)
        if coordinate in actual_ensemble_summaries:
            raise ValueError("duplicate stability ensemble summary")
        actual_ensemble_summaries.add(coordinate)
        if row.get("result_kind") != "noise_prediction_ensemble_summary":
            raise ValueError("stability ensemble-summary kind mismatch")
        source = sorted(
            ensemble_by_level.get(coordinate, []),
            key=lambda item: int(item["repeat"]),
        )
        if len(source) != repeats or int(row.get("repeat_count", -1)) != repeats:
            raise ValueError("stability ensemble-summary count mismatch")
        _verify_repeated_summary(
            row,
            [
                _metric_values(item, label="stability ensemble-repeat row")
                for item in source
            ],
            dimension="repeat",
            label="stability ensemble summary",
        )
    expected_ensemble_summaries = {
        (model_key, level)
        for model_key in random_model_keys
        for level in levels
    } if include_ensemble else set()
    if actual_ensemble_summaries != expected_ensemble_summaries:
        raise ValueError(
            "stability ensemble summaries do not cover the configured design"
        )


def _verify_prediction_capture_artifact(
    root: Path,
    *,
    resolved_study: Mapping[str, Any],
    summary: Mapping[str, Any],
    dataset_reference: Mapping[str, Any],
    selection_hash: str,
    frozen_plan_hash: str,
    frozen_model_archive_sha256: str,
    models: Mapping[str, Any],
    test_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    ensemble_rows: list[dict[str, Any]],
) -> None:
    capture_spec = resolved_study.get("prediction_capture")
    capture_path = root / PREDICTION_CAPTURE_FILENAME
    if capture_spec is None:
        if capture_path.exists():
            raise ValueError(
                "study without prediction capture published a capture artifact"
            )
        if (
            summary.get("prediction_capture_status") != "not_configured"
            or summary.get("prediction_capture_file") is not None
            or summary.get("prediction_capture_entry_count") != 0
            or summary.get("prediction_capture_content_hash") is not None
        ):
            raise ValueError("run summary has false prediction-capture metadata")
        return
    if not isinstance(capture_spec, Mapping) or not capture_path.is_file():
        raise ValueError("configured prediction capture artifact is missing")
    payload = load_prediction_capture(capture_path)
    verify_prediction_capture_payload(
        payload,
        capture_spec=capture_spec,
        dataset_artifact_id=str(dataset_reference["artifact_id"]),
        dataset_split_hash=str(dataset_reference["split_hash"]),
        selection_record_hash=selection_hash,
        frozen_plan_hash=frozen_plan_hash,
        frozen_model_archive_sha256=frozen_model_archive_sha256,
    )
    entries = payload["entries"]
    if (
        summary.get("prediction_capture_status") != "complete"
        or summary.get("prediction_capture_file")
        != PREDICTION_CAPTURE_FILENAME
        or int(summary.get("prediction_capture_entry_count", -1))
        != len(entries)
        or summary.get("prediction_capture_content_hash")
        != payload["capture_content_hash"]
        or summary.get("prediction_capture_spectrum_storage")
        != (
            "predeclared_samples_plus_all_test_per_coefficient_aggregates"
        )
    ):
        raise ValueError("run summary prediction-capture binding mismatch")

    readout_ids = set(capture_spec["readout_ids"])
    capture_seeds = [int(value) for value in (
        capture_spec["random_feature_members"]["seeds"]
    )]
    include_ensemble = bool(capture_spec["include_ensemble"])
    expected_coordinates: set[tuple[str, str, int | None]] = set()
    for model_key, archive_entry in models.items():
        if archive_entry["readout_id"] not in readout_ids:
            continue
        model = archive_entry["model"]
        if model.get("kind") == "random_feature_ridge":
            expected_coordinates.update(
                (
                    str(model_key),
                    "independent_seed_realization",
                    seed,
                )
                for seed in capture_seeds
            )
            if include_ensemble:
                expected_coordinates.add(
                    (str(model_key), "prediction_ensemble", None)
                )
        else:
            expected_coordinates.add((str(model_key), "single_model", None))
    actual_coordinates: set[tuple[str, str, int | None]] = set()
    primary_by_binding = {_row_binding(row): row for row in test_rows}
    seed_by_binding_seed = {
        (*_row_binding(row), int(row["seed"])): row for row in seed_rows
    }
    ensemble_by_binding = {
        _row_binding(row): row for row in ensemble_rows
    }
    for entry in entries:
        model_key = str(entry["model_key"])
        semantics = str(entry["prediction_semantics"])
        seed = None if entry.get("seed") is None else int(entry["seed"])
        coordinate = (model_key, semantics, seed)
        if coordinate in actual_coordinates:
            raise ValueError("duplicate prediction capture coordinate")
        actual_coordinates.add(coordinate)
        archive_entry = models.get(model_key)
        if not isinstance(archive_entry, Mapping):
            raise ValueError("prediction capture references an unknown model")
        binding = (
            entry.get("case_id"),
            entry.get("readout_id"),
            entry.get("candidate_id"),
        )
        expected_binding = (
            archive_entry["case_id"],
            archive_entry["readout_id"],
            archive_entry["candidate_id"],
        )
        trial = TrialSpec.model_validate(archive_entry["trial"])
        model = archive_entry["model"]
        if (
            binding != expected_binding
            or entry.get("variant_id") != archive_entry["variant_id"]
            or entry.get("readout_kind") != model["kind"]
            or entry.get("feature_condition")
            != trial.feature.model_dump(mode="json")
            or entry.get("feature_system_condition_hash")
            != feature_system_condition_hash(trial)
            or int(entry["n_tar"]) != int(trial.input.n_tar)
            or int(entry["q"]) != int(trial.output.q)
        ):
            raise ValueError("prediction capture frozen-model binding mismatch")
        if semantics == "single_model":
            metric_row = primary_by_binding[binding]
            metric_field = "test_coefficient_mse"
        elif semantics == "independent_seed_realization":
            metric_row = seed_by_binding_seed[(*binding, int(seed))]
            metric_field = "test_coefficient_mse"
            member = next(
                (
                    item
                    for item in model["members"]
                    if int(item["seed"]) == int(seed)
                ),
                None,
            )
            if (
                not isinstance(member, Mapping)
                or entry.get("frozen_member_parameter_hash")
                != random_feature_member_parameter_hash(model, member)
            ):
                raise ValueError(
                    "prediction capture random-feature member mismatch"
                )
        elif semantics == "prediction_ensemble":
            metric_row = ensemble_by_binding[binding]
            metric_field = "test_ensemble_coefficient_mse"
        else:
            raise ValueError("unknown prediction capture semantics")
        _require_close(
            entry.get("test_coefficient_mse"),
            float(metric_row[metric_field]),
            label="prediction capture canonical coefficient metric",
        )
    if actual_coordinates != expected_coordinates:
        raise ValueError(
            "prediction capture does not cover the predeclared models/seeds"
        )


def _verify_report_publication(
    root: Path,
    *,
    resolved_study: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    if summary.get("numerical_publication_status") != (
        "complete_verified_before_reporting"
    ):
        raise ValueError("numerical publication status is invalid")
    figures = summary.get("figures")
    if not isinstance(figures, list) or any(
        not isinstance(value, str) or Path(value).name != value
        for value in figures
    ):
        raise ValueError("run summary figure list is invalid")
    figure_root = root / "figures"
    actual = (
        sorted(
            path.relative_to(figure_root).as_posix()
            for path in figure_root.rglob("*")
            if path.is_file()
        )
        if figure_root.is_dir()
        else []
    )
    if actual != sorted(figures):
        raise ValueError("reported figures do not match completed artifacts")
    generate_plots = bool(
        resolved_study.get("execution", {}).get("generate_plots", True)
    )
    status = summary.get("report_status")
    if status == "complete":
        if (
            not generate_plots
            or summary.get("report_source")
            != "verified_completed_run_read_only"
        ):
            raise ValueError("completed report has invalid source semantics")
    elif status == "numerical_complete_report_not_generated":
        if not generate_plots or figures or summary.get("report_source") is not None:
            raise ValueError("pending report state is invalid")
    elif status == "not_requested":
        if generate_plots or figures or summary.get("report_source") is not None:
            raise ValueError("unrequested report state is invalid")
    else:
        raise ValueError("unknown report publication status")


def _verify_study_semantics(root: Path, manifest: Mapping[str, Any]) -> None:
    manifest_schema = manifest.get("schema_version")
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("study-run identity must be an object")
    expected_identity_schema = {
        "pol-study-run-manifest-v8": "pol-study-run-identity-v8",
        "pol-study-run-manifest-v9": "pol-study-run-identity-v8",
        "pol-study-run-manifest-v10": "pol-study-run-identity-v9",
        "pol-study-run-manifest-v11": "pol-study-run-identity-v10",
        "pol-study-run-manifest-v12": "pol-study-run-identity-v11",
        "pol-study-run-manifest-v13": "pol-study-run-identity-v12",
        "pol-study-run-manifest-v14": "pol-study-run-identity-v13",
    }[str(manifest_schema)]
    if identity.get("schema_version") != expected_identity_schema:
        raise ValueError("unsupported legacy study-run identity")
    verify_execution_device_policy(
        identity,
        boundary="study-run identity",
    )
    environment = identity.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("study-run numerical environment is missing")
    verify_execution_device_policy(
        environment,
        boundary="study-run numerical environment",
    )
    run_hash = stable_object_hash(dict(identity))
    resolved_study = json.loads(
        (root / "resolved_study.json").read_text(encoding="utf-8")
    )
    if resolved_study != identity.get("study"):
        raise ValueError("resolved study does not match manifest identity")

    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    expected_summary_schema = {
        "pol-study-run-manifest-v8": "pol-study-run-summary-v8",
        "pol-study-run-manifest-v9": "pol-study-run-summary-v9",
        "pol-study-run-manifest-v10": "pol-study-run-summary-v10",
        "pol-study-run-manifest-v11": "pol-study-run-summary-v11",
        "pol-study-run-manifest-v12": "pol-study-run-summary-v12",
        "pol-study-run-manifest-v13": "pol-study-run-summary-v13",
        "pol-study-run-manifest-v14": "pol-study-run-summary-v14",
    }[str(manifest_schema)]
    if summary.get("schema_version") != expected_summary_schema:
        raise ValueError("unsupported study-run summary schema")
    verify_execution_device_policy(
        summary,
        boundary="study-run summary",
    )
    if summary.get("run_hash") != run_hash:
        raise ValueError("study-run summary hash does not match manifest identity")
    if summary.get("study") != resolved_study.get("name"):
        raise ValueError("study-run summary name mismatch")
    if summary.get("profile") != resolved_study.get("profile"):
        raise ValueError("study-run summary profile mismatch")
    if manifest_schema in {
        "pol-study-run-manifest-v9",
        "pol-study-run-manifest-v10",
        "pol-study-run-manifest-v11",
        "pol-study-run-manifest-v12",
        "pol-study-run-manifest-v13",
        "pol-study-run-manifest-v14",
    }:
        global_axes = resolved_study.get("global_axes", [])
        variants = resolved_study.get("variants", [])
        if not isinstance(global_axes, list) or not isinstance(variants, list):
            raise ValueError("resolved study has invalid global-axis planning")
        planned_combinations = 1
        for axis in global_axes:
            if not isinstance(axis, Mapping):
                raise ValueError("resolved study global axis is invalid")
            values = axis.get("values")
            if not isinstance(values, list):
                raise ValueError("resolved study global-axis values are invalid")
            planned_combinations *= len(values)
        planned_cases = planned_combinations * len(variants)
        if summary.get("planned_global_axis_combination_count") != (
            planned_combinations
        ):
            raise ValueError("study summary global-axis combination mismatch")
        if summary.get("planned_global_axis_case_count") != planned_cases:
            raise ValueError("study summary planned global-axis case mismatch")
        if summary.get("evaluated_global_axis_case_count") != summary.get(
            "case_count"
        ):
            raise ValueError("study summary evaluated global-axis case mismatch")
        if summary.get("skipped_global_axis_case_count") != (
            planned_cases - int(summary.get("case_count", -1))
        ):
            raise ValueError("study summary skipped global-axis case mismatch")

    selection = json.loads(
        (root / "selection_record.json").read_text(encoding="utf-8")
    )
    expected_selection_schema = (
        "pol-selection-record-v8"
        if manifest_schema in {
            "pol-study-run-manifest-v12",
            "pol-study-run-manifest-v13",
            "pol-study-run-manifest-v14",
        }
        else "pol-selection-record-v7"
    )
    if selection.get("schema_version") != expected_selection_schema:
        raise ValueError("unsupported selection-record schema")
    verify_execution_device_policy(
        selection,
        boundary="selection record",
    )
    assert_selection_record_safe(selection)
    if selection.get("test_data_used") is not False:
        raise ValueError("selection record is not test-isolated")
    selection_hash = stable_object_hash(selection)
    if summary.get("selection_record_hash") != selection_hash:
        raise ValueError("selection-record hash mismatch")

    plan = json.loads(
        (root / "frozen_evaluation_plan.json").read_text(encoding="utf-8")
    )
    expected_plan_schema = {
        "pol-study-run-manifest-v12": "pol-frozen-evaluation-plan-v8",
        "pol-study-run-manifest-v13": "pol-frozen-evaluation-plan-v9",
        "pol-study-run-manifest-v14": "pol-frozen-evaluation-plan-v9",
    }.get(str(manifest_schema), "pol-frozen-evaluation-plan-v7")
    if plan.get("schema_version") != expected_plan_schema:
        raise ValueError("unsupported frozen evaluation plan schema")
    verify_execution_device_policy(
        plan,
        boundary="frozen evaluation plan",
    )
    stored_plan_hash = plan.pop("plan_content_hash", None)
    computed_plan_hash = stable_object_hash(plan)
    plan["plan_content_hash"] = stored_plan_hash
    if stored_plan_hash != computed_plan_hash:
        raise ValueError("frozen evaluation plan content hash mismatch")
    if plan.get("test_data_used") is not False:
        raise ValueError("frozen evaluation plan is not test-isolated")
    if plan.get("selection_record_hash") != selection_hash:
        raise ValueError("frozen plan selection binding mismatch")
    if summary.get("frozen_plan_hash") != stored_plan_hash:
        raise ValueError("run summary frozen-plan hash mismatch")
    legacy_test_contract = {
        "schema_version": "pol-test-evaluation-contract-v1",
        "random_feature_primary": "independent_seed_metric_summary",
        "random_feature_seed_result": "independent_seed_realization",
        "random_feature_ensemble_result": "prediction_ensemble",
        "seed_standard_deviation_ddof": 1,
        "confidence_level": 0.95,
        "confidence_interval_method": "student_t",
    }
    test_contract_v2 = {
        **legacy_test_contract,
        "schema_version": "pol-test-evaluation-contract-v2",
        "training_subset_policy": "canonical_train_order_prefix_v1",
        "training_subset_selection_boundary": (
            "all_subset_models_frozen_before_any_test_access"
        ),
    }
    expected_test_contract = (
        test_evaluation_contract()
        if manifest_schema in {
            "pol-study-run-manifest-v13",
            "pol-study-run-manifest-v14",
        }
        else test_contract_v2
        if manifest_schema == "pol-study-run-manifest-v12"
        else legacy_test_contract
    )
    if plan.get("test_evaluation_contract") != expected_test_contract:
        raise ValueError("unsupported frozen test-evaluation contract")
    _verify_selection_search_contract(
        resolved_study=resolved_study,
        selection=selection,
        plan=plan,
    )
    resolved_provenance = {
        variant["id"]: variant["selection_source"]
        for variant in resolved_study.get("variants", [])
        if isinstance(variant, Mapping)
        and variant.get("selection_source") is not None
    }
    if any(
        not isinstance(provenance, Mapping)
        or provenance.get("kind")
        != "resolved_completed_study_selection"
        for provenance in resolved_provenance.values()
    ):
        raise ValueError(
            "resolved study contains unresolved selection-source provenance"
        )
    if selection.get("selection_source_provenance") != resolved_provenance:
        raise ValueError(
            "selection record does not match resolved selection-source provenance"
        )
    if summary.get("selection_source_binding_count") != len(
        resolved_provenance
    ):
        raise ValueError("study summary selection-source count mismatch")
    if summary.get("selection_source_provenance_hash") != (
        stable_object_hash(resolved_provenance)
    ):
        raise ValueError("study summary selection-source hash mismatch")

    dataset_reference = json.loads(
        (root / "dataset_reference.json").read_text(encoding="utf-8")
    )
    if not isinstance(dataset_reference, Mapping):
        raise ValueError("study dataset reference must be an object")
    if (
        dataset_reference.get("schema_version")
        != "pol-study-dataset-reference-v3"
    ):
        raise ValueError("unsupported study dataset-reference schema")
    verify_execution_device_policy(
        dataset_reference,
        boundary="study dataset reference",
    )
    dataset_binding_proof = dataset_reference.get("binding_proof")
    if not isinstance(dataset_binding_proof, Mapping):
        raise ValueError("study dataset reference has no binding proof")
    # Load the data package first because its public package initialization
    # establishes the existing data/validation binding import order.
    import_module("pol.data.dataset")
    from pol.validation.binding import verify_binding_proof

    verify_binding_proof(dataset_binding_proof)
    expected_dataset_binding = {
        "dataset_binding_kind": dataset_binding_proof["binding_kind"],
        "dataset_binding_status": dataset_binding_proof["status"],
        "dataset_target_reference_validation_status": dataset_binding_proof[
            "target_reference_validation_status"
        ],
        "dataset_binding_proof_hash": dataset_binding_proof["proof_hash"],
    }
    for source_name, source in (
        ("identity", identity),
        ("selection record", selection),
        ("frozen plan", plan),
        ("run summary", summary),
        ("dataset reference", dataset_reference),
    ):
        for field, expected_value in expected_dataset_binding.items():
            if source.get(field) != expected_value:
                raise ValueError(
                    f"{source_name} dataset validation binding mismatch: {field}"
                )
        verify_execution_device_policy(
            source,
            boundary=source_name,
        )
    if identity.get("dataset_artifact_id") != dataset_reference.get(
        "artifact_id"
    ):
        raise ValueError("manifest dataset binding mismatch")
    if identity.get("dataset_split_hash") != dataset_reference.get("split_hash"):
        raise ValueError("manifest split binding mismatch")
    if plan.get("dataset_artifact_id") != dataset_reference.get("artifact_id"):
        raise ValueError("frozen plan dataset binding mismatch")
    if plan.get("dataset_split_hash") != dataset_reference.get("split_hash"):
        raise ValueError("frozen plan split binding mismatch")
    if summary.get("dataset_artifact_id") != dataset_reference.get(
        "artifact_id"
    ):
        raise ValueError("run summary dataset binding mismatch")
    if dataset_reference.get("validation_artifact_id") != (
        dataset_binding_proof.get("certificate_artifact_id")
    ):
        raise ValueError("study dataset-reference certificate binding mismatch")

    model_name = plan.get("frozen_models_file")
    if not isinstance(model_name, str) or Path(model_name).name != model_name:
        raise ValueError("unsafe frozen model filename")
    model_path = root / model_name
    if file_sha256(model_path) != plan.get("frozen_models_sha256"):
        raise ValueError("frozen model archive hash mismatch")
    archive = torch.load(model_path, map_location="cpu", weights_only=True)
    expected_archive_schema = {
        "pol-study-run-manifest-v12": "pol-frozen-model-archive-v8",
        "pol-study-run-manifest-v13": "pol-frozen-model-archive-v9",
        "pol-study-run-manifest-v14": "pol-frozen-model-archive-v9",
    }.get(str(manifest_schema), "pol-frozen-model-archive-v7")
    if archive.get("schema_version") != expected_archive_schema:
        raise ValueError("unsupported frozen model archive schema")
    verify_execution_device_policy(
        archive,
        boundary="frozen model archive",
    )
    require_cpu_tensors(
        archive,
        boundary="frozen model archive load",
        name="archive",
    )
    if archive.get("selection_record_hash") != selection_hash:
        raise ValueError("frozen model archive selection binding mismatch")
    verify_selection_source_provenance_bindings(
        selection=selection,
        plan=plan,
        archive=archive,
    )
    models = archive.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("frozen model archive models must be an object")
    expected = {
        (case_id, readout_id, candidate_id)
        for case_id, case in plan.get("cases", {}).items()
        for readout_id, candidate_id in case.get(
            "selected_by_readout", {}
        ).items()
    }
    actual = {
        (
            entry.get("case_id"),
            entry.get("readout_id"),
            entry.get("candidate_id"),
        )
        for entry in models.values()
        if isinstance(entry, Mapping)
    }
    if actual != expected or len(actual) != len(models):
        raise ValueError("frozen model archive does not match selected candidates")
    entry_by_binding = {
        (
            entry["case_id"],
            entry["readout_id"],
            entry["candidate_id"],
        ): entry
        for entry in models.values()
    }
    model_by_binding = {
        binding: entry["model"]
        for binding, entry in entry_by_binding.items()
    }
    verify_representative_feature_bindings(
        models,
        selection=selection,
        plan=plan,
    )
    direct_diagnostic_count, direct_zero_fill_count = (
        verify_frozen_decoder_bindings(
            models,
            selection=selection,
            plan=plan,
        )
    )

    validation_rows = load_rows(root / "validation_trials.csv")
    test_rows = load_rows(root / "test_metrics.csv")
    seed_rows = load_rows(root / "random_feature_seed_metrics.csv")
    ensemble_rows = load_rows(root / "random_feature_ensemble_metrics.csv")
    multiplier_rows = load_rows(root / "heat_multiplier.csv")
    multiplier_summary_rows = load_rows(root / "heat_multiplier_summary.csv")
    noise_rows = load_rows(root / "noise_robustness.csv")
    stability_model_rows = load_rows(root / "readout_stability_models.csv")
    stability_repeat_rows = load_rows(
        root / "readout_stability_noise_repeats.csv"
    )
    stability_summary_rows = load_rows(
        root / "readout_stability_noise_summary.csv"
    )
    stability_ensemble_repeat_rows = load_rows(
        root / "readout_stability_noise_ensemble_repeats.csv"
    )
    stability_ensemble_summary_rows = load_rows(
        root / "readout_stability_noise_ensemble_summary.csv"
    )
    comparison_rows = load_rows(root / "selected_comparison.csv")
    skipped_trials = json.loads(
        (root / "skipped_trials.json").read_text(encoding="utf-8")
    )
    if not isinstance(skipped_trials, list):
        raise ValueError("skipped-trial artifact must be a list")
    if manifest_schema in {
        "pol-study-run-manifest-v9",
        "pol-study-run-manifest-v10",
        "pol-study-run-manifest-v11",
        "pol-study-run-manifest-v12",
        "pol-study-run-manifest-v13",
        "pol-study-run-manifest-v14",
    }:
        for row in validation_rows:
            _verify_result_row_contract(
                row,
                table="validation",
                provenance_by_variant=resolved_provenance,
                model=model_by_binding.get(_row_binding(row)),
                expected_schema=(
                    "pol-study-result-row-v3"
                    if manifest_schema in {
                        "pol-study-run-manifest-v13",
                        "pol-study-run-manifest-v14",
                    }
                    else "pol-study-result-row-v2"
                    if manifest_schema == "pol-study-run-manifest-v12"
                    else "pol-study-result-row-v1"
                ),
            )
        for table, rows in (
            ("primary test", test_rows),
            ("random-feature seed", seed_rows),
            ("random-feature ensemble", ensemble_rows),
        ):
            for row in rows:
                _verify_result_row_contract(
                    row,
                    table=table,
                    provenance_by_variant=resolved_provenance,
                    model=model_by_binding.get(_row_binding(row)),
                    expected_schema=(
                        "pol-study-result-row-v3"
                        if manifest_schema in {
                            "pol-study-run-manifest-v13",
                            "pol-study-run-manifest-v14",
                        }
                        else "pol-study-result-row-v2"
                        if manifest_schema == "pol-study-run-manifest-v12"
                        else "pol-study-result-row-v1"
                    ),
                )
        _verify_fixed_dimension_rows(
            resolved_study,
            {
                "validation": validation_rows,
                "primary test": test_rows,
                "random-feature seed": seed_rows,
                "random-feature ensemble": ensemble_rows,
            },
        )
    if manifest_schema in {
        "pol-study-run-manifest-v12",
        "pol-study-run-manifest-v13",
        "pol-study-run-manifest-v14",
    }:
        _verify_training_subset_contract(
            resolved_study=resolved_study,
            selection=selection,
            plan=plan,
            models=models,
            validation_rows=validation_rows,
            test_rows=test_rows,
            seed_rows=seed_rows,
            ensemble_rows=ensemble_rows,
        )
    expected_grid_skips = [
        {"case_id": case_id, **item}
        for case_id, case in selection["cases"].items()
        for item in case["skipped_candidates"]
        if case.get("search_kind") == "grid"
    ]
    actual_grid_skips = [
        item
        for item in skipped_trials
        if isinstance(item, Mapping)
        and str(item.get("stage", "")).startswith("grid:")
    ]
    if actual_grid_skips != expected_grid_skips:
        raise ValueError("skipped-trial grid evidence mismatch")
    if summary.get("validation_row_count") != len(validation_rows):
        raise ValueError("run summary validation-row count mismatch")
    if summary.get("primary_test_row_count") != len(test_rows):
        raise ValueError("run summary primary-test-row count mismatch")
    if summary.get("random_feature_seed_row_count") != len(seed_rows):
        raise ValueError("run summary random-feature-seed-row count mismatch")
    if summary.get("random_feature_ensemble_row_count") != len(
        ensemble_rows
    ):
        raise ValueError("run summary random-feature-ensemble-row count mismatch")
    if summary.get("heat_multiplier_coefficient_row_count") != len(
        multiplier_rows
    ):
        raise ValueError("run summary heat-multiplier coefficient-row count mismatch")
    if summary.get("heat_multiplier_summary_row_count") != len(
        multiplier_summary_rows
    ):
        raise ValueError("run summary heat-multiplier summary-row count mismatch")
    if manifest_schema in {
        "pol-study-run-manifest-v11",
        "pol-study-run-manifest-v12",
        "pol-study-run-manifest-v13",
        "pol-study-run-manifest-v14",
    }:
        for field, rows in (
            ("readout_stability_model_row_count", stability_model_rows),
            ("readout_stability_repeat_row_count", stability_repeat_rows),
            ("readout_stability_summary_row_count", stability_summary_rows),
            (
                "readout_stability_ensemble_repeat_row_count",
                stability_ensemble_repeat_rows,
            ),
            (
                "readout_stability_ensemble_summary_row_count",
                stability_ensemble_summary_rows,
            ),
        ):
            if summary.get(field) != len(rows):
                raise ValueError(f"run summary {field} mismatch")
    elif summary.get("noise_robustness_row_count") != len(noise_rows):
        raise ValueError("run summary noise-robustness row count mismatch")
    if summary.get("skipped_trial_count") != len(skipped_trials):
        raise ValueError("run summary skipped-trial count mismatch")
    selection_cases = selection["cases"]
    planned_grid = sum(
        int(case.get("planned_cartesian_cell_count") or 0)
        for case in selection_cases.values()
    )
    evaluated_grid = sum(
        int(case.get("evaluated_candidate_count") or 0)
        for case in selection_cases.values()
        if case.get("search_kind") == "grid"
    )
    skipped_grid = sum(
        int(case.get("skipped_candidate_count") or 0)
        for case in selection_cases.values()
        if case.get("search_kind") == "grid"
    )
    if summary.get("planned_cartesian_cell_count") != planned_grid:
        raise ValueError("run summary planned Cartesian count mismatch")
    if summary.get("evaluated_cartesian_cell_count") != evaluated_grid:
        raise ValueError("run summary evaluated Cartesian count mismatch")
    if summary.get("skipped_cartesian_cell_count") != skipped_grid:
        raise ValueError("run summary skipped Cartesian count mismatch")
    diagnostic_specs = resolved_study.get("diagnostics", [])
    if manifest_schema in {
        "pol-study-run-manifest-v11",
        "pol-study-run-manifest-v12",
        "pol-study-run-manifest-v13",
        "pol-study-run-manifest-v14",
    }:
        legacy_noise_path = root / "noise_robustness.csv"
        if legacy_noise_path.exists():
            raise ValueError(
                "v11 study published a legacy noise-robustness table"
            )
        stability_configured = any(
            isinstance(item, Mapping)
            and item.get("kind") == "readout_stability_noise"
            for item in diagnostic_specs
        )
        stability_paths = (
            root / "readout_stability_models.csv",
            root / "readout_stability_noise_repeats.csv",
            root / "readout_stability_noise_summary.csv",
            root / "readout_stability_noise_ensemble_repeats.csv",
            root / "readout_stability_noise_ensemble_summary.csv",
        )
        if not stability_configured and any(
            path.exists() for path in stability_paths
        ):
            raise ValueError(
                "study without stability diagnostic published a stability table"
            )
        _verify_readout_stability_diagnostics(
            resolved_study=resolved_study,
            models=models,
            selection_hash=selection_hash,
            frozen_plan_hash=stored_plan_hash,
            model_rows=stability_model_rows,
            repeat_rows=stability_repeat_rows,
            summary_rows=stability_summary_rows,
            ensemble_repeat_rows=stability_ensemble_repeat_rows,
            ensemble_summary_rows=stability_ensemble_summary_rows,
        )
    else:
        noise_configured = any(
            isinstance(item, Mapping)
            and item.get("kind") == "noise_robustness"
            for item in diagnostic_specs
        )
        if not noise_configured and (root / "noise_robustness.csv").exists():
            raise ValueError(
                "study without noise diagnostic published a noise table"
            )
    _verify_heat_multiplier_diagnostics(
        root,
        resolved_study=resolved_study,
        expected_bindings=expected,
        entry_by_binding=entry_by_binding,
        model_by_binding=model_by_binding,
        dataset_binding_proof=dataset_binding_proof,
        coefficient_rows=multiplier_rows,
        summary_rows=multiplier_summary_rows,
    )
    if (
        summary.get("direct_decoder_diagnostic_count")
        != direct_diagnostic_count
    ):
        raise ValueError("run summary direct-decoder diagnostic count mismatch")
    if (
        summary.get("direct_decoder_zero_fill_count")
        != direct_zero_fill_count
    ):
        raise ValueError("run summary direct-decoder zero-fill count mismatch")
    if summary.get("direct_decoder_zero_fill_applied") is not (
        direct_zero_fill_count > 0
    ):
        raise ValueError("run summary direct-decoder zero-fill flag mismatch")
    for row in validation_rows:
        if row.get("readout_kind") == "direct_fourier_decoder":
            try:
                row_J = int(row["J"])
                row_q = int(row["q"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "direct validation row has invalid J/q"
                ) from exc
            frozen_entry = entry_by_binding.get(_row_binding(row))
            if frozen_entry is None:
                J, q = row_J, row_q
            else:
                frozen_trial = TrialSpec.model_validate(frozen_entry["trial"])
                J = int(frozen_trial.feature.observation.J)
                q = int(frozen_trial.output.q)
                if (row_J, row_q) != (J, q):
                    raise ValueError(
                        "selected direct validation row J/q does not match "
                        "the frozen trial"
                    )
            verify_fixed_fourier_decoder_diagnostic(
                row,
                observation_count=J,
                requested_q=q,
                boundary="direct validation row",
            )
        else:
            verify_no_decoder_diagnostic(
                row,
                boundary="non-direct validation row",
            )
    selected_validation = {
        (row.get("case_id"), row.get("readout_id"), row.get("candidate_id"))
        for row in validation_rows
        if str(row.get("selected", "")).lower() == "true"
    }
    if selected_validation != expected:
        raise ValueError(
            "validation selected rows do not match frozen candidates"
        )
    _verify_validation_selection_order(
        resolved_study=resolved_study,
        selection=selection,
        validation_rows=validation_rows,
    )
    actual_test_rows = {
        (row.get("case_id"), row.get("readout_id"), row.get("candidate_id"))
        for row in test_rows
    }
    if len(test_rows) != len(expected) or actual_test_rows != expected:
        raise ValueError("test rows do not match frozen candidates")

    def verify_test_binding(
        row: Mapping[str, Any],
        *,
        table: str,
    ) -> None:
        if str(row.get("selected", "")).lower() != "true":
            raise ValueError(f"{table} row is not marked as selected")
        if row.get("selection_record_hash") != selection_hash:
            raise ValueError(f"{table} row selection binding mismatch")
        if row.get("frozen_plan_hash") != stored_plan_hash:
            raise ValueError(f"{table} row frozen-plan binding mismatch")

    seed_rows_by_binding: dict[
        tuple[Any, Any, Any],
        list[dict[str, Any]],
    ] = {}
    for row in seed_rows:
        verify_test_binding(row, table="random-feature seed")
        verify_no_decoder_diagnostic(
            row,
            boundary="random-feature seed row",
        )
        seed_rows_by_binding.setdefault(_row_binding(row), []).append(row)
    ensemble_rows_by_binding: dict[
        tuple[Any, Any, Any],
        list[dict[str, Any]],
    ] = {}
    for row in ensemble_rows:
        verify_test_binding(row, table="random-feature ensemble")
        verify_no_decoder_diagnostic(
            row,
            boundary="random-feature ensemble row",
        )
        ensemble_rows_by_binding.setdefault(_row_binding(row), []).append(row)

    random_bindings = {
        binding
        for binding, model in model_by_binding.items()
        if isinstance(model, Mapping)
        and model.get("kind") == "random_feature_ridge"
    }
    if set(seed_rows_by_binding) != random_bindings:
        raise ValueError(
            "per-seed rows do not match frozen random-feature models"
        )
    if set(ensemble_rows_by_binding) != random_bindings:
        raise ValueError(
            "ensemble rows do not match frozen random-feature models"
        )

    seed_summary_suffixes = (
        "_seed_mean",
        "_seed_std",
        "_seed_ci95_low",
        "_seed_ci95_high",
        "_seed_q25",
        "_seed_median",
        "_seed_q75",
    )
    seed_metadata_fields = (
        "test_seed_count",
        "test_seed_std_ddof",
        "test_confidence_level",
        "test_confidence_interval_method",
        "test_seed_descriptive_quantiles",
        "test_seed_quantile_method",
        "test_seed_quantiles_are_uncertainty_interval",
    )
    stability_models_by_map_hash = {
        (
            _row_binding(item),
            str(item["random_map_parameter_hash"]),
        ): item
        for item in stability_model_rows
        if item.get("random_map_parameter_hash") not in ("", None)
    }
    if len(stability_models_by_map_hash) != sum(
        item.get("random_map_parameter_hash") not in ("", None)
        for item in stability_model_rows
    ):
        raise ValueError("duplicate random-map hash in stability model table")
    for row in test_rows:
        verify_test_binding(row, table="primary test")
        binding = _row_binding(row)
        model = model_by_binding[binding]
        if not isinstance(model, Mapping):
            raise ValueError("frozen model entry is not an object")
        if model.get("kind") == "direct_fourier_decoder":
            try:
                row_J = int(row["J"])
                row_q = int(row["q"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("direct test row has invalid J/q") from exc
            frozen_trial = TrialSpec.model_validate(
                entry_by_binding[binding]["trial"]
            )
            J = int(frozen_trial.feature.observation.J)
            q = int(frozen_trial.output.q)
            if (row_J, row_q) != (J, q):
                raise ValueError(
                    "direct test row J/q does not match the frozen trial"
                )
            verify_fixed_fourier_decoder_diagnostic(
                row,
                observation_count=J,
                requested_q=q,
                boundary="direct test row",
            )
        else:
            verify_no_decoder_diagnostic(
                row,
                boundary="non-direct primary test row",
            )
        if model.get("kind") != "random_feature_ridge":
            if row.get("test_result_kind") != "single_model":
                raise ValueError(
                    "deterministic primary row has the wrong result kind"
                )
            if any(
                _has_csv_value(row, key) for key in seed_metadata_fields
            ):
                raise ValueError(
                    "single-model primary row has false seed uncertainty"
                )
            if any(
                _has_csv_value(row, key)
                for key in row
                if key.endswith(seed_summary_suffixes)
            ):
                raise ValueError(
                    "single-model primary row has false seed summary"
                )
            continue

        if (
            row.get("test_result_kind")
            != "independent_seed_metric_summary"
        ):
            raise ValueError(
                "random-feature primary row has the wrong result kind"
            )
        members = model.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError(
                "frozen random-feature model has too few members"
            )
        member_seeds = [int(member["seed"]) for member in members]
        if len(member_seeds) != len(set(member_seeds)):
            raise ValueError(
                "frozen random-feature member seeds are not unique"
            )
        matching_seed_rows = seed_rows_by_binding[binding]
        try:
            row_seeds = [
                int(seed_row["seed"]) for seed_row in matching_seed_rows
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("per-seed row has an invalid seed") from exc
        if len(matching_seed_rows) != len(member_seeds):
            raise ValueError(
                "per-seed row count does not match frozen members"
            )
        if (
            len(row_seeds) != len(set(row_seeds))
            or set(row_seeds) != set(member_seeds)
        ):
            raise ValueError(
                "per-seed IDs do not match frozen member seeds"
            )
        if int(row.get("test_seed_count", -1)) != len(member_seeds):
            raise ValueError(
                "primary seed count does not match frozen members"
            )
        if int(row.get("test_seed_std_ddof", -1)) != 1:
            raise ValueError(
                "primary seed standard-deviation ddof is not one"
            )
        _require_close(
            row.get("test_confidence_level"),
            0.95,
            label="primary confidence level",
        )
        if row.get("test_confidence_interval_method") != "student_t":
            raise ValueError(
                "primary confidence interval method is not Student-t"
            )
        if manifest_schema in {
            "pol-study-run-manifest-v13",
            "pol-study-run-manifest-v14",
        }:
            if row.get("test_seed_descriptive_quantiles") != (
                "[0.25,0.5,0.75]"
            ):
                raise ValueError(
                    "primary descriptive seed quantiles are not canonical"
                )
            if row.get("test_seed_quantile_method") != "linear":
                raise ValueError("primary seed quantile method is not linear")
            if str(
                row.get("test_seed_quantiles_are_uncertainty_interval")
            ).lower() != "false":
                raise ValueError(
                    "descriptive seed quantiles are mislabeled as uncertainty"
                )
        if any(
            seed_row.get("test_result_kind")
            != "independent_seed_realization"
            for seed_row in matching_seed_rows
        ):
            raise ValueError("per-seed row has the wrong result kind")
        if manifest_schema in {
            "pol-study-run-manifest-v13",
            "pol-study-run-manifest-v14",
        }:
            member_by_seed = {
                int(member["seed"]): member for member in members
            }
            for seed_row in matching_seed_rows:
                seed = int(seed_row["seed"])
                expected_fields = random_feature_member_result_fields(
                    model,
                    member_by_seed[seed],
                )
                for key, expected_value in expected_fields.items():
                    actual_value = seed_row.get(key)
                    if isinstance(expected_value, float):
                        _require_close(
                            actual_value,
                            expected_value,
                            label=f"per-seed realization {key}",
                        )
                    elif isinstance(expected_value, bool):
                        if str(actual_value).lower() != str(
                            expected_value
                        ).lower():
                            raise ValueError(
                                f"per-seed realization {key} mismatch"
                            )
                    elif isinstance(expected_value, int):
                        if int(actual_value) != expected_value:
                            raise ValueError(
                                f"per-seed realization {key} mismatch"
                            )
                    elif expected_value is None:
                        if actual_value not in ("", None):
                            raise ValueError(
                                f"per-seed realization {key} mismatch"
                            )
                    elif actual_value != expected_value:
                        raise ValueError(
                            f"per-seed realization {key} mismatch"
                        )
                map_hash = str(
                    expected_fields["random_map_parameter_hash"]
                )
                stability_row = stability_models_by_map_hash.get(
                    (binding, map_hash)
                )
                if stability_models_by_map_hash and stability_row is None:
                    raise ValueError(
                        "per-seed row is not linked to its stability model"
                    )
                if stability_row is not None:
                    for key in (
                        "weight_frobenius_norm",
                        "weight_operator_norm",
                        "bias_norm",
                    ):
                        _require_close(
                            stability_row.get(key),
                            expected_fields[key],
                            label=f"stability-model link {key}",
                        )

        first_seed_row = matching_seed_rows[0]
        metric_keys = tuple(
            sorted(
                key
                for key, value in first_seed_row.items()
                if key.startswith("test_")
                and key != "test_result_kind"
                and value not in ("", None)
            )
        )
        if not metric_keys:
            raise ValueError("per-seed row has no test metrics")
        metric_items: list[dict[str, float]] = []
        for seed_row in matching_seed_rows:
            active_keys = tuple(
                sorted(
                    key
                    for key, value in seed_row.items()
                    if key.startswith("test_")
                    and key != "test_result_kind"
                    and value not in ("", None)
                )
            )
            if active_keys != metric_keys:
                raise ValueError(
                    "per-seed rows have inconsistent metric fields"
                )
            try:
                metric_items.append(
                    {key: float(seed_row[key]) for key in metric_keys}
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "per-seed metric is not numeric"
                ) from exc
        expected_summary = summarize_independent_seed_metrics(metric_items)
        for key, expected_value in expected_summary.items():
            _require_close(
                row.get(key),
                expected_value,
                label=f"primary metric {key}",
            )
        for key in metric_keys:
            _require_close(
                row.get(key),
                float(row[f"{key}_seed_mean"]),
                label=f"canonical metric {key}",
            )

        matching_ensemble_rows = ensemble_rows_by_binding[binding]
        if len(matching_ensemble_rows) != 1:
            raise ValueError(
                "random-feature model must have exactly one ensemble row"
            )
        ensemble_row = matching_ensemble_rows[0]
        if ensemble_row.get("test_result_kind") != "prediction_ensemble":
            raise ValueError("ensemble row has the wrong result kind")
        if int(ensemble_row.get("ensemble_member_count", -1)) != len(
            member_seeds
        ):
            raise ValueError(
                "ensemble member count does not match frozen members"
            )
        if manifest_schema in {
            "pol-study-run-manifest-v13",
            "pol-study-run-manifest-v14",
        }:
            expected_member_hashes = [
                random_feature_member_result_fields(model, member)[
                    "frozen_member_parameter_hash"
                ]
                for member in members
            ]
            if ensemble_row.get("ensemble_member_seeds_hash") != (
                stable_object_hash(member_seeds)
            ):
                raise ValueError("ensemble member seed hash mismatch")
            if ensemble_row.get("ensemble_member_parameters_hash") != (
                stable_object_hash(expected_member_hashes)
            ):
                raise ValueError("ensemble member parameter hash mismatch")
        expected_ensemble_keys = {
            key.replace("test_", "test_ensemble_", 1)
            for key in metric_keys
            if (
                manifest_schema
                in {
                    "pol-study-run-manifest-v9",
                    "pol-study-run-manifest-v10",
                    "pol-study-run-manifest-v11",
                    "pol-study-run-manifest-v12",
                    "pol-study-run-manifest-v13",
                    "pol-study-run-manifest-v14",
                }
                or "representation_floor" not in key
            )
        }
        actual_ensemble_keys = {
            key
            for key, value in ensemble_row.items()
            if key.startswith("test_ensemble_") and value not in ("", None)
        }
        if actual_ensemble_keys != expected_ensemble_keys:
            raise ValueError(
                "ensemble metric fields do not match prediction metrics"
            )
        try:
            for key in actual_ensemble_keys:
                float(ensemble_row[key])
        except (TypeError, ValueError) as exc:
            raise ValueError("ensemble metric is not numeric") from exc

    if manifest_schema == "pol-study-run-manifest-v14":
        _verify_prediction_capture_artifact(
            root,
            resolved_study=resolved_study,
            summary=summary,
            dataset_reference=dataset_reference,
            selection_hash=selection_hash,
            frozen_plan_hash=str(stored_plan_hash),
            frozen_model_archive_sha256=str(
                plan["frozen_models_sha256"]
            ),
            models=models,
            test_rows=test_rows,
            seed_rows=seed_rows,
            ensemble_rows=ensemble_rows,
        )
        _verify_report_publication(
            root,
            resolved_study=resolved_study,
            summary=summary,
        )

    if manifest_schema in {
        "pol-study-run-manifest-v10",
        "pol-study-run-manifest-v11",
        "pol-study-run-manifest-v12",
        "pol-study-run-manifest-v13",
        "pol-study-run-manifest-v14",
    }:
        expected_comparison = build_selected_comparison_rows(
            validation_rows=validation_rows,
            test_rows=test_rows,
        )
        if comparison_rows != expected_comparison:
            raise ValueError(
                "selected comparison table does not match canonical result tables"
            )
        if summary.get("selected_comparison_row_count") != len(
            comparison_rows
        ):
            raise ValueError(
                "run summary selected-comparison row count mismatch"
            )

    events = json.loads((root / "events.json").read_text(encoding="utf-8"))
    names = [
        item.get("event") for item in events if isinstance(item, Mapping)
    ]
    required = (
        "selection_complete",
        "convergence_complete",
        "freeze_written",
        "freeze_read_back",
        "first_test_state_solve",
        "first_test_metric",
    )
    if any(name not in names for name in required):
        raise ValueError("study-run event log is incomplete")
    if not (
        names.index("selection_complete")
        < names.index("freeze_written")
        < names.index("freeze_read_back")
        < names.index("first_test_state_solve")
        <= names.index("first_test_metric")
    ):
        raise ValueError(
            "study-run event order violates the freeze/test boundary"
        )
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("study-run event must be an object")
        if event.get("event") in {
            "freeze_written",
            "freeze_read_back",
            "first_test_state_solve",
            "first_test_metric",
        } and event.get("plan_content_hash") != stored_plan_hash:
            raise ValueError(
                "study-run event has the wrong frozen-plan binding"
            )


def verify_study_run(path: Path | str) -> dict[str, Any]:
    root = Path(path).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"not a safe study run directory: {root}")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("study run has no manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {
        "pol-study-run-manifest-v8",
        "pol-study-run-manifest-v9",
        "pol-study-run-manifest-v10",
        "pol-study-run-manifest-v11",
        "pol-study-run-manifest-v12",
        "pol-study-run-manifest-v13",
        "pol-study-run-manifest-v14",
    }:
        raise ValueError("unsupported study-run manifest")
    expected_records = manifest.get("files")
    if not isinstance(expected_records, list):
        raise ValueError("study-run files must be a list")
    actual_names: list[str] = []
    for path_item in root.rglob("*"):
        if path_item.is_symlink():
            raise ValueError(f"study run contains a symlink: {path_item}")
        if path_item.is_file() and path_item.name != "manifest.json":
            actual_names.append(path_item.relative_to(root).as_posix())
    expected_names = [
        record["relative_path"] for record in expected_records
    ]
    if sorted(actual_names) != sorted(expected_names):
        raise ValueError("study-run file tree differs from manifest")
    if manifest_records(root, expected_names) != expected_records:
        raise ValueError("study-run bytes differ from manifest")
    _verify_study_semantics(root, manifest)
    return manifest
