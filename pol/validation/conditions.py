"""Canonical target-solver conditions and target-reference contracts."""
from __future__ import annotations

import json
import math
from typing import Any, Mapping

from pol.runtime.hashing import stable_object_hash
from pol.systems.burgers import normalize_solver_name, step_metadata

CONVERGENCE_ROW_SCHEMA_VERSION = "pol-reference-convergence-row-v3"
CONVERGENCE_CSV_SCHEMA_VERSION = "pol-reference-convergence-csv-v3"
CROSS_SOLVER_CHECK_SCHEMA_VERSION = "pol-burgers-cross-solver-check-v2"
CROSS_SOLVER_METRIC_DEFINITION = {
    "name": "symmetric_two_norm_over_sum",
    "version": 1,
    "field_relative_l2": "2 * ||a-b||_L2 / (||a||_L2 + ||b||_L2)",
    "low_mode_relative_l2": (
        "2 * ||A_q-B_q||_2 / (||A_q||_2 + ||B_q||_2)"
    ),
    "zero_denominator_policy": "machine_epsilon_clamp_with_zero_numerator",
    "pass_rule": (
        "all three relative metrics are at or below their named tolerances"
    ),
}
CONVERGENCE_ROW_FIELDS = (
    "check_kind",
    "candidate_axis",
    "coarse_candidate_index",
    "fine_candidate_index",
    "coarse_reference_candidate_index",
    "fine_reference_candidate_index",
    "coarse_nx",
    "fine_nx",
    "coarse_condition_index",
    "fine_condition_index",
    "coarse_condition_json",
    "coarse_condition_hash",
    "fine_condition_json",
    "fine_condition_hash",
    "common_nx",
    "coarse_requested_outer_dt",
    "coarse_requested_fine_dt",
    "coarse_outer_step_count",
    "coarse_effective_substep",
    "coarse_substeps_per_outer",
    "coarse_dt",
    "coarse_nonlinear_filter",
    "fine_requested_outer_dt",
    "fine_requested_fine_dt",
    "fine_outer_step_count",
    "fine_effective_substep",
    "fine_substeps_per_outer",
    "fine_dt",
    "fine_nonlinear_filter",
    "mean_relative_l2",
    "max_relative_l2",
    "low_mode_relative_l2",
    "status",
    "row_hash",
)


def _canonical_condition_json(condition: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(condition),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _condition_step_fields(
    prefix: str,
    condition: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        f"{prefix}_requested_outer_dt": condition.get(
            "requested_outer_dt"
        ),
        f"{prefix}_requested_fine_dt": condition.get(
            "requested_fine_dt"
        ),
        f"{prefix}_outer_step_count": condition.get("outer_step_count"),
        f"{prefix}_effective_substep": condition.get("effective_substep"),
        f"{prefix}_substeps_per_outer": condition.get(
            "substeps_per_outer"
        ),
        f"{prefix}_dt": condition.get("dt"),
        f"{prefix}_nonlinear_filter": condition.get(
            "nonlinear_filter"
        ),
    }


def make_convergence_row(
    *,
    check_kind: str,
    candidate_axis: str,
    coarse_candidate_index: int,
    fine_candidate_index: int,
    coarse_reference_candidate_index: int,
    fine_reference_candidate_index: int,
    coarse_nx: int,
    fine_nx: int,
    coarse_condition_index: int,
    fine_condition_index: int,
    coarse_condition: Mapping[str, Any],
    fine_condition: Mapping[str, Any],
    common_nx: int,
    metrics: Mapping[str, float],
    status: str,
) -> dict[str, Any]:
    """Build one canonical long-form convergence comparison row."""
    row = {
        "check_kind": check_kind,
        "candidate_axis": candidate_axis,
        "coarse_candidate_index": coarse_candidate_index,
        "fine_candidate_index": fine_candidate_index,
        "coarse_reference_candidate_index": (
            coarse_reference_candidate_index
        ),
        "fine_reference_candidate_index": fine_reference_candidate_index,
        "coarse_nx": coarse_nx,
        "fine_nx": fine_nx,
        "coarse_condition_index": coarse_condition_index,
        "fine_condition_index": fine_condition_index,
        "coarse_condition_json": _canonical_condition_json(
            coarse_condition
        ),
        "coarse_condition_hash": stable_object_hash(
            dict(coarse_condition)
        ),
        "fine_condition_json": _canonical_condition_json(fine_condition),
        "fine_condition_hash": stable_object_hash(dict(fine_condition)),
        "common_nx": common_nx,
        **_condition_step_fields("coarse", coarse_condition),
        **_condition_step_fields("fine", fine_condition),
        "mean_relative_l2": float(metrics["mean_relative_l2"]),
        "max_relative_l2": float(metrics["max_relative_l2"]),
        "low_mode_relative_l2": float(
            metrics["low_mode_relative_l2"]
        ),
        "status": status,
    }
    row["row_hash"] = stable_object_hash(row)
    return row


def reference_refinement_proof(
    candidates: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    if any(type(value) is not int for value in candidates):
        raise ValueError("reference_nx_candidates must contain integers")
    values = list(candidates)
    if (
        len(values) < 2
        or values != sorted(set(values))
        or any(value < 2 for value in values)
    ):
        raise ValueError(
            "reference_nx_candidates must be strictly increasing and unique"
        )
    return {
        "schema_version": "pol-reference-refinement-proof-v1",
        "status": "pass",
        "ordered_candidates": values,
        "candidate_hash": stable_object_hash(values),
        "adjacent_pairs": [
            {
                "coarse_candidate_index": index,
                "fine_candidate_index": index + 1,
                "coarse_nx": values[index],
                "fine_nx": values[index + 1],
                "strictly_increasing": True,
                "status": "pass",
            }
            for index in range(len(values) - 1)
        ],
    }


def canonical_solver_name(system_kind: str, solver: str | None) -> str:
    if system_kind == "heat":
        if solver not in (None, "spectral_exact"):
            raise ValueError("heat numerical condition must use spectral_exact")
        return "spectral_exact"
    if system_kind == "burgers":
        try:
            return normalize_solver_name(str(solver))
        except ValueError as exc:
            raise ValueError(f"unsupported Burgers solver: {solver!r}") from exc
    if system_kind == "reaction_diffusion":
        if solver != "semi_implicit_spectral_euler":
            raise ValueError(
                "reaction-diffusion numerical condition must use "
                "semi_implicit_spectral_euler"
            )
        return "semi_implicit_spectral_euler"
    raise ValueError(
        f"target-reference numerical conditions do not support {system_kind!r}"
    )


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def canonical_numerical_condition(
    system_kind: str,
    values: Mapping[str, Any],
    *,
    evolution_time: float | None = None,
) -> dict[str, Any]:
    """Return the exact JSON condition used by certificates and binding."""
    if system_kind == "heat":
        return {"solver": canonical_solver_name("heat", values.get("solver"))}
    if system_kind == "reaction_diffusion":
        if evolution_time is None:
            raise ValueError(
                "reaction-diffusion canonical numerical condition requires "
                "evolution_time"
            )
        final_time = _positive_finite(
            evolution_time,
            name="reaction-diffusion evolution time",
        )
        dt = _positive_finite(
            values.get("dt"),
            name="reaction-diffusion dt",
        )
        steps = round(final_time / dt)
        if (
            steps < 1
            or abs(steps * dt - final_time)
            > 1e-10 * max(1.0, abs(final_time))
        ):
            raise ValueError(
                "reaction-diffusion dt must align exactly with final time"
            )
        nonlinear_filter = values.get("nonlinear_filter")
        if nonlinear_filter not in {"none", "two_thirds"}:
            raise ValueError(
                "reaction-diffusion nonlinear_filter must be none or "
                "two_thirds"
            )
        return {
            "solver": canonical_solver_name(
                "reaction_diffusion",
                values.get("solver"),
            ),
            "dt": dt,
            "nonlinear_filter": nonlinear_filter,
        }
    if system_kind != "burgers":
        raise ValueError(
            f"target-reference numerical conditions do not support {system_kind!r}"
        )
    if evolution_time is None:
        raise ValueError(
            "Burgers canonical numerical condition requires evolution_time"
        )
    solver = canonical_solver_name("burgers", values.get("solver"))
    dealias = values.get("dealias")
    if type(dealias) is not bool:
        raise ValueError("Burgers dealias must be a boolean")
    requested_outer_dt = values.get(
        "requested_outer_dt",
        values.get("dt"),
    )
    requested_fine_dt = values.get(
        "requested_fine_dt",
        values.get("fine_dt"),
    )
    steps = step_metadata(
        solver=solver,
        dt=_positive_finite(
            requested_outer_dt,
            name="Burgers requested outer dt",
        ),
        fine_dt=(
            None
            if requested_fine_dt is None
            else _positive_finite(
                requested_fine_dt,
                name="Burgers requested fine dt",
            )
        ),
        final_time=_positive_finite(
            evolution_time,
            name="Burgers evolution time",
        ),
    )
    return {
        **steps,
        "dealias": dealias,
    }


def burgers_refinement_proof(
    candidates: list[Mapping[str, Any]]
    | tuple[Mapping[str, Any], ...],
    *,
    evolution_time: float,
) -> dict[str, Any]:
    """Canonicalize and prove a single-family coarse-to-fine sequence."""
    if len(candidates) < 2:
        raise ValueError("at least two Burgers time candidates are required")
    conditions = [
        canonical_numerical_condition(
            "burgers",
            candidate,
            evolution_time=evolution_time,
        )
        for candidate in candidates
    ]
    family = conditions[0]["solver"]
    dealias = conditions[0]["dealias"]
    adjacent_pairs: list[dict[str, Any]] = []
    for coarse_index, (coarse, fine) in enumerate(
        zip(conditions[:-1], conditions[1:])
    ):
        fine_index = coarse_index + 1
        if fine["solver"] != family:
            raise ValueError(
                "Burgers time_candidates must use one canonical solver family"
            )
        if fine["dealias"] != dealias:
            raise ValueError(
                "Burgers time_candidates must use one dealias policy"
            )
        if family == "etdrk4":
            if not (
                fine["requested_outer_dt"]
                < coarse["requested_outer_dt"]
            ):
                raise ValueError(
                    "ETDRK4 time_candidates must have strictly decreasing dt"
                )
            outer_nonincreasing = True
            effective_nonincreasing = True
            strict_outer = True
            strict_effective = True
        else:
            outer_nonincreasing = (
                fine["requested_outer_dt"]
                <= coarse["requested_outer_dt"]
            )
            effective_nonincreasing = (
                fine["effective_substep"]
                <= coarse["effective_substep"]
            )
            strict_outer = (
                fine["requested_outer_dt"]
                < coarse["requested_outer_dt"]
            )
            strict_effective = (
                fine["effective_substep"]
                < coarse["effective_substep"]
            )
            if not (
                outer_nonincreasing
                and effective_nonincreasing
                and (strict_outer or strict_effective)
            ):
                raise ValueError(
                    "split-step time_candidates must refine requested outer dt "
                    "and/or the actual effective substep in coarse-to-fine order"
                )
        adjacent_pairs.append(
            {
                "coarse_candidate_index": coarse_index,
                "fine_candidate_index": fine_index,
                "coarse_condition_hash": stable_object_hash(coarse),
                "fine_condition_hash": stable_object_hash(fine),
                "same_canonical_solver_family": True,
                "same_dealias_policy": True,
                "requested_outer_dt_nonincreasing": outer_nonincreasing,
                "effective_substep_nonincreasing": effective_nonincreasing,
                "requested_outer_dt_strictly_smaller": strict_outer,
                "effective_substep_strictly_smaller": strict_effective,
                "status": "pass",
            }
        )
    if any(
        condition["solver"] != family
        or condition["dealias"] != dealias
        for condition in conditions
    ):
        raise ValueError(
            "Burgers time_candidates must keep solver family and dealias fixed"
        )
    return {
        "schema_version": "pol-burgers-refinement-proof-v1",
        "status": "pass",
        "evolution_time": float(evolution_time),
        "canonical_solver_family": family,
        "dealias": dealias,
        "ordered_candidates": conditions,
        "candidate_hashes": [
            stable_object_hash(condition) for condition in conditions
        ],
        "adjacent_pairs": adjacent_pairs,
    }


def reaction_diffusion_refinement_proof(
    candidates: list[Mapping[str, Any]]
    | tuple[Mapping[str, Any], ...],
    *,
    evolution_time: float,
) -> dict[str, Any]:
    """Prove a fixed-method, fixed-filter, strictly decreasing dt sequence."""
    if len(candidates) < 2:
        raise ValueError(
            "at least two reaction-diffusion time candidates are required"
        )
    conditions = [
        canonical_numerical_condition(
            "reaction_diffusion",
            candidate,
            evolution_time=evolution_time,
        )
        for candidate in candidates
    ]
    solver = conditions[0]["solver"]
    nonlinear_filter = conditions[0]["nonlinear_filter"]
    adjacent_pairs: list[dict[str, Any]] = []
    for coarse_index, (coarse, fine) in enumerate(
        zip(conditions[:-1], conditions[1:])
    ):
        if fine["solver"] != solver:
            raise ValueError(
                "reaction-diffusion time_candidates must use one solver"
            )
        if fine["nonlinear_filter"] != nonlinear_filter:
            raise ValueError(
                "reaction-diffusion time_candidates must use one "
                "nonlinear_filter; filter switching is not refinement"
            )
        if not fine["dt"] < coarse["dt"]:
            raise ValueError(
                "reaction-diffusion time_candidates must have strictly "
                "decreasing dt"
            )
        adjacent_pairs.append(
            {
                "coarse_candidate_index": coarse_index,
                "fine_candidate_index": coarse_index + 1,
                "coarse_condition_hash": stable_object_hash(coarse),
                "fine_condition_hash": stable_object_hash(fine),
                "same_solver": True,
                "same_nonlinear_filter": True,
                "dt_strictly_decreasing": True,
                "status": "pass",
            }
        )
    if any(
        condition["solver"] != solver
        or condition["nonlinear_filter"] != nonlinear_filter
        for condition in conditions
    ):
        raise ValueError(
            "reaction-diffusion time_candidates must keep solver and "
            "nonlinear_filter fixed"
        )
    return {
        "schema_version": (
            "pol-reaction-diffusion-refinement-proof-v1"
        ),
        "status": "pass",
        "evolution_time": float(evolution_time),
        "solver": solver,
        "nonlinear_filter": nonlinear_filter,
        "ordered_candidates": conditions,
        "candidate_hashes": [
            stable_object_hash(condition) for condition in conditions
        ],
        "adjacent_pairs": adjacent_pairs,
    }


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def canonical_invariant_parameters(
    system_kind: str,
    values: Mapping[str, Any],
) -> dict[str, float]:
    if system_kind == "heat":
        return {
            "nu": _positive_finite(values.get("nu"), name="heat nu"),
        }
    if system_kind == "burgers":
        advection = values.get("advection_coefficient", 1.0)
        if isinstance(advection, bool) or not isinstance(
            advection, (int, float)
        ):
            raise ValueError(
                "Burgers advection_coefficient must be finite"
            )
        advection_value = float(advection)
        if not math.isfinite(advection_value):
            raise ValueError(
                "Burgers advection_coefficient must be finite"
            )
        return {
            "nu": _positive_finite(values.get("nu"), name="Burgers nu"),
            "advection_coefficient": advection_value,
        }
    if system_kind == "reaction_diffusion":
        return {
            "nu": _positive_finite(
                values.get("nu"),
                name="reaction-diffusion nu",
            ),
            "alpha": _finite_number(
                values.get("alpha"),
                name="reaction-diffusion alpha",
            ),
            "beta": _finite_number(
                values.get("beta"),
                name="reaction-diffusion beta",
            ),
        }
    raise ValueError(
        f"target-reference invariant parameters do not support {system_kind!r}"
    )


def _coarsest_passing_suffix_index(
    rows: list[Mapping[str, Any]],
) -> int | None:
    for index in range(len(rows)):
        if all(row.get("status") == "pass" for row in rows[index:]):
            return index
    return None


def _canonical_tolerances(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {
        "mean_relative_l2",
        "max_relative_l2",
        "low_mode_relative_l2",
    }:
        raise ValueError("convergence tolerances are incomplete")
    result: dict[str, float] = {}
    for name in (
        "mean_relative_l2",
        "max_relative_l2",
        "low_mode_relative_l2",
    ):
        raw = value[name]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0.0
        ):
            raise ValueError("convergence tolerances must be finite and nonnegative")
        result[name] = float(raw)
    return result


def _row_metrics_and_status(
    row: Mapping[str, Any],
    tolerances: Mapping[str, float],
) -> tuple[dict[str, float], str]:
    metrics: dict[str, float] = {}
    for name in (
        "mean_relative_l2",
        "max_relative_l2",
        "low_mode_relative_l2",
    ):
        value = row.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("convergence rows require finite nonnegative metrics")
        metrics[name] = float(value)
    status = (
        "pass"
        if all(metrics[name] <= tolerances[name] for name in metrics)
        else "fail"
    )
    return metrics, status


def _require_expected_row(
    row: Any,
    *,
    tolerances: Mapping[str, float],
    expected: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != set(
        CONVERGENCE_ROW_FIELDS
    ):
        raise ValueError("convergence row has unknown or missing fields")
    metrics, status = _row_metrics_and_status(row, tolerances)
    rebuilt = make_convergence_row(
        **expected,
        metrics=metrics,
        status=status,
    )
    if stable_object_hash(dict(row)) != stable_object_hash(rebuilt):
        raise ValueError(
            "convergence row semantics, status, or row hash are inconsistent"
        )
    return rebuilt


def _cross_solver_discrepancy_hash_payload(
    *,
    definition: Mapping[str, Any],
    finest_conditions: Mapping[str, Any],
    common_nx: int,
    sample_ids: list[int],
    metrics: Mapping[str, Any] | None,
    not_evaluated_reason: str | None,
) -> dict[str, Any]:
    return {
        "definition": dict(definition),
        "finest_conditions": dict(finest_conditions),
        "common_nx": common_nx,
        "sample_ids": sample_ids,
        "metrics": None if metrics is None else dict(metrics),
        "not_evaluated_reason": not_evaluated_reason,
    }


def cross_solver_discrepancy_evidence_hash(
    *,
    finest_conditions: Mapping[str, Any],
    common_nx: int,
    sample_ids: list[int],
    metrics: Mapping[str, Any] | None,
    not_evaluated_reason: str | None,
) -> str:
    return stable_object_hash(
        _cross_solver_discrepancy_hash_payload(
            definition=CROSS_SOLVER_METRIC_DEFINITION,
            finest_conditions=finest_conditions,
            common_nx=common_nx,
            sample_ids=sample_ids,
            metrics=metrics,
            not_evaluated_reason=not_evaluated_reason,
        )
    )


def validate_cross_solver_validation_block(
    block: Mapping[str, Any],
) -> None:
    """Reconstruct a Burgers cross-solver diagnostic from its evidence."""
    expected_keys = {
        "schema_version",
        "enabled",
        "status",
        "role",
        "context",
        "tolerances",
        "self_convergence",
        "finest_conditions",
        "discrepancy_definition",
        "discrepancy_metrics",
        "discrepancy_status",
        "discrepancy_not_evaluated_reason",
        "discrepancy_evidence_hash",
    }
    if set(block) != expected_keys:
        raise ValueError(
            "cross-solver validation block has unknown or missing fields"
        )
    if (
        block.get("schema_version") != CROSS_SOLVER_CHECK_SCHEMA_VERSION
        or block.get("enabled") is not True
        or block.get("role")
        != "supporting_evidence_not_primary_allowed_refinement"
    ):
        raise ValueError("unsupported cross-solver validation semantics")
    definition = block.get("discrepancy_definition")
    if stable_object_hash(definition) != stable_object_hash(
        CROSS_SOLVER_METRIC_DEFINITION
    ):
        raise ValueError("cross-solver discrepancy definition is invalid")

    context = block.get("context")
    if not isinstance(context, Mapping) or set(context) != {
        "system_kind",
        "invariant_parameters",
        "evolution_time",
        "domain_length",
        "dtype",
        "dealias",
        "common_nx",
        "reference_candidate_index",
        "sample_ids",
    }:
        raise ValueError("cross-solver validation context is incomplete")
    if context.get("system_kind") != "burgers":
        raise ValueError("cross-solver validation requires Burgers")
    invariants = context.get("invariant_parameters")
    if not isinstance(invariants, Mapping) or stable_object_hash(
        dict(invariants)
    ) != stable_object_hash(
        canonical_invariant_parameters("burgers", invariants)
    ):
        raise ValueError("cross-solver Burgers parameters are not canonical")
    evolution_time = _positive_finite(
        context.get("evolution_time"),
        name="cross-solver evolution_time",
    )
    domain_length = _positive_finite(
        context.get("domain_length"),
        name="cross-solver domain_length",
    )
    if context.get("dtype") not in {"float32", "float64"}:
        raise ValueError("cross-solver dtype is unsupported")
    if type(context.get("dealias")) is not bool:
        raise ValueError("cross-solver dealias must be boolean")
    common_nx = context.get("common_nx")
    reference_index = context.get("reference_candidate_index")
    if (
        type(common_nx) is not int
        or common_nx < 2
        or type(reference_index) is not int
        or reference_index < 0
    ):
        raise ValueError("cross-solver common grid metadata is invalid")
    sample_ids = context.get("sample_ids")
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or any(type(value) is not int or value < 0 for value in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError("cross-solver sample IDs are invalid")

    tolerances = _canonical_tolerances(block.get("tolerances"))
    self_convergence = block.get("self_convergence")
    finest_conditions = block.get("finest_conditions")
    if (
        not isinstance(self_convergence, Mapping)
        or set(self_convergence) != {"split_step", "etdrk4"}
        or not isinstance(finest_conditions, Mapping)
        or set(finest_conditions) != {"split_step", "etdrk4"}
    ):
        raise ValueError("cross-solver family evidence is incomplete")

    self_statuses: dict[str, str] = {}
    verified_finest: dict[str, dict[str, Any]] = {}
    for family in ("split_step", "etdrk4"):
        evidence = self_convergence[family]
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "status",
            "ordered_candidates",
            "candidate_refinement_proof",
            "rows",
            "rows_hash",
            "pairwise_row_hashes",
            "selected_candidate_index",
            "selected_condition",
            "finest_candidate_index",
            "finest_condition",
            "runtime_solver_metadata",
        }:
            raise ValueError(
                f"cross-solver {family} self-convergence evidence is invalid"
            )
        candidates = evidence.get("ordered_candidates")
        if not isinstance(candidates, list) or any(
            not isinstance(value, Mapping) for value in candidates
        ):
            raise ValueError(
                f"cross-solver {family} candidates are invalid"
            )
        proof = burgers_refinement_proof(
            candidates,
            evolution_time=evolution_time,
        )
        if (
            proof["canonical_solver_family"] != family
            or proof["dealias"] != context["dealias"]
            or stable_object_hash(evidence.get("candidate_refinement_proof"))
            != stable_object_hash(proof)
            or stable_object_hash(candidates)
            != stable_object_hash(proof["ordered_candidates"])
        ):
            raise ValueError(
                f"cross-solver {family} refinement proof is inconsistent"
            )
        rows = evidence.get("rows")
        if not isinstance(rows, list) or len(rows) != len(candidates) - 1:
            raise ValueError(
                f"cross-solver {family} self-convergence rows are incomplete"
            )
        verified_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            verified_rows.append(
                _require_expected_row(
                    row,
                    tolerances=tolerances,
                    expected={
                        "check_kind": "temporal",
                        "candidate_axis": "numerical_condition",
                        "coarse_candidate_index": index,
                        "fine_candidate_index": index + 1,
                        "coarse_reference_candidate_index": reference_index,
                        "fine_reference_candidate_index": reference_index,
                        "coarse_nx": common_nx,
                        "fine_nx": common_nx,
                        "coarse_condition_index": index,
                        "fine_condition_index": index + 1,
                        "coarse_condition": candidates[index],
                        "fine_condition": candidates[index + 1],
                        "common_nx": common_nx,
                    },
                )
            )
        selected_index = _coarsest_passing_suffix_index(verified_rows)
        expected_status = "pass" if selected_index is not None else "fail"
        finest_index = len(candidates) - 1
        if (
            evidence.get("status") != expected_status
            or evidence.get("selected_candidate_index") != selected_index
            or stable_object_hash(evidence.get("selected_condition"))
            != stable_object_hash(
                None
                if selected_index is None
                else candidates[selected_index]
            )
            or evidence.get("finest_candidate_index") != finest_index
            or stable_object_hash(evidence.get("finest_condition"))
            != stable_object_hash(candidates[finest_index])
            or evidence.get("rows_hash") != stable_object_hash(rows)
            or evidence.get("pairwise_row_hashes")
            != [row["row_hash"] for row in rows]
        ):
            raise ValueError(
                f"cross-solver {family} selection or row hashes are invalid"
            )
        metadata_rows = evidence.get("runtime_solver_metadata")
        if (
            not isinstance(metadata_rows, list)
            or len(metadata_rows) != len(candidates)
        ):
            raise ValueError(
                f"cross-solver {family} runtime metadata is incomplete"
            )
        for condition, metadata in zip(
            candidates,
            metadata_rows,
            strict=True,
        ):
            expected_metadata = {
                "kind": "burgers",
                "nu": float(invariants["nu"]),
                "time": evolution_time,
                **dict(condition),
                "domain_length": domain_length,
                "dtype": context["dtype"],
                "device": "cpu",
            }
            if stable_object_hash(metadata) != stable_object_hash(
                expected_metadata
            ):
                raise ValueError(
                    f"cross-solver {family} runtime step metadata is invalid"
                )
        self_statuses[family] = expected_status
        verified_finest[family] = dict(candidates[finest_index])
        if stable_object_hash(finest_conditions[family]) != stable_object_hash(
            verified_finest[family]
        ):
            raise ValueError(
                f"cross-solver {family} finest condition is inconsistent"
            )

    metrics = block.get("discrepancy_metrics")
    reason = block.get("discrepancy_not_evaluated_reason")
    both_self_pass = all(value == "pass" for value in self_statuses.values())
    if both_self_pass:
        if not isinstance(metrics, Mapping) or set(metrics) != {
            "mean_absolute_l2",
            "max_absolute_l2",
            "mean_relative_l2",
            "max_relative_l2",
            "low_mode_relative_l2",
        }:
            raise ValueError("cross-solver discrepancy metrics are incomplete")
        canonical_metrics: dict[str, float] = {}
        for name, value in metrics.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    "cross-solver discrepancy metrics must be finite and "
                    "nonnegative"
                )
            canonical_metrics[name] = float(value)
        discrepancy_status = (
            "pass"
            if all(
                canonical_metrics[name] <= tolerances[name]
                for name in (
                    "mean_relative_l2",
                    "max_relative_l2",
                    "low_mode_relative_l2",
                )
            )
            else "fail"
        )
        if reason is not None:
            raise ValueError(
                "passing self-convergence cannot have a skipped discrepancy"
            )
    else:
        canonical_metrics = {}
        discrepancy_status = "not_evaluated"
        if metrics is not None or reason != (
            "self_convergence_must_pass_before_cross_comparison"
        ):
            raise ValueError(
                "failed self-convergence must skip cross comparison"
            )
    if block.get("discrepancy_status") != discrepancy_status:
        raise ValueError("cross-solver discrepancy status is inconsistent")
    hash_payload = _cross_solver_discrepancy_hash_payload(
        definition=CROSS_SOLVER_METRIC_DEFINITION,
        finest_conditions=verified_finest,
        common_nx=common_nx,
        sample_ids=sample_ids,
        metrics=None if metrics is None else canonical_metrics,
        not_evaluated_reason=reason,
    )
    if block.get("discrepancy_evidence_hash") != stable_object_hash(
        hash_payload
    ):
        raise ValueError("cross-solver discrepancy evidence hash is invalid")
    expected_status = (
        "pass"
        if both_self_pass and discrepancy_status == "pass"
        else "fail"
    )
    if block.get("status") != expected_status:
        raise ValueError("cross-solver overall status is inconsistent")


def validate_target_reference_contract(
    contract: Mapping[str, Any],
) -> None:
    """Reject incomplete or non-reconstructable v4 target contracts."""
    expected_keys = {
        "schema_version",
        "system_kind",
        "invariant_parameters",
        "evolution_time",
        "dtype",
        "domain_length",
        "reference_resolution",
        "numerical_method_validation",
        "convergence_evidence",
        "selection_policy",
        "allowed_refinement_relation",
    }
    if set(contract) != expected_keys:
        raise ValueError(
            "validation target-reference contract has unknown or missing fields"
        )
    if contract.get("schema_version") != "pol-target-reference-contract-v4":
        raise ValueError("unsupported target-reference contract schema")
    system_kind = contract.get("system_kind")
    if system_kind not in {"heat", "burgers", "reaction_diffusion"}:
        raise ValueError("unsupported target-reference system kind")
    invariants = contract.get("invariant_parameters")
    if not isinstance(invariants, Mapping) or stable_object_hash(
        dict(invariants)
    ) != stable_object_hash(
        canonical_invariant_parameters(system_kind, invariants)
    ):
        raise ValueError(
            "validation target-reference invariant parameters are not canonical"
        )
    evolution_time = _positive_finite(
        contract.get("evolution_time"),
        name="target-reference evolution_time",
    )
    _positive_finite(
        contract.get("domain_length"),
        name="target-reference domain_length",
    )
    if contract.get("dtype") not in {"float32", "float64"}:
        raise ValueError("target-reference dtype is unsupported")
    if (
        contract.get("selection_policy")
        != "coarsest_passing_with_finest_pair_required"
    ):
        raise ValueError("unsupported target-reference selection policy")

    reference = contract.get("reference_resolution")
    method = contract.get("numerical_method_validation")
    evidence = contract.get("convergence_evidence")
    relation = contract.get("allowed_refinement_relation")
    if not all(
        isinstance(value, Mapping)
        for value in (reference, method, evidence, relation)
    ):
        raise ValueError("validation target-reference contract is incomplete")
    if set(reference) != {
        "selected_value",
        "selected_candidate_index",
        "finest_value",
        "finest_candidate_index",
        "candidates",
        "candidate_refinement_proof",
    }:
        raise ValueError("reference-resolution contract has invalid fields")
    nx_candidates = reference.get("candidates")
    if not isinstance(nx_candidates, list):
        raise ValueError("reference-resolution candidates must be a list")
    expected_reference_proof = reference_refinement_proof(nx_candidates)
    if stable_object_hash(reference.get("candidate_refinement_proof")) != (
        stable_object_hash(expected_reference_proof)
    ):
        raise ValueError("reference-resolution refinement proof is invalid")

    if set(method) != {
        "kind",
        "selected_condition",
        "selected_candidate_index",
        "finest_condition",
        "finest_candidate_index",
        "candidates",
        "candidate_refinement_proof",
        "temporal_status",
    }:
        raise ValueError("numerical-method validation has invalid fields")
    method_kind = method.get("kind")
    conditions = method.get("candidates")
    if not isinstance(conditions, list) or any(
        not isinstance(value, Mapping) for value in conditions
    ):
        raise ValueError("numerical-method candidates are invalid")
    canonical_conditions = [
        canonical_numerical_condition(
            str(system_kind),
            value,
            evolution_time=evolution_time,
        )
        for value in conditions
    ]
    if stable_object_hash(conditions) != stable_object_hash(
        canonical_conditions
    ):
        raise ValueError("numerical-method candidates are not canonical")
    if method_kind == "analytic_exact":
        if (
            system_kind != "heat"
            or conditions != [{"solver": "spectral_exact"}]
            or method.get("candidate_refinement_proof") is not None
            or method.get("temporal_status") != "analytic_exact"
        ):
            raise ValueError("analytic-exact numerical-method contract is invalid")
    elif method_kind == "candidate_refinement":
        if system_kind not in {"burgers", "reaction_diffusion"}:
            raise ValueError(
                "candidate-refinement numerical-method contract is invalid"
            )
        if system_kind == "burgers":
            expected_time_proof = burgers_refinement_proof(
                conditions,
                evolution_time=evolution_time,
            )
        else:
            expected_time_proof = reaction_diffusion_refinement_proof(
                conditions,
                evolution_time=evolution_time,
            )
        if stable_object_hash(
            method.get("candidate_refinement_proof")
        ) != stable_object_hash(expected_time_proof):
            raise ValueError(
                "candidate refinement proof is invalid"
            )
        if method.get("temporal_status") != "converged":
            raise ValueError("temporal convergence is not passing")
    else:
        raise ValueError("unsupported numerical-method validation kind")

    if set(evidence) != {
        "schema_version",
        "row_schema_version",
        "row_semantics",
        "blank_field_semantics",
        "tolerances",
        "rows",
        "rows_hash",
        "pairwise_row_hashes",
        "joint_row_hash",
        "observed_order_diagnostics",
    }:
        raise ValueError("convergence evidence has invalid fields")
    if (
        evidence.get("schema_version") != CONVERGENCE_CSV_SCHEMA_VERSION
        or evidence.get("row_schema_version")
        != CONVERGENCE_ROW_SCHEMA_VERSION
        or evidence.get("row_semantics")
        != "adjacent_candidate_pairs_plus_selected_vs_finest"
        or evidence.get("blank_field_semantics")
        != {
            "requested_fine_dt": (
                "null/blank exactly for ETDRK4, reaction-diffusion, and "
                "analytic heat conditions"
            ),
            "burgers_step_metadata": (
                "null/blank for analytic heat and reaction-diffusion "
                "conditions"
            ),
            "reaction_diffusion_method_fields": (
                "null/blank for analytic heat and Burgers conditions"
            ),
        }
        or evidence.get("observed_order_diagnostics") != []
    ):
        raise ValueError("unsupported convergence row semantics")
    tolerances = _canonical_tolerances(evidence.get("tolerances"))
    rows = evidence.get("rows")
    if not isinstance(rows, list):
        raise ValueError("convergence evidence rows must be a list")
    spatial_rows = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("check_kind") == "spatial"
    ]
    temporal_rows = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("check_kind") == "temporal"
    ]
    joint_rows = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("check_kind") == "joint"
    ]
    expected_row_count = (
        len(nx_candidates) - 1
        + (len(conditions) - 1 if system_kind != "heat" else 0)
        + 1
    )
    if (
        len(rows) != expected_row_count
        or len(spatial_rows) != len(nx_candidates) - 1
        or len(temporal_rows)
        != (len(conditions) - 1 if system_kind != "heat" else 0)
        or len(joint_rows) != 1
        or rows
        != [*spatial_rows, *temporal_rows, *joint_rows]
    ):
        raise ValueError("convergence rows do not cover the required comparisons")

    finest_reference_index = len(nx_candidates) - 1
    finest_condition_index = len(conditions) - 1
    verified_spatial: list[dict[str, Any]] = []
    for index, row in enumerate(spatial_rows):
        verified_spatial.append(
            _require_expected_row(
                row,
                tolerances=tolerances,
                expected={
                    "check_kind": "spatial",
                    "candidate_axis": "reference_resolution",
                    "coarse_candidate_index": index,
                    "fine_candidate_index": index + 1,
                    "coarse_reference_candidate_index": index,
                    "fine_reference_candidate_index": index + 1,
                    "coarse_nx": nx_candidates[index],
                    "fine_nx": nx_candidates[index + 1],
                    "coarse_condition_index": finest_condition_index,
                    "fine_condition_index": finest_condition_index,
                    "coarse_condition": conditions[
                        finest_condition_index
                    ],
                    "fine_condition": conditions[finest_condition_index],
                    "common_nx": (
                        nx_candidates[index + 1]
                        if system_kind == "heat"
                        else nx_candidates[index]
                    ),
                },
            )
        )
    reference_index = _coarsest_passing_suffix_index(verified_spatial)
    if reference_index is None:
        raise ValueError("finest spatial pair is not passing")

    verified_temporal: list[dict[str, Any]] = []
    for index, row in enumerate(temporal_rows):
        verified_temporal.append(
            _require_expected_row(
                row,
                tolerances=tolerances,
                expected={
                    "check_kind": "temporal",
                    "candidate_axis": "numerical_condition",
                    "coarse_candidate_index": index,
                    "fine_candidate_index": index + 1,
                    "coarse_reference_candidate_index": (
                        finest_reference_index
                    ),
                    "fine_reference_candidate_index": (
                        finest_reference_index
                    ),
                    "coarse_nx": nx_candidates[
                        finest_reference_index
                    ],
                    "fine_nx": nx_candidates[finest_reference_index],
                    "coarse_condition_index": index,
                    "fine_condition_index": index + 1,
                    "coarse_condition": conditions[index],
                    "fine_condition": conditions[index + 1],
                    "common_nx": nx_candidates[finest_reference_index],
                },
            )
        )
    condition_index = (
        0
        if system_kind == "heat"
        else _coarsest_passing_suffix_index(verified_temporal)
    )
    if condition_index is None:
        raise ValueError("finest temporal pair is not passing")

    expected_joint_candidate_indices = (
        (reference_index, finest_reference_index)
        if system_kind == "heat"
        else (condition_index, finest_condition_index)
    )
    verified_joint = _require_expected_row(
        joint_rows[0],
        tolerances=tolerances,
        expected={
            "check_kind": "joint",
            "candidate_axis": "coupled",
            "coarse_candidate_index": (
                expected_joint_candidate_indices[0]
            ),
            "fine_candidate_index": expected_joint_candidate_indices[1],
            "coarse_reference_candidate_index": reference_index,
            "fine_reference_candidate_index": finest_reference_index,
            "coarse_nx": nx_candidates[reference_index],
            "fine_nx": nx_candidates[finest_reference_index],
            "coarse_condition_index": condition_index,
            "fine_condition_index": finest_condition_index,
            "coarse_condition": conditions[condition_index],
            "fine_condition": conditions[finest_condition_index],
            "common_nx": (
                nx_candidates[finest_reference_index]
                if system_kind == "heat"
                else nx_candidates[reference_index]
            ),
        },
    )
    if verified_joint["status"] != "pass":
        raise ValueError("selected-versus-finest joint convergence is not passing")

    if (
        reference.get("selected_candidate_index") != reference_index
        or reference.get("selected_value") != nx_candidates[reference_index]
        or reference.get("finest_candidate_index")
        != finest_reference_index
        or reference.get("finest_value")
        != nx_candidates[finest_reference_index]
    ):
        raise ValueError(
            "selected reference suffix cannot be reconstructed from rows"
        )
    if (
        method.get("selected_candidate_index") != condition_index
        or stable_object_hash(method.get("selected_condition"))
        != stable_object_hash(conditions[condition_index])
        or method.get("finest_candidate_index") != finest_condition_index
        or stable_object_hash(method.get("finest_condition"))
        != stable_object_hash(conditions[finest_condition_index])
    ):
        raise ValueError(
            "selected numerical-condition suffix cannot be reconstructed from rows"
        )

    pairwise_hashes = [
        row["row_hash"] for row in [*spatial_rows, *temporal_rows]
    ]
    if (
        evidence.get("rows_hash") != stable_object_hash(rows)
        or evidence.get("pairwise_row_hashes") != pairwise_hashes
        or evidence.get("joint_row_hash") != joint_rows[0]["row_hash"]
    ):
        raise ValueError("convergence row hashes are inconsistent")

    reference_indices = list(range(reference_index, len(nx_candidates)))
    condition_indices = list(range(condition_index, len(conditions)))
    expected_relation = {
        "kind": "validated_candidate_suffix_exact_membership",
        "reference_nx_allowed_indices": reference_indices,
        "reference_nx_allowed_values": [
            nx_candidates[index] for index in reference_indices
        ],
        "numerical_condition_allowed_indices": condition_indices,
        "numerical_condition_allowed_values": [
            conditions[index] for index in condition_indices
        ],
    }
    if stable_object_hash(dict(relation)) != stable_object_hash(
        expected_relation
    ):
        raise ValueError(
            "validation allowed refinement relation is not the reconstructed "
            "candidate suffix"
        )


__all__ = [
    "CONVERGENCE_CSV_SCHEMA_VERSION",
    "CONVERGENCE_ROW_FIELDS",
    "CONVERGENCE_ROW_SCHEMA_VERSION",
    "CROSS_SOLVER_CHECK_SCHEMA_VERSION",
    "CROSS_SOLVER_METRIC_DEFINITION",
    "burgers_refinement_proof",
    "canonical_invariant_parameters",
    "canonical_numerical_condition",
    "canonical_solver_name",
    "cross_solver_discrepancy_evidence_hash",
    "make_convergence_row",
    "reaction_diffusion_refinement_proof",
    "reference_refinement_proof",
    "validate_cross_solver_validation_block",
    "validate_target_reference_contract",
]
