from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
import pytest

from pol.config.loader import load_dataset_spec, load_validation_spec
from pol.data.dataset import ensure_dataset, load_dataset
from pol.data.finite import build_feature_initial_state, derive_finite_view
from pol.data.initial_conditions import generate_grf_archive
from pol.data.splits import build_data_split
from pol.math.periodic import periodic_grid, spectral_resample_periodic
from pol.numerics.initial_conditions import GRF_SAMPLER_SEMANTICS
from pol.validation.model1_consistency import (
    MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION,
)
from pol.validation.quadrature import (
    FIELD_QUADRATURE_CHECK_SCHEMA_VERSION,
    run_field_quadrature_check,
)
from pol.validation.runner import ensure_validation, load_validation_certificate
from pol.runtime.artifacts import manifest_records
from pol.runtime.hashing import stable_object_hash
from pol.runtime.io import write_strict_json
from pol.artifacts import verify_artifact
from tests.helpers import write_json, write_tiny_heat_stack, write_tiny_stack


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh_artifact_record(root: Path, relative_path: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    for index, record in enumerate(manifest["files"]):
        if record["relative_path"] == relative_path:
            manifest["files"][index] = manifest_records(
                root,
                [relative_path],
            )[0]
            break
    else:
        raise AssertionError(f"manifest has no record for {relative_path}")
    write_strict_json(manifest_path, manifest)


def _enable_tiny_cross_solver(raw: dict[str, object]) -> None:
    raw["target_reference"]["cross_solver_validation"] = {
        "schema_version": "pol-burgers-cross-solver-spec-v1",
        "enabled": True,
        "context": {
            "system_kind": "burgers",
            "nu": 0.05,
            "advection_coefficient": 1.0,
            "final_time": 0.02,
            "domain_length": 1.0,
            "dtype": "float64",
            "dealias": True,
        },
        "solvers": {
            "split_step": {
                "candidates": [
                    {
                        "solver": "split_step",
                        "dt": 0.01,
                        "fine_dt": 0.005,
                        "dealias": True,
                    },
                    {
                        "solver": "split_step",
                        "dt": 0.01,
                        "fine_dt": 0.0025,
                        "dealias": True,
                    },
                ]
            },
            "etdrk4": {
                "candidates": [
                    {
                        "solver": "etdrk4",
                        "dt": 0.01,
                        "fine_dt": None,
                        "dealias": True,
                    },
                    {
                        "solver": "etdrk4",
                        "dt": 0.005,
                        "fine_dt": None,
                        "dealias": True,
                    },
                ]
            },
        },
        "tolerances": {
            "mean_relative_l2": 1.0,
            "max_relative_l2": 1.0,
            "low_mode_relative_l2": 1.0,
        },
    }


def _tiny_cross_solver_outcome(tmp_path: Path):
    validation_path, _, _ = write_tiny_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    _enable_tiny_cross_solver(raw)
    write_json(validation_path, raw)
    spec = load_validation_spec(validation_path, repo_root=tmp_path)
    return spec, ensure_validation(spec)


def _tiny_reaction_diffusion_spec(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    raw = json.loads(
        (
            repo_root
            / "configs/validation/reaction_diffusion_smoke.json"
        ).read_text(encoding="utf-8")
    )
    raw["name"] = "tiny_reaction_diffusion_validation"
    raw["artifact_root"] = str(tmp_path / "artifacts")
    path = write_json(tmp_path / "reaction_validation.json", raw)
    return load_validation_spec(path, repo_root=tmp_path)


def test_foundation_validation_publishes_passing_certificate(tmp_path: Path) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    spec = load_validation_spec(validation_path, repo_root=tmp_path)
    outcome = ensure_validation(spec)
    assert outcome.certificate["status"] == "pass"
    checks = json.loads((outcome.reference.path / "checks.json").read_text(encoding="utf-8"))
    assert checks["finite_input_interface"]["dimension_independence"]["n_tar_le_J_exercised"]
    zero_fill = checks["fixed_decoder"]["zero_fill_characterization"]
    assert zero_fill["status"] == "pass"
    assert zero_fill["requested_q"] == 7
    assert zero_fill["observable_q"] == 3
    assert zero_fill["retained_q"] == 3
    assert zero_fill["zero_filled_coefficient_count"] == 4
    assert zero_fill["zero_filled_mode_count"] == 2
    assert zero_fill["zero_filled_coefficient_index_range"] == {
        "start_inclusive": 3,
        "stop_exclusive": 7,
    }
    assert zero_fill["zero_filled_mode_range"] == {
        "start_inclusive": 2,
        "stop_inclusive": 3,
    }
    assert zero_fill["observable_part"]["status"] == "pass"
    assert zero_fill["zero_filled_part"] == {
        "status": "pass",
        "exact_zero": True,
        "coefficient_count": 4,
    }
    assert (
        outcome.certificate["foundation_contract"][
            "fixed_decoder_bandwidth_contract"
        ]
        == zero_fill
    )
    matched = checks["matched_model1_pipeline"]
    assert matched["schema_version"] == (
        MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION
    )
    assert matched["status"] == "pass"
    assert matched["case_count"] == 8
    assert matched["positive_case_count"] == 7
    assert matched["negative_control_case_count"] == 1
    assert all(case["status"] == "pass" for case in matched["cases"])
    foundation_matched = outcome.certificate["foundation_contract"][
        "matched_model1_pipeline_contract"
    ]
    assert foundation_matched["status"] == "pass"
    assert foundation_matched["detailed_cases_hash"] == matched["cases_hash"]
    assert foundation_matched["case_count"] == matched["case_count"]
    assert foundation_matched["check_hash"] == stable_object_hash(matched)
    quadrature = checks["field_quadrature"]
    assert quadrature["schema_version"] == (
        FIELD_QUADRATURE_CHECK_SCHEMA_VERSION
    )
    assert quadrature["status"] == "pass"
    assert quadrature["convergence"]["candidate_n_ref"] == [
        8,
        15,
        16,
        31,
        32,
    ]
    assert quadrature["convergence"]["selected_n_ref"] == 15
    assert quadrature["convergence"]["allowed_suffix_n_ref"] == [
        15,
        16,
        31,
        32,
    ]
    foundation_quadrature = outcome.certificate["foundation_contract"][
        "field_quadrature_contract"
    ]
    assert foundation_quadrature["status"] == "pass"
    assert foundation_quadrature["selected_n_ref"] == 15
    assert foundation_quadrature["check_hash"] == stable_object_hash(
        quadrature
    )
    assert foundation_quadrature[
        "metric_wrapper_consistency_status"
    ] == "pass"
    assert foundation_quadrature[
        "data_space_invariance_status"
    ] == "pass"
    assert foundation_quadrature[
        "representation_floor_consistency_status"
    ] == "pass"
    calibration = outcome.certificate["foundation_contract"][
        "calibration_provenance"
    ]
    assert calibration["calibration_sample_ids"] == [0, 1]
    assert calibration["calibration_split_membership"] == {
        "0": "train",
        "1": "train",
    }
    assert calibration["calibration_test_overlap_count"] == 0
    assert calibration["split_policy"] == "cpu_torch_randperm"
    assert calibration["split_policy_version"] == 1
    assert calibration["split_hash"] == (
        "80e69dd30b1caa4acae41729789c90449c8749292a6ceb85680949656dd503e1"
    )
    assert (outcome.reference.path / "master_initial_conditions.pt").is_file()


def test_common_foundation_checks_execute_once_for_each_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pol.validation.runner as validation_runner

    original = validation_runner.run_foundation_checks
    calls: list[str] = []

    def counted(spec, archive):
        calls.append(spec.target_reference.kind)
        return original(spec, archive)

    monkeypatch.setattr(
        validation_runner,
        "run_foundation_checks",
        counted,
    )
    burgers_path, _, _ = write_tiny_stack(tmp_path / "burgers")
    heat_path, _, _ = write_tiny_heat_stack(tmp_path / "heat")
    specs = (
        load_validation_spec(
            burgers_path,
            repo_root=tmp_path / "burgers",
        ),
        load_validation_spec(
            heat_path,
            repo_root=tmp_path / "heat",
        ),
        _tiny_reaction_diffusion_spec(tmp_path / "reaction_diffusion"),
    )
    for spec in specs:
        ensure_validation(spec)
    assert calls == [
        "burgers_convergence",
        "heat_analytic",
        "reaction_diffusion_convergence",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("coefficient_max_abs_error", 1.0),
        ("tolerance", 1.0),
        ("status", "fail"),
    ],
)
def test_certificate_loader_rejects_matched_model1_case_summary_tamper(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    certificate_path = outcome.reference.path / "certificate.json"
    certificate = _read_json(certificate_path)
    foundation = certificate["foundation_contract"]
    contract = foundation["matched_model1_pipeline_contract"]
    contract["case_summaries"][0][field] = value
    contract["case_summaries_hash"] = stable_object_hash(
        contract["case_summaries"]
    )
    certificate["foundation_contract_hash"] = stable_object_hash(
        foundation
    )
    write_strict_json(certificate_path, certificate)
    _refresh_artifact_record(outcome.reference.path, "certificate.json")

    with pytest.raises(ValueError, match="certificate contract"):
        load_validation_certificate(outcome.reference.path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("coefficient_max_abs_error", 1.0),
        ("tolerance", 1.0),
        ("status", "fail"),
    ],
)
def test_certificate_loader_rejects_matched_model1_detailed_case_tamper(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    checks_path = outcome.reference.path / "checks.json"
    checks = _read_json(checks_path)
    matched = checks["matched_model1_pipeline"]
    matched["cases"][0][field] = value
    matched["cases_hash"] = stable_object_hash(matched["cases"])
    write_strict_json(checks_path, checks)
    _refresh_artifact_record(outcome.reference.path, "checks.json")

    with pytest.raises(ValueError, match="matched Model 1 pipeline check"):
        load_validation_certificate(outcome.reference.path)


def test_certificate_loader_rejects_pre_field_quadrature_certificate(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    certificate_path = outcome.reference.path / "certificate.json"
    certificate = _read_json(certificate_path)
    certificate["schema_version"] = "pol-validation-certificate-v11"
    write_strict_json(certificate_path, certificate)
    _refresh_artifact_record(outcome.reference.path, "certificate.json")

    with pytest.raises(ValueError, match="Phase 2-05B requires"):
        load_validation_certificate(outcome.reference.path)


@pytest.mark.parametrize(
    "tamper",
    [
        "candidate_order",
        "selected_grid",
        "allowed_suffix",
        "tolerance",
        "status",
    ],
)
def test_certificate_loader_rejects_field_quadrature_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    checks_path = outcome.reference.path / "checks.json"
    checks = _read_json(checks_path)
    quadrature = checks["field_quadrature"]
    convergence = quadrature["convergence"]
    if tamper == "candidate_order":
        convergence["candidate_n_ref"] = [8, 16, 15, 31, 32]
    elif tamper == "selected_grid":
        convergence["selected_n_ref"] = 16
    elif tamper == "allowed_suffix":
        convergence["allowed_suffix_n_ref"] = [16, 31, 32]
    elif tamper == "tolerance":
        convergence["tolerance"] = 1.0
    elif tamper == "status":
        quadrature["status"] = "fail"
    else:
        raise AssertionError(tamper)
    quadrature["convergence_hash"] = stable_object_hash(convergence)
    write_strict_json(checks_path, checks)
    _refresh_artifact_record(outcome.reference.path, "checks.json")

    with pytest.raises(ValueError, match="field quadrature"):
        load_validation_certificate(outcome.reference.path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selected_n_ref", 16),
        ("allowed_suffix_n_ref", [16, 31, 32]),
        ("tolerance", 1.0),
        ("status", "fail"),
        ("check_hash", "0" * 64),
    ],
)
def test_certificate_loader_rejects_field_quadrature_foundation_tamper(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    certificate_path = outcome.reference.path / "certificate.json"
    certificate = _read_json(certificate_path)
    foundation = certificate["foundation_contract"]
    foundation["field_quadrature_contract"][field] = value
    certificate["foundation_contract_hash"] = stable_object_hash(
        foundation
    )
    write_strict_json(certificate_path, certificate)
    _refresh_artifact_record(outcome.reference.path, "certificate.json")

    with pytest.raises(ValueError, match="certificate contract"):
        load_validation_certificate(outcome.reference.path)


def test_field_quadrature_check_is_synthetic_and_profile_independent() -> None:
    first = run_field_quadrature_check(domain_length=1.0)
    second = run_field_quadrature_check(domain_length=1.0)
    assert first == second
    assert first["metric_wrapper_consistency"][
        "same_prediction_all_reference_grids"
    ] is True
    assert first["metric_wrapper_consistency"][
        "same_continuous_target_all_reference_grids"
    ] is True
    assert first["metric_wrapper_consistency"][
        "only_reference_quadrature_grid_changes"
    ] is True


def test_reaction_diffusion_characterization_and_convergence_certificate_pass(
    tmp_path: Path,
) -> None:
    outcome = ensure_validation(_tiny_reaction_diffusion_spec(tmp_path))
    checks = json.loads(
        (outcome.reference.path / "checks.json").read_text(
            encoding="utf-8"
        )
    )
    characterization = checks["reaction_diffusion_characterization"]
    assert characterization["status"] == "pass"
    assert characterization["zero_equilibrium"]["exact_zero"] is True
    assert {
        case["grid_parity"]
        for case in characterization["constant_scalar_recurrence"]
    } == {"odd", "even"}
    assert {
        case["nonlinear_filter"]
        for case in characterization["constant_scalar_recurrence"]
    } == {"none", "two_thirds"}
    assert all(
        case["status"] == "pass"
        for case in characterization["constant_scalar_recurrence"]
    )
    assert all(
        case["status"] == "pass"
        for case in characterization["nonzero_equilibria"]
    )
    assert all(
        case["beta"] == 0.0 and case["status"] == "pass"
        for case in characterization["beta_zero_linear_modes"]
    )

    contract = outcome.certificate["target_reference_contract"]
    method = contract["numerical_method_validation"]
    rows = contract["convergence_evidence"]["rows"]
    assert contract["system_kind"] == "reaction_diffusion"
    assert contract["invariant_parameters"] == {
        "nu": 0.05,
        "alpha": 1.0,
        "beta": 1.0,
    }
    assert [row["check_kind"] for row in rows] == [
        "spatial",
        "temporal",
        "temporal",
        "joint",
    ]
    assert all(
        math.isfinite(row[field])
        for row in rows
        for field in (
            "mean_relative_l2",
            "max_relative_l2",
            "low_mode_relative_l2",
        )
    )
    assert method["selected_candidate_index"] == 1
    assert method["finest_candidate_index"] == 2
    assert method["selected_condition"] == {
        "solver": "semi_implicit_spectral_euler",
        "dt": 0.0025,
        "nonlinear_filter": "two_thirds",
    }
    assert contract["allowed_refinement_relation"][
        "numerical_condition_allowed_indices"
    ] == [1, 2]
    assert rows[-1]["coarse_dt"] == 0.0025
    assert rows[-1]["fine_dt"] == 0.00125
    assert rows[-1]["coarse_nonlinear_filter"] == "two_thirds"
    assert rows[-1]["fine_nonlinear_filter"] == "two_thirds"
    assert rows[-1]["status"] == "pass"


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/validation/heat_smoke.json",
        "configs/validation/foundation_smoke.json",
        "configs/validation/reaction_diffusion_smoke.json",
    ],
)
def test_checked_in_smoke_validations_include_matched_model1_pipeline(
    tmp_path: Path,
    relative_path: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw = _read_json(repo_root / relative_path)
    raw["artifact_root"] = str(
        tmp_path / Path(relative_path).stem / "artifacts"
    )
    spec_path = write_json(
        tmp_path / f"{Path(relative_path).stem}.json",
        raw,
    )
    outcome = ensure_validation(
        load_validation_spec(spec_path, repo_root=repo_root)
    )
    contract = outcome.certificate["foundation_contract"][
        "matched_model1_pipeline_contract"
    ]
    assert outcome.certificate["checks"][
        "matched_model1_pipeline"
    ] == "pass"
    assert contract["status"] == "pass"
    assert contract["case_count"] == 8
    quadrature = outcome.certificate["foundation_contract"][
        "field_quadrature_contract"
    ]
    assert outcome.certificate["checks"]["field_quadrature"] == "pass"
    assert quadrature["status"] == "pass"
    assert quadrature["selected_n_ref"] == 15


def test_reaction_diffusion_nonfinite_solve_publishes_verified_failure_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _tiny_reaction_diffusion_spec(tmp_path)

    def nonfinite_evolve(values, evolution, *, domain_length):
        return torch.full_like(values, torch.inf), {}

    monkeypatch.setattr(
        "pol.validation.reference_convergence.evolve",
        nonfinite_evolve,
    )
    with pytest.raises(RuntimeError, match="non-finite state"):
        ensure_validation(spec)
    roots = list(
        (tmp_path / "artifacts" / "validation_failures").iterdir()
    )
    assert len(roots) == 1
    manifest = verify_artifact(roots[0])
    assert manifest["kind"] == "validation_failures"
    failure = json.loads(
        (roots[0] / "failure.json").read_text(encoding="utf-8")
    )
    assert failure["schema_version"] == "pol-validation-failure-v1"
    assert failure["status"] == "fail"
    assert failure["diagnostic"]["failure_kind"] == (
        "nonfinite_solver_state"
    )
    assert failure["diagnostic"]["system_kind"] == (
        "reaction_diffusion"
    )
    assert failure["diagnostic_hash"] == stable_object_hash(
        failure["diagnostic"]
    )


def test_burgers_convergence_rows_are_complete_finite_and_reconstructable(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    convergence = outcome.certificate["target_reference_contract"][
        "convergence_evidence"
    ]
    rows = convergence["rows"]
    assert [row["check_kind"] for row in rows] == [
        "spatial",
        "temporal",
        "joint",
    ]
    assert all(
        math.isfinite(row[name])
        for row in rows
        for name in (
            "mean_relative_l2",
            "max_relative_l2",
            "low_mode_relative_l2",
        )
    )
    assert all(
        row["row_hash"]
        == stable_object_hash(
            {key: value for key, value in row.items() if key != "row_hash"}
        )
        for row in rows
    )
    temporal = rows[1]
    assert temporal["coarse_requested_outer_dt"] == 0.01
    assert temporal["fine_requested_outer_dt"] == 0.01
    assert temporal["coarse_requested_fine_dt"] == 0.005
    assert temporal["fine_requested_fine_dt"] == 0.0025
    assert temporal["coarse_effective_substep"] == 0.005
    assert temporal["fine_effective_substep"] == 0.0025
    assert temporal["coarse_substeps_per_outer"] == 2
    assert temporal["fine_substeps_per_outer"] == 4
    contract = outcome.certificate["target_reference_contract"]
    assert contract["reference_resolution"]["selected_candidate_index"] == 0
    assert contract["numerical_method_validation"][
        "selected_candidate_index"
    ] == 0
    assert contract["allowed_refinement_relation"][
        "reference_nx_allowed_indices"
    ] == [0, 1]
    assert contract["allowed_refinement_relation"][
        "numerical_condition_allowed_indices"
    ] == [0, 1]


def test_cross_solver_evidence_has_independent_self_rows_and_symmetric_metrics(
    tmp_path: Path,
) -> None:
    _, outcome = _tiny_cross_solver_outcome(tmp_path)
    certificate = outcome.certificate
    block = certificate["cross_solver_validation"]
    checks = json.loads(
        (outcome.reference.path / "checks.json").read_text(encoding="utf-8")
    )
    assert block == checks["cross_solver_validation"]
    assert certificate["cross_solver_validation_hash"] == (
        stable_object_hash(block)
    )
    assert block["status"] == block["discrepancy_status"] == "pass"
    assert block["role"] == (
        "supporting_evidence_not_primary_allowed_refinement"
    )
    assert block["context"]["sample_ids"] == [0, 1]
    assert block["context"]["common_nx"] == 32
    for family in ("split_step", "etdrk4"):
        evidence = block["self_convergence"][family]
        assert evidence["status"] == "pass"
        assert len(evidence["rows"]) == 1
        assert evidence["rows_hash"] == stable_object_hash(evidence["rows"])
        assert evidence["pairwise_row_hashes"] == [
            evidence["rows"][0]["row_hash"]
        ]
        assert all(
            math.isfinite(evidence["rows"][0][name])
            for name in (
                "mean_relative_l2",
                "max_relative_l2",
                "low_mode_relative_l2",
            )
        )
    split_row = block["self_convergence"]["split_step"]["rows"][0]
    etd_row = block["self_convergence"]["etdrk4"]["rows"][0]
    assert (
        split_row["coarse_requested_outer_dt"],
        split_row["coarse_requested_fine_dt"],
        split_row["coarse_effective_substep"],
        split_row["coarse_substeps_per_outer"],
    ) == (0.01, 0.005, 0.005, 2)
    assert (
        etd_row["fine_requested_outer_dt"],
        etd_row["fine_requested_fine_dt"],
        etd_row["fine_effective_substep"],
        etd_row["fine_substeps_per_outer"],
    ) == (0.005, None, 0.005, 1)
    assert set(block["discrepancy_metrics"]) == {
        "mean_absolute_l2",
        "max_absolute_l2",
        "mean_relative_l2",
        "max_relative_l2",
        "low_mode_relative_l2",
    }
    assert all(
        math.isfinite(value)
        for value in block["discrepancy_metrics"].values()
    )
    primary = certificate["target_reference_contract"]
    assert {
        condition["solver"]
        for condition in primary["allowed_refinement_relation"][
            "numerical_condition_allowed_values"
        ]
    } == {"split_step"}


def test_cross_solver_validation_is_deterministic_on_rerun(
    tmp_path: Path,
) -> None:
    spec, first = _tiny_cross_solver_outcome(tmp_path)
    second = ensure_validation(spec, force=True)
    assert (
        first.certificate["cross_solver_validation"]
        == second.certificate["cross_solver_validation"]
    )


def test_disabled_cross_solver_validation_performs_no_cross_solve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    spec = load_validation_spec(validation_path, repo_root=tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled cross-solver diagnostic was executed")

    monkeypatch.setattr(
        "pol.validation.burgers_reference._burgers_cross_solver_validation",
        forbidden,
    )
    outcome = ensure_validation(spec)
    checks = json.loads(
        (outcome.reference.path / "checks.json").read_text(encoding="utf-8")
    )
    assert "cross_solver_validation" not in checks
    assert outcome.certificate["cross_solver_validation"] is None
    assert outcome.certificate["cross_solver_validation_hash"] is None


def test_cross_solver_comparison_waits_for_both_self_convergence_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    _enable_tiny_cross_solver(raw)
    raw["target_reference"]["cross_solver_validation"]["tolerances"] = {
        "mean_relative_l2": 0.0,
        "max_relative_l2": 0.0,
        "low_mode_relative_l2": 0.0,
    }
    write_json(validation_path, raw)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "cross comparison ran before self-convergence passed"
        )

    monkeypatch.setattr(
        "pol.validation.burgers_reference.symmetric_field_discrepancy",
        forbidden,
    )
    with pytest.raises(RuntimeError, match="cross_solver_validation.*fail"):
        ensure_validation(
            load_validation_spec(validation_path, repo_root=tmp_path)
        )


def test_burgers_finest_pair_failure_fails_overall_validation(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    raw["target_reference"]["reference_tolerances"] = {
        "mean_relative_l2": 0.0,
        "max_relative_l2": 0.0,
        "low_mode_relative_l2": 0.0,
    }
    validation_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="reference_convergence.*fail"):
        ensure_validation(
            load_validation_spec(validation_path, repo_root=tmp_path)
        )


def test_burgers_reference_convergence_supports_odd_even_adjacent_grids(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    raw["target_reference"]["reference_nx_candidates"] = [15, 16]
    raw["full_interface"]["n_tar"] = 15
    validation_path.write_text(json.dumps(raw), encoding="utf-8")
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    row = outcome.certificate["target_reference_contract"][
        "convergence_evidence"
    ]["rows"][0]
    assert (row["coarse_nx"], row["fine_nx"], row["common_nx"]) == (
        15,
        16,
        15,
    )


def test_heat_validation_separates_analytic_and_spatial_claims(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_heat_stack(tmp_path)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    checks = json.loads(
        (outcome.reference.path / "checks.json").read_text(encoding="utf-8")
    )
    analytic = checks["heat_analytic"]
    convergence = checks["reference_convergence"]
    assert analytic["status"] == "pass"
    assert analytic["temporal_status"] == "analytic_exact"
    assert convergence["status"] == "pass"
    assert convergence["spatial_status"] == "pass"
    assert convergence["temporal_status"] == "analytic_exact"

    cases = analytic["cases"]
    assert len(cases) == 7
    assert {case["dtype"] for case in cases} == {"float32", "float64"}
    assert {case["nx"] % 2 for case in cases} == {0, 1}
    assert any(case["domain_length"] != 1.0 for case in cases)
    assert {case["basis"] for case in cases} >= {
        "constant",
        "cosine",
        "sine",
        "constant_plus_sine_cosine",
        "nyquist_cosine_unpaired",
    }
    assert all(case["status"] == "pass" for case in cases)
    assert all(case["finite_status"] == "pass" for case in cases)

    contract = outcome.certificate["target_reference_contract"]
    method = contract["numerical_method_validation"]
    relation = contract["allowed_refinement_relation"]
    assert contract["system_kind"] == "heat"
    assert method == {
        "kind": "analytic_exact",
        "selected_condition": {"solver": "spectral_exact"},
        "selected_candidate_index": 0,
        "finest_condition": {"solver": "spectral_exact"},
        "finest_candidate_index": 0,
        "candidates": [{"solver": "spectral_exact"}],
        "candidate_refinement_proof": None,
        "temporal_status": "analytic_exact",
    }
    assert "time_discretization" not in contract
    assert relation["reference_nx_allowed_values"] == [16, 32]
    assert relation["numerical_condition_allowed_values"] == [
        {"solver": "spectral_exact"}
    ]
    with (
        outcome.reference.path / "reference_convergence.csv"
    ).open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert all(row["check_kind"] != "temporal" for row in csv_rows)
    assert all(
        row[field] == ""
        for row in csv_rows
        for field in (
            "coarse_requested_outer_dt",
            "coarse_requested_fine_dt",
            "coarse_effective_substep",
            "fine_requested_outer_dt",
            "fine_requested_fine_dt",
            "fine_effective_substep",
        )
    )


def test_grf_archive_records_the_sampler_domain_and_preserves_dtype_device() -> None:
    archive = generate_grf_archive(
        total_samples=2,
        nx=9,
        seed=23,
        gamma=2.0,
        tau=5.0,
        sigma=1.0,
        mean=0.0,
        domain_length=2.0,
        dtype="float32",
        device="cpu",
    )
    assert archive.domain_length == archive.metadata["domain_length"] == 2.0
    assert archive.metadata["sampler_semantics"] == GRF_SAMPLER_SEMANTICS
    assert archive.values.dtype == torch.float32
    assert archive.values.device == torch.device("cpu")
    assert torch.equal(
        archive.fourier,
        torch.fft.rfft(archive.values, dim=-1, norm="forward"),
    )


def test_nonunit_domain_is_bound_across_grf_archive_and_certificate(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    raw["domain"]["length"] = 2.0
    write_json(validation_path, raw)
    spec = load_validation_spec(validation_path, repo_root=tmp_path)
    outcome = ensure_validation(spec)
    root = outcome.reference.path
    resolved = json.loads(
        (root / "resolved_spec.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    master = torch.load(
        root / "master_initial_conditions.pt",
        map_location="cpu",
        weights_only=True,
    )
    foundation = outcome.certificate["foundation_contract"]
    master_binding = foundation["master_initial_conditions"]

    assert manifest["identity"]["schema_version"] == "pol-validation-identity-v12"
    assert master["schema_version"] == "pol-initial-condition-archive-v4"
    assert outcome.certificate["schema_version"] == "pol-validation-certificate-v12"
    assert foundation["schema_version"] == "pol-validation-foundation-contract-v8"
    assert master_binding["schema_version"] == (
        "pol-master-initial-condition-binding-v3"
    )
    assert resolved["domain"]["length"] == 2.0
    assert manifest["identity"]["spec"]["domain"]["length"] == 2.0
    assert master["domain_length"] == 2.0
    assert master["metadata"]["domain_length"] == 2.0
    assert foundation["domain_length"] == 2.0
    assert foundation["grf_sampler_domain_length"] == 2.0
    assert master_binding["domain_length"] == 2.0
    assert master_binding["metadata"]["domain_length"] == 2.0
    assert {
        manifest["identity"]["grf_sampler_semantics"],
        master["metadata"]["sampler_semantics"],
        foundation["grf_sampler_semantics"],
        master_binding["metadata"]["sampler_semantics"],
    } == {GRF_SAMPLER_SEMANTICS}


def test_reusable_split_preserves_pre_gate_dataset_ids_and_hash() -> None:
    split = build_data_split(
        total_samples=12,
        n_train=8,
        n_validation=2,
        n_test=2,
        seed=17,
    )
    assert torch.equal(
        split.train_ids,
        torch.tensor([3, 10, 0, 7, 1, 11, 6, 4], dtype=torch.long),
    )
    assert torch.equal(
        split.validation_ids,
        torch.tensor([2, 9], dtype=torch.long),
    )
    assert torch.equal(
        split.test_ids,
        torch.tensor([8, 5], dtype=torch.long),
    )
    assigned = torch.cat(
        (split.train_ids, split.validation_ids, split.test_ids)
    )
    assert torch.equal(torch.sort(assigned).values, torch.arange(12))
    assert torch.unique(assigned).numel() == 12
    assert split.split_hash(torch.arange(12, dtype=torch.long)) == (
        "80e69dd30b1caa4acae41729789c90449c8749292a6ceb85680949656dd503e1"
    )


def test_calibration_test_overlap_fails_before_compute_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    raw["target_reference"]["calibration_sample_ids"] = [0, 8]
    write_json(validation_path, raw)
    spec = load_validation_spec(validation_path, repo_root=tmp_path)
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("called")
        raise AssertionError("validation compute/publication must not start")

    monkeypatch.setattr(
        "pol.validation.runner.generate_grf_archive",
        forbidden,
    )
    monkeypatch.setattr(
        "pol.validation.reference_convergence.evolve",
        forbidden,
    )
    monkeypatch.setattr(
        "pol.validation.runner.ArtifactStore.publish",
        forbidden,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"offending calibration IDs=\[8\].*split seed=17.*"
            r"train/validation/test counts=8/2/2.*"
            r"must belong to train or validation"
        ),
    ):
        ensure_validation(spec)
    assert calls == []
    assert not (tmp_path / "artifacts").exists()


def test_train_and_validation_calibration_ids_are_accepted(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    raw["target_reference"]["calibration_sample_ids"] = [0, 2]
    write_json(validation_path, raw)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    provenance = outcome.certificate["foundation_contract"][
        "calibration_provenance"
    ]
    assert provenance["calibration_split_membership"] == {
        "0": "train",
        "2": "validation",
    }
    assert provenance["calibration_test_overlap_count"] == 0


def test_reference_dataset_reuses_validation_and_has_disjoint_splits(tmp_path: Path) -> None:
    _, dataset_path, _ = write_tiny_stack(tmp_path)
    spec = load_dataset_spec(dataset_path, repo_root=tmp_path)
    dataset = ensure_dataset(spec, repo_root=tmp_path)
    reloaded = load_dataset(dataset.path)
    train = set(reloaded.train_ids.tolist())
    validation = set(reloaded.validation_ids.tolist())
    test = set(reloaded.test_ids.tolist())
    assert train.isdisjoint(validation) and train.isdisjoint(test) and validation.isdisjoint(test)
    assert train | validation | test == set(reloaded.sample_ids.tolist())
    assert reloaded.train_ids.tolist() == [3, 10, 0, 7, 1, 11, 6, 4]
    assert reloaded.validation_ids.tolist() == [2, 9]
    assert reloaded.test_ids.tolist() == [8, 5]
    assert reloaded.split_hash == (
        "80e69dd30b1caa4acae41729789c90449c8749292a6ceb85680949656dd503e1"
    )
    assert reloaded.inputs_reference.shape == reloaded.targets_reference.shape == (12, 32)
    assert torch.isfinite(reloaded.targets_reference).all()


def test_finite_data_resolution_cannot_exceed_validated_reference() -> None:
    values = torch.zeros(2, 16, dtype=torch.float64)
    with pytest.raises(ValueError, match="must not exceed"):
        derive_finite_view(
            torch.tensor([0, 1], dtype=torch.long),
            values,
            values,
            n_tar=32,
            q=9,
            domain_length=1.0,
        )


def test_discarded_reference_modes_do_not_reach_feature_initial_state() -> None:
    x = periodic_grid(64, 1.0)
    low = 0.3 + torch.cos(4.0 * torch.pi * x)
    differing_reference = low + 0.25 * torch.cos(40.0 * torch.pi * x)
    finite = spectral_resample_periodic(
        torch.stack([low, differing_reference]), 16, domain_length=1.0
    )
    feature_initial = build_feature_initial_state(
        finite, n_sur=32, domain_length=1.0
    )
    assert torch.allclose(finite[0], finite[1], atol=1e-11, rtol=1e-11)
    assert torch.allclose(
        feature_initial[0], feature_initial[1], atol=1e-11, rtol=1e-11
    )
