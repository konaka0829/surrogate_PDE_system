from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from pol.config.loader import (
    load_dataset_spec,
    load_study_spec,
    load_validation_spec,
)
from pol.config.models import GridSearchSpec, MetricMapReporterSpec
from pol.plotting.reporters import build_metric_map_data
from pol.runtime.hashing import stable_object_hash
from pol.study.cases import build_cases, plan_study
from pol.study.evaluation import CandidateEvaluation
from pol.study.protocol import build_selection_cases
from pol.study.results import validation_result_rows
from pol.study.runner import run_study, verify_study_run
from pol.study.search import run_search
from tests.helpers import write_json, write_tiny_stack


class _ValidationOnlyRows(dict[str, object]):
    def __getitem__(self, key: str) -> object:
        if key.startswith("test_"):
            raise AssertionError("selection must not access test metrics")
        return super().__getitem__(key)


class _SelectionStub:
    def __init__(self, *, tied: bool = False) -> None:
        self.tied = tied

    def evaluate_selection(self, trial) -> CandidateEvaluation:
        candidate_id = stable_object_hash(trial.model_dump(mode="json"))
        evolution = trial.feature.evolution
        assert evolution is not None
        metric = (
            1.0
            if self.tied
            else float(evolution.system.nu) + float(evolution.time)
        )
        rows = {
            readout.id: _ValidationOnlyRows(
                {
                    "candidate_id": candidate_id,
                    "readout_id": readout.id,
                    "readout_kind": readout.kind,
                    "validation_field_relative_l2_mean": metric,
                    "test_forbidden": -metric,
                }
            )
            for readout in trial.readouts
        }
        return CandidateEvaluation(
            candidate_id=candidate_id,
            trial=trial,
            rows=rows,
            selection_models={
                readout.id: {"kind": readout.kind}
                for readout in trial.readouts
            },
            inner_selections={
                readout.id: {"selection": "validation"}
                for readout in trial.readouts
            },
            feature_cache_id=f"feature-{candidate_id}",
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tiny_grid_study(tmp_path: Path) -> Path:
    _, _, study_path = write_tiny_stack(
        tmp_path,
        include_diagnostics=False,
    )
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["name"] = "tiny_parameter_time_landscape"
    raw["profile"] = "smoke"
    raw["variants"][0]["search"] = {
        "kind": "grid",
        "axes": [
            {
                "path": "feature.evolution.system.nu",
                "values": [0.05, 0.1],
            },
            {
                "path": "feature.evolution.time",
                "values": [0.05, 0.1],
            },
        ],
    }
    raw["reporters"] = []
    return write_json(study_path, raw)


def test_checked_in_coordinate_and_landscape_studies_are_distinct() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    studies = repo_root / "studies"
    assert not (studies / "surrogate_parameter_time.json").exists()
    assert not (studies / "surrogate_parameter_time_smoke.json").exists()

    coordinate = load_study_spec(
        studies / "surrogate_parameter_time_coordinate_search_smoke.json",
        repo_root=repo_root,
    )
    landscape = load_study_spec(
        studies / "surrogate_parameter_time_landscape_smoke.json",
        repo_root=repo_root,
    )
    coordinate_plan = plan_study(coordinate)
    landscape_plan = plan_study(landscape)

    assert coordinate.name == "surrogate_parameter_time_coordinate_search"
    assert {case["search_kind"] for case in coordinate_plan["cases"]} == {
        "coordinate"
    }
    assert coordinate_plan["planned_cartesian_cell_count"] == 0
    assert landscape.name == "surrogate_parameter_time_landscape"
    assert {case["search_kind"] for case in landscape_plan["cases"]} == {
        "grid"
    }
    assert landscape_plan["case_count"] == 3
    assert landscape_plan["planned_cartesian_cell_count"] == 12
    assert {variant.id for variant in landscape.variants} == {
        "heat",
        "burgers",
        "reaction_diffusion",
    }
    assert not coordinate.diagnostics
    assert not landscape.diagnostics
    assert int(landscape.base_trial.output.q) > int(
        landscape.base_trial.feature.observation.J
    )
    assert {
        readout.kind for readout in landscape.base_trial.readouts
    } == {
        "direct_fourier_decoder",
        "affine_ridge",
        "random_feature_ridge",
    }
    dataset = load_dataset_spec(landscape.dataset_spec, repo_root=repo_root)
    assert dataset.binding.kind == "validated_reference"
    assert dataset.target.system.kind == "burgers"


def test_main_landscape_is_only_planned_and_uses_validated_solver_conditions(
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = load_study_spec(
        repo_root / "studies/surrogate_parameter_time_landscape.json",
        repo_root=repo_root,
    )
    dataset = load_dataset_spec(spec.dataset_spec, repo_root=repo_root)
    validation = load_validation_spec(
        dataset.validation_spec,
        repo_root=repo_root,
    )
    plan = plan_study(
        spec,
        canonical_n_train=int(validation.samples.n_train),
    )
    assert spec.profile == "main"
    assert plan["planned_cartesian_cell_count"] == 75
    assert all(
        case["planned_cartesian_cell_count"] == 25
        for case in plan["cases"]
    )
    workload = plan["workload"]
    random = workload["random_feature"]
    assert workload["status"] == "resolved"
    assert workload["candidate_trial_upper_bound"] == 75
    assert workload["feature_state_solve_upper_bound"] == 75
    assert workload["configured_readout_count"] == 9
    assert workload["affine"] == {
        "zeta_fit_count": 450,
        "zero_zeta_svd_count": 75,
    }
    assert random["unique_random_map_count"] == 6_750
    assert random["train_validation_lift_count"] == 13_500
    assert random["ridge_fit_count"] == 40_500
    assert (
        random["selected_candidate_evaluation_member_fit_count"] == 30
    )
    assert random["eager_legacy_evaluation_member_fit_count"] == 750
    assert random["lazy_total_ridge_fit_count"] == 40_530
    assert random["eager_legacy_total_ridge_fit_count"] == 41_250
    assert random["maximum_lifted_dimension"] == 768
    assert random["maximum_target_dimension"] == 257
    assert random["maximum_training_sample_count"] == int(
        validation.samples.n_train
    )

    variants = {variant.id: variant for variant in spec.variants}
    burgers_reference = load_validation_spec(
        repo_root / "configs/validation/foundation_main.json",
        repo_root=repo_root,
    ).target_reference.reference_evolution.system
    reaction_reference = load_validation_spec(
        repo_root / "configs/validation/reaction_diffusion_main.json",
        repo_root=repo_root,
    ).target_reference.reference_evolution.system
    burgers = variants["burgers"].overrides["feature.evolution.system"]
    reaction = variants["reaction_diffusion"].overrides[
        "feature.evolution.system"
    ]
    assert burgers == burgers_reference.model_dump(mode="json")
    for field in ("solver", "dt", "nonlinear_filter"):
        assert reaction[field] == getattr(reaction_reference, field)
    assert variants["heat"].overrides["feature.evolution.system"] == {
        "kind": "heat",
        "nu": 0.01,
    }


def test_smoke_landscape_workload_matches_declared_formula() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = load_study_spec(
        repo_root
        / "studies/surrogate_parameter_time_landscape_smoke.json",
        repo_root=repo_root,
    )
    dataset = load_dataset_spec(spec.dataset_spec, repo_root=repo_root)
    validation = load_validation_spec(
        dataset.validation_spec,
        repo_root=repo_root,
    )
    plan = plan_study(
        spec,
        canonical_n_train=int(validation.samples.n_train),
    )
    workload = plan["workload"]
    random = workload["random_feature"]
    candidate_count = 3 * 2 * 2
    structure_count = 1 * 1 * 1
    selection_seed_count = 2
    zeta_count = 2
    evaluation_seed_count = 2
    assert workload["candidate_trial_upper_bound"] == candidate_count
    assert random["unique_random_map_count"] == (
        candidate_count * structure_count * selection_seed_count
    )
    assert random["ridge_fit_count"] == (
        candidate_count
        * structure_count
        * selection_seed_count
        * zeta_count
    )
    assert random["selected_candidate_evaluation_member_fit_count"] == (
        3 * evaluation_seed_count
    )
    assert random["eager_legacy_evaluation_member_fit_count"] == (
        candidate_count * evaluation_seed_count
    )

    smoke_script = (repo_root / "scripts/run_smoke.sh").read_text(
        encoding="utf-8"
    )
    assert "surrogate_parameter_time_landscape_smoke.json" in smoke_script
    assert "surrogate_parameter_time_landscape.json" not in smoke_script


def test_grid_search_evaluates_the_complete_declared_product_in_order() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = load_study_spec(
        repo_root / "studies/surrogate_parameter_time_landscape_smoke.json",
        repo_root=repo_root,
    )
    cases, skipped = build_cases(spec)
    assert not skipped
    case = next(item for item in cases if item.variant_id == "heat")
    outcome = run_search(
        _SelectionStub(),
        case.trial,
        case.search,
        metric=spec.selection.metric,
        tolerance=spec.selection.tie_tolerance,
        invalid_policy="error",
    )
    expected = [
        {
            "feature.evolution.system.nu": nu,
            "feature.evolution.time": time,
        }
        for nu, time in itertools.product((0.005, 0.01), (0.01, 0.02))
    ]
    assert outcome.search_kind == "grid"
    assert outcome.planned_cartesian_cell_count == 4
    assert [cell["axis_values"] for cell in outcome.grid_cells] == expected
    assert [cell["candidate_id"] for cell in outcome.grid_cells] == list(
        outcome.candidate_order
    )
    assert all(cell["status"] == "evaluated" for cell in outcome.grid_cells)
    rows = validation_result_rows(case=case, outcome=outcome)
    assert {row["search_kind"] for row in rows} == {"grid"}
    assert {int(row["grid_cell_index"]) for row in rows} == {0, 1, 2, 3}


def test_coordinate_search_rows_cannot_be_mistaken_for_grid_cells() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = load_study_spec(
        repo_root
        / "studies/surrogate_parameter_time_coordinate_search_smoke.json",
        repo_root=repo_root,
    )
    case = build_cases(spec)[0][0]
    outcome = run_search(
        _SelectionStub(),
        case.trial,
        case.search,
        metric=spec.selection.metric,
        tolerance=spec.selection.tie_tolerance,
        invalid_policy="error",
    )
    assert outcome.search_kind == "coordinate"
    assert outcome.planned_cartesian_cell_count is None
    assert not outcome.grid_cells
    assert all(
        stage.startswith("coordinate:")
        for stages in outcome.stages_by_candidate.values()
        for stage in stages
    )


def test_representative_condition_uses_validation_ties_and_candidate_order(
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = load_study_spec(
        repo_root / "studies/surrogate_parameter_time_landscape_smoke.json",
        repo_root=repo_root,
    )
    case = next(
        item for item in build_cases(spec)[0] if item.variant_id == "heat"
    )
    outcome = run_search(
        _SelectionStub(tied=True),
        case.trial,
        case.search,
        metric=spec.selection.metric,
        tolerance=spec.selection.tie_tolerance,
        invalid_policy="error",
    )
    evaluations = {
        (case.case_id, evaluation.candidate_id): evaluation
        for evaluation in outcome.evaluations
    }
    selection = build_selection_cases(
        spec=spec,
        cases=[case],
        outcomes={case.case_id: outcome},
        evaluations=evaluations,
    )[case.case_id]
    assert set(selection["selected_by_readout"].values()) == {
        outcome.candidate_order[0]
    }
    assert selection["representative_candidate_id"] == (
        outcome.candidate_order[0]
    )
    condition = selection["representative_feature_condition"]
    assert condition["selection_split"] == "validation"
    assert condition["candidate_id"] == outcome.candidate_order[0]
    assert not any(key.startswith("test_") for key in condition)


def test_grid_skip_evidence_preserves_invalid_cell_reason() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = load_study_spec(
        repo_root / "studies/surrogate_parameter_time_landscape_smoke.json",
        repo_root=repo_root,
    )
    case = next(
        item for item in build_cases(spec)[0] if item.variant_id == "burgers"
    )
    search = GridSearchSpec.model_validate(
        {
            "kind": "grid",
            "axes": [
                {
                    "path": "feature.evolution.time",
                    "values": [0.01, 0.015],
                }
            ],
        }
    )
    outcome = run_search(
        _SelectionStub(),
        case.trial,
        search,
        metric=spec.selection.metric,
        tolerance=spec.selection.tie_tolerance,
        invalid_policy="skip",
    )
    assert outcome.planned_cartesian_cell_count == 2
    assert len(outcome.evaluations) == 1
    assert len(outcome.skipped) == 1
    assert outcome.grid_cells[1]["status"] == "skipped"
    assert "align" in outcome.grid_cells[1]["reason"]


def test_system_override_and_metric_map_schema_are_strict(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "studies/surrogate_parameter_time_landscape_smoke.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["variants"][0]["overrides"]["feature.evolution.system"][
        "unknown_parameter"
    ] = 1
    path = write_json(tmp_path / "invalid-system.json", raw)
    spec = load_study_spec(path, repo_root=repo_root)
    with pytest.raises(ValueError, match="unknown_parameter|extra"):
        plan_study(spec)

    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["reporters"][0]["unknown_parameter"] = True
    path = write_json(tmp_path / "invalid-reporter.json", raw)
    with pytest.raises(ValueError, match="unknown_parameter"):
        load_study_spec(path, repo_root=repo_root)

    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["schema_version"] = "pol-study-v1"
    path = write_json(tmp_path / "legacy-study.json", raw)
    with pytest.raises(ValueError, match="unsupported legacy study schema"):
        load_study_spec(path, repo_root=repo_root)


def test_invalid_system_override_stops_before_dataset_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw = json.loads(
        (
            repo_root
            / "studies/surrogate_parameter_time_landscape_smoke.json"
        ).read_text(encoding="utf-8")
    )
    raw["variants"][0]["overrides"]["feature.evolution.system"][
        "unknown_parameter"
    ] = 1
    path = write_json(tmp_path / "invalid-before-data.json", raw)
    spec = load_study_spec(path, repo_root=repo_root)

    def forbidden_dataset_access(*args, **kwargs):
        raise AssertionError("dataset access must not start")

    monkeypatch.setattr(
        "pol.study.runner._prepare_dataset_and_identity",
        forbidden_dataset_access,
    )
    with pytest.raises(ValueError, match="unknown_parameter|extra"):
        run_study(spec, repo_root=repo_root)


def test_metric_map_handles_missing_selected_and_numeric_axis_order() -> None:
    spec = MetricMapReporterSpec.model_validate(
        {
            "kind": "metric_map",
            "filename": "map",
            "x": "feature_nu",
            "y": "feature_time",
            "x_values": [0.1, 0.001, 0.01],
            "y_values": [2.0, 1.0],
            "metric": "validation_error",
            "split": "validation",
            "readout_id": "affine",
            "variant_id": "heat",
            "mark_selected": True,
        }
    )
    rows = [
        {
            "variant_id": "heat",
            "readout_id": "affine",
            "search_kind": "grid",
            "feature_nu": 0.01,
            "feature_time": 2.0,
            "validation_error": 0.2,
            "selected": True,
        },
        {
            "variant_id": "heat",
            "readout_id": "affine",
            "search_kind": "grid",
            "feature_nu": 0.001,
            "feature_time": 1.0,
            "validation_error": 0.1,
            "selected": False,
        },
    ]
    data = build_metric_map_data(spec, rows)
    assert data is not None
    assert data.x_values == (0.001, 0.01, 0.1)
    assert data.y_values == (1.0, 2.0)
    assert data.selected_cells == ((1, 1),)
    assert data.matrix[0, 0] == pytest.approx(0.1)
    assert np.isnan(data.matrix[0, 1])
    assert np.isnan(data.matrix[1, 2])

    with pytest.raises(ValueError, match="duplicate cell"):
        build_metric_map_data(spec, [*rows, dict(rows[0])])
    coordinate = [dict(row, search_kind="coordinate") for row in rows]
    with pytest.raises(ValueError, match="coordinate-search"):
        build_metric_map_data(spec, coordinate)


def test_tiny_landscape_freezes_before_test_and_writes_no_noise_table(
    tmp_path: Path,
) -> None:
    study_path = _tiny_grid_study(tmp_path)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    result = run_study(spec, repo_root=tmp_path)
    verify_study_run(result.path)

    assert result.summary["planned_cartesian_cell_count"] == 4
    assert result.summary["evaluated_cartesian_cell_count"] == 4
    assert result.summary["skipped_cartesian_cell_count"] == 0
    assert result.summary["readout_stability_summary_row_count"] == 0
    assert not (result.path / "noise_robustness.csv").exists()
    assert not (
        result.path / "readout_stability_noise_summary.csv"
    ).exists()
    selection = json.loads(
        (result.path / "selection_record.json").read_text(encoding="utf-8")
    )
    case = selection["cases"]["heat"]
    assert case["candidate_order"] == [
        cell["candidate_id"] for cell in case["grid_cells"]
    ]
    assert case["representative_feature_condition"][
        "selection_split"
    ] == "validation"
    assert len(_read_csv(result.path / "validation_trials.csv")) == 12
    events = json.loads(
        (result.path / "events.json").read_text(encoding="utf-8")
    )
    names = [event["event"] for event in events]
    assert names.index("freeze_read_back") < names.index(
        "first_test_state_solve"
    )


def test_representative_condition_mismatch_stops_before_test_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_path = _tiny_grid_study(tmp_path)
    spec = load_study_spec(study_path, repo_root=tmp_path)

    import pol.study.protocol as protocol_module

    original = protocol_module.build_selection_cases

    def tampered_selection_cases(*args, **kwargs):
        cases = original(*args, **kwargs)
        condition = cases["heat"]["representative_feature_condition"]
        condition["feature"]["evolution"]["system"]["nu"] = 999.0
        return cases

    def forbidden_test(*args, **kwargs):
        raise AssertionError("test feature access must not start")

    monkeypatch.setattr(
        protocol_module,
        "build_selection_cases",
        tampered_selection_cases,
    )
    monkeypatch.setattr(
        "pol.study.trial.TrialEngine.evaluate_test",
        forbidden_test,
    )
    with pytest.raises(
        ValueError,
        match="representative feature condition differs from frozen trial",
    ):
        run_study(spec, repo_root=tmp_path)
