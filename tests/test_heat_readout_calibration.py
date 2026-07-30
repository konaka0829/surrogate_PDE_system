from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pol.config.loader import (
    load_dataset_spec,
    load_study_spec,
)
from pol.config.models import (
    HeatMultiplierComparisonReporterSpec,
    HeatMultiplierDiagnosticSpec,
    TrialSpec,
)
from pol.learning.ridge import l2_synthesis_matrix
from pol.math.fourier import real_fourier_synthesis
from pol.plotting.reporters import generate_reporters
from pol.study.diagnostics import heat_multiplier_diagnostic
from pol.study.runner import plan_study, run_study
from tests.helpers import write_json, write_tiny_stack


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _trial(
    *,
    surrogate_nu: float,
    surrogate_time: float,
    q: int = 5,
    J: int = 8,
) -> TrialSpec:
    return TrialSpec.model_validate(
        {
            "input": {"n_tar": max(16, q), "resampling": "spectral"},
            "feature": {
                "kind": "pde_dynamics",
                "evolution": {
                    "system": {"kind": "heat", "nu": surrogate_nu},
                    "time": surrogate_time,
                },
                "n_sur": max(16, J),
                "observation": {
                    "kind": "equispaced_points",
                    "J": J,
                    "l2_scale": True,
                },
            },
            "output": {"kind": "real_fourier", "q": q},
            "readouts": [
                {"id": "direct", "kind": "direct_fourier_decoder"}
            ],
        }
    )


def _dataset(
    *,
    target_nu: float,
    target_time: float,
    domain_length: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        target_metadata={
            "kind": "heat",
            "nu": target_nu,
            "time": target_time,
        },
        domain_length=domain_length,
    )


def _ideal_vector(
    q: int,
    *,
    target_nu: float,
    target_time: float,
    surrogate_nu: float,
    surrogate_time: float,
    domain_length: float,
) -> torch.Tensor:
    values = [1.0]
    for mode in range(1, (q - 1) // 2 + 1):
        wavenumber = 2.0 * math.pi * mode / domain_length
        value = math.exp(
            -(
                target_nu * target_time
                - surrogate_nu * surrogate_time
            )
            * wavenumber ** 2
        )
        values.extend((value, value))
    return torch.tensor(values, dtype=torch.float64)


def _affine_model_for_effective(
    trial: TrialSpec,
    *,
    domain_length: float,
    effective: torch.Tensor,
    zeta: float = 1e-8,
) -> dict[str, object]:
    synthesis = l2_synthesis_matrix(
        int(trial.output.q),
        int(trial.feature.observation.J),
        domain_length=domain_length,
        dtype=torch.float64,
    )
    return {
        "kind": "affine_ridge",
        "zeta": zeta,
        "W": effective @ torch.linalg.pinv(synthesis),
        "b": torch.zeros(int(trial.output.q), dtype=torch.float64),
    }


def _diagnose_affine(
    *,
    target_nu: float,
    target_time: float,
    surrogate_nu: float,
    surrogate_time: float,
    domain_length: float,
    effective: torch.Tensor | None = None,
    q: int = 5,
    J: int = 8,
    floor: float = 1e-14,
):
    trial = _trial(
        surrogate_nu=surrogate_nu,
        surrogate_time=surrogate_time,
        q=q,
        J=J,
    )
    if effective is None:
        ideal = _ideal_vector(
            q,
            target_nu=target_nu,
            target_time=target_time,
            surrogate_nu=surrogate_nu,
            surrogate_time=surrogate_time,
            domain_length=domain_length,
        )
        effective = torch.diag(ideal)
    model = _affine_model_for_effective(
        trial,
        domain_length=domain_length,
        effective=effective,
    )
    return heat_multiplier_diagnostic(
        HeatMultiplierDiagnosticSpec(
            identifiable_multiplier_floor=floor
        ),
        dataset=_dataset(
            target_nu=target_nu,
            target_time=target_time,
            domain_length=domain_length,
        ),
        trial=trial,
        model=model,
        case_id="case",
        candidate_id="candidate",
        variant_id="variant",
        readout_id="affine",
    )


def test_matched_heat_has_unit_ideal_multiplier_on_resolved_modes() -> None:
    result = _diagnose_affine(
        target_nu=0.1,
        target_time=0.2,
        surrogate_nu=0.1,
        surrogate_time=0.2,
        domain_length=1.0,
    )
    assert result.summary_row["diffusion_condition"] == "matched"
    assert result.summary_row["identifiable_mode_count"] == 3
    assert result.summary_row["diagonal_rmse"] == pytest.approx(0.0, abs=1e-14)
    for row in result.coefficient_rows:
        assert row["identifiable"] is True
        assert row["ideal_readout_multiplier"] == pytest.approx(1.0)


def test_l2_synthesis_matrix_preserves_resolved_non_unit_domain_values() -> None:
    q, J, domain_length = 5, 8, 2.3
    actual = l2_synthesis_matrix(
        q,
        J,
        domain_length=domain_length,
        dtype=torch.float64,
    )
    expected = real_fourier_synthesis(
        torch.eye(q, dtype=torch.float64),
        J,
        domain_length=domain_length,
    ).T * math.sqrt(domain_length / J)
    assert torch.allclose(actual, expected, atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize(
    ("surrogate_nu", "condition", "amplifies"),
    [
        (0.05, "under_diffusive", False),
        (0.15, "more_diffusive", True),
    ],
)
def test_heat_multiplier_matches_analytic_mismatch_on_non_unit_domain(
    surrogate_nu: float,
    condition: str,
    amplifies: bool,
) -> None:
    domain_length = 2.5
    target_nu = 0.1
    time = 0.2
    result = _diagnose_affine(
        target_nu=target_nu,
        target_time=time,
        surrogate_nu=surrogate_nu,
        surrogate_time=time,
        domain_length=domain_length,
    )
    cosine_mode_one = result.coefficient_rows[1]
    wavenumber = 2.0 * math.pi / domain_length
    target = math.exp(-target_nu * time * wavenumber ** 2)
    surrogate = math.exp(-surrogate_nu * time * wavenumber ** 2)
    assert cosine_mode_one["target_heat_multiplier"] == pytest.approx(target)
    assert cosine_mode_one["surrogate_heat_multiplier"] == pytest.approx(
        surrogate
    )
    assert cosine_mode_one["ideal_readout_multiplier"] == pytest.approx(
        target / surrogate
    )
    assert cosine_mode_one["amplification"] is amplifies
    assert result.summary_row["diffusion_condition"] == condition
    assert (
        result.summary_row["inverse_amplification_required"] is amplifies
    )


def test_heat_multiplier_threshold_is_underflow_safe() -> None:
    result = _diagnose_affine(
        target_nu=1.0,
        target_time=1.0,
        surrogate_nu=10.0,
        surrogate_time=10.0,
        domain_length=1.0,
        q=5,
        J=8,
        floor=1e-12,
        effective=torch.eye(5, dtype=torch.float64),
    )
    high_mode = result.coefficient_rows[-1]
    assert high_mode["surrogate_heat_multiplier"] == 0.0
    assert high_mode["multiplier_identifiable"] is False
    assert high_mode["identifiable"] is False
    assert (
        high_mode["diagnostic_status"]
        == "surrogate_multiplier_below_identifiable_floor"
    )
    assert high_mode["ideal_readout_multiplier"] is None
    assert high_mode["relative_diagonal_error"] is None


def test_even_observation_nyquist_and_q_greater_than_J_are_explicit() -> None:
    trial = _trial(
        surrogate_nu=0.1,
        surrogate_time=0.2,
        q=9,
        J=4,
    )
    result = heat_multiplier_diagnostic(
        HeatMultiplierDiagnosticSpec(
            identifiable_multiplier_floor=1e-14
        ),
        dataset=_dataset(
            target_nu=0.1,
            target_time=0.2,
            domain_length=1.0,
        ),
        trial=trial,
        model={
            "kind": "direct_fourier_decoder",
            "q": 9,
            "domain_length": 1.0,
        },
        case_id="case",
        candidate_id="candidate",
        variant_id="matched",
        readout_id="direct",
    )
    nyquist_cosine = result.coefficient_rows[3]
    nyquist_sine = result.coefficient_rows[4]
    assert nyquist_cosine["mode_index"] == 2
    assert (
        nyquist_cosine["observation_status"]
        == "even_grid_nyquist_cosine_not_pair_identifiable"
    )
    assert (
        nyquist_sine["observation_status"]
        == "even_grid_nyquist_sine_zero_on_observation_grid"
    )
    assert nyquist_cosine["identifiable"] is False
    assert nyquist_sine["identifiable"] is False


def test_heat_multiplier_summary_aggregates_errors_and_marks_model3_na() -> None:
    ideal = _ideal_vector(
        5,
        target_nu=0.1,
        target_time=0.2,
        surrogate_nu=0.15,
        surrogate_time=0.2,
        domain_length=1.0,
    )
    effective = torch.diag(ideal + 0.1)
    effective[0, 1] = 0.3
    result = _diagnose_affine(
        target_nu=0.1,
        target_time=0.2,
        surrogate_nu=0.15,
        surrogate_time=0.2,
        domain_length=1.0,
        effective=effective,
    )
    summary = result.summary_row
    assert summary["identifiable_mode_count"] == 3
    assert summary["identifiable_coefficient_count"] == 5
    assert summary["diagonal_rmse"] == pytest.approx(0.1)
    assert summary["diagonal_max_error"] == pytest.approx(0.1)
    assert summary["off_diagonal_frobenius_norm"] == pytest.approx(0.3)
    assert summary["max_ideal_amplification"] == pytest.approx(
        float(ideal.max())
    )
    assert summary["selected_zeta"] == pytest.approx(1e-8)

    trial = _trial(surrogate_nu=0.15, surrogate_time=0.2)
    random_result = heat_multiplier_diagnostic(
        HeatMultiplierDiagnosticSpec(),
        dataset=_dataset(
            target_nu=0.1,
            target_time=0.2,
            domain_length=1.0,
        ),
        trial=trial,
        model={"kind": "random_feature_ridge", "zeta": 1e-6},
        case_id="case",
        candidate_id="candidate",
        variant_id="more_diffusive",
        readout_id="random",
    )
    assert random_result.coefficient_rows == ()
    assert random_result.summary_row["applicable"] is False
    assert (
        random_result.summary_row["diagnostic_status"]
        == "not_applicable_nonlinear_readout"
    )
    assert random_result.summary_row["selected_zeta"] == pytest.approx(1e-6)


def test_checked_heat_calibration_design_has_three_physical_variants() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for filename, target_nu in (
        ("heat_readout_calibration.json", 0.01),
        ("heat_readout_calibration_smoke.json", 0.1),
    ):
        spec = load_study_spec(
            repo_root / "studies" / filename,
            repo_root=repo_root,
        )
        dataset = load_dataset_spec(spec.dataset_spec, repo_root=repo_root)
        assert dataset.binding.kind == "validated_reference"
        assert float(dataset.target.system.nu) == pytest.approx(target_nu)
        assert {variant.id for variant in spec.variants} == {
            "under_diffusive",
            "matched",
            "more_diffusive",
        }
        assert {item.kind for item in spec.diagnostics} == {
            "heat_multiplier"
        }
        assert "noise_curve" not in {
            reporter.kind for reporter in spec.reporters
        }
        feature_nu = {
            variant.id: float(
                variant.overrides["feature.evolution.system.nu"]
            )
            for variant in spec.variants
        }
        assert feature_nu["under_diffusive"] < target_nu
        assert feature_nu["matched"] == pytest.approx(target_nu)
        assert feature_nu["more_diffusive"] > target_nu

    smoke = load_study_spec(
        repo_root / "studies" / "heat_readout_calibration_smoke.json",
        repo_root=repo_root,
    )
    plan = plan_study(smoke)
    assert plan["case_count"] == 9
    assert all(
        set(case["readout_ids"]) == {"direct", "affine", "random"}
        for case in plan["cases"]
    )


def test_heat_study_publishes_no_noise_table_and_freezes_before_test(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    result = run_study(
        load_study_spec(study_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    assert not (result.path / "noise_robustness.csv").exists()
    assert result.summary["readout_stability_summary_row_count"] == 0
    assert not (
        result.path / "readout_stability_noise_summary.csv"
    ).exists()
    summaries = _read_csv(result.path / "heat_multiplier_summary.csv")
    assert {row["readout_id"] for row in summaries} == {
        "direct",
        "affine",
        "random",
    }
    random_summary = next(
        row for row in summaries if row["readout_id"] == "random"
    )
    assert (
        random_summary["diagnostic_status"]
        == "not_applicable_nonlinear_readout"
    )
    events = json.loads(
        (result.path / "events.json").read_text(encoding="utf-8")
    )
    names = [event["event"] for event in events]
    assert names.index("freeze_read_back") < names.index(
        "first_test_state_solve"
    )
    assert names.index("first_test_state_solve") <= names.index(
        "first_test_metric"
    )


def test_heat_diagnostic_and_reporter_reject_unknown_keys(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["diagnostics"][0]["unknown_scientific_key"] = True
    write_json(study_path, raw)
    with pytest.raises(
        ValueError,
        match=r"\$\.diagnostics\[0\]\.heat_multiplier",
    ):
        load_study_spec(study_path, repo_root=tmp_path)

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        HeatMultiplierComparisonReporterSpec.model_validate(
            {
                "kind": "heat_multiplier_comparison",
                "filename": "comparison",
                "unknown_scientific_key": True,
            }
        )


def test_heat_multiplier_reporter_reads_rows_only(tmp_path: Path) -> None:
    rows = [
        {
            "variant_id": "matched",
            "readout_id": "affine",
            "q": 3,
            "coefficient_index": index,
            "ideal_readout_multiplier": 1.0,
            "effective_learned_diagonal": 1.0 + 0.01 * index,
        }
        for index in range(3)
    ]
    reporter = HeatMultiplierComparisonReporterSpec(
        filename="ideal_vs_effective",
        readout_id="affine",
        q=3,
        formats=("png",),
        dpi=60,
    )
    created = generate_reporters(
        [reporter],
        validation_rows=[],
        test_rows=[],
        multiplier_rows=rows,
        noise_rows=[],
        output_dir=tmp_path,
    )
    assert created == ["ideal_vs_effective.png"]
    assert (tmp_path / "ideal_vs_effective.png").is_file()
