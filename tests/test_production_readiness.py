from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_main_plan_audit_strictly_plans_every_declared_main_spec() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/plan_main.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(completed.stdout)
    assert audit["schema_version"] == "pol-production-plan-audit-v2"
    assert audit["status"] == "pass"
    assert audit["mode"] == "read_only_plan"
    assert audit["main_execution"] is False
    assert audit["filesystem_mutation"] is False
    assert len(audit["validation_specs"]) == 3
    assert len(audit["dataset_specs"]) == 2
    assert len(audit["study_specs"]) == 9
    assert len(audit["digital_baseline_specs"]) == 1
    assert len(audit["report_specs"]) == 1
    assert all(
        item["parse_status"] == "pass" and item["main_marker"] is True
        for family in (
            "validation_specs",
            "dataset_specs",
            "study_specs",
            "digital_baseline_specs",
            "report_specs",
        )
        for item in audit[family]
    )
    studies = {item["name"]: item for item in audit["study_specs"]}
    assert {
        name: (
            item["case_count"],
            item["candidate_upper_bound"],
            item["random_feature_evaluation_seed_count_per_case"],
        )
        for name, item in studies.items()
    } == {
        "heat_readout_calibration": (12, 12, 10),
        "surrogate_parameter_time_coordinate_search": (2, 120, 10),
        "surrogate_parameter_time_landscape": (3, 75, 10),
        "dynamic_feature_baseline_comparison": (4, 4, 10),
        "readout_stability_noise": (1, 1, 10),
        "learning_curve": (6, 6, 10),
        "random_feature_seed_statistics": (1, 1, 32),
        "observation_output_budget": (40, 40, 10),
        "input_simulation_resolution": (32, 32, 10),
    }
    assert all(item["filesystem_mutation"] is False for item in studies.values())
    report = audit["report_specs"][0]
    assert report["source_count"] == 4
    assert report["reporter_count"] == 4
    digital = audit["digital_baseline_specs"][0]
    assert digital["model_kind"] == "fno1d"
    assert digital["candidate_count"] == 2
    assert digital["selection_seed_count"] == 5
    assert digital["evaluation_seed_count"] == 10
    assert digital["optimizer_step_upper_bound"] == 64000
    assert digital["filesystem_mutation"] is False


def test_main_orchestrator_requires_confirmation_and_has_no_all_stage() -> None:
    script = ROOT / "scripts" / "run_main.sh"
    assert os.access(script, os.X_OK)
    source = script.read_text(encoding="utf-8")
    assert "POL_CONFIRM_MAIN" in source
    assert "sleep " not in source
    assert "--force" not in source
    for name in (
        "heat_readout_calibration.json",
        "surrogate_parameter_time_coordinate_search.json",
        "surrogate_parameter_time_landscape.json",
        "dynamic_feature_baseline_comparison.json",
        "digital_baselines/fno1d.json",
        "readout_stability_noise.json",
        "learning_curve.json",
        "random_feature_seed_statistics.json",
        "observation_output_budget.json",
        "input_simulation_resolution.json",
        "surrogate_operator_summary.json",
    ):
        assert name in source

    environment = dict(os.environ)
    environment.pop("POL_CONFIRM_MAIN", None)
    refused = subprocess.run(
        ["bash", str(script), "--stage", "validation"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 3
    assert refused.stdout == ""
    assert "Refusing main execution" in refused.stderr

    environment["POL_CONFIRM_MAIN"] = "YES"
    no_all = subprocess.run(
        ["bash", str(script), "--stage", "all"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert no_all.returncode == 2
    assert "There is deliberately no \"all\" stage" in no_all.stdout


def test_audit_selection_and_report_sources_have_no_hidden_execution_path() -> None:
    audit_source = (ROOT / "scripts" / "plan_main.py").read_text(
        encoding="utf-8"
    )
    selection_source = (
        ROOT / "pol" / "study" / "selection_source.py"
    ).read_text(encoding="utf-8")
    reporting_source = (
        ROOT / "pol" / "reporting" / "runner.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("ensure_dataset", "ensure_validation", "run_study("):
        assert forbidden not in audit_source
        assert forbidden not in reporting_source
    assert "test_metrics.csv" not in selection_source
    assert "random_feature_ensemble_metrics.csv" not in selection_source

    core = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "pol").rglob("*.py")
    )
    assert re.search(r"\bE[0-7]\b", core) is None
    assert "paper1" not in core
    assert "Figure " not in core
    for removed in (
        "finite_surrogate_resolution_map.json",
        "observation_output_map.json",
        "surrogate_parameter_time.json",
    ):
        assert not (ROOT / "studies" / removed).exists()
