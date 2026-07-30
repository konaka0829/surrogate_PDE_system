from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _catalog_profiles(
    relative_directory: str,
    *,
    explicit_main_suffix: bool,
) -> tuple[set[str], dict[str, set[str]]]:
    paths = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / relative_directory).glob("*.json")
    }
    profiles: dict[str, set[str]] = {}
    for relative in paths:
        stem = Path(relative).stem
        if stem.endswith("_smoke"):
            family = stem.removesuffix("_smoke")
            profile = "smoke"
        elif explicit_main_suffix and stem.endswith("_main"):
            family = stem.removesuffix("_main")
            profile = "main"
        else:
            family = stem
            profile = "main"
        profiles.setdefault(family, set()).add(profile)
    return paths, profiles


def _plan_catalog_constant(name: str) -> set[str]:
    tree = ast.parse(
        (ROOT / "scripts" / "plan_main.py").read_text(encoding="utf-8")
    )
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError(f"missing plan catalog constant: {name}")


def test_checked_in_profile_catalog_matches_docs_and_main_audit() -> None:
    declarations = {
        "VALIDATIONS": ("configs/validation", True),
        "DATASETS": ("configs/datasets", True),
        "STUDIES": ("studies", False),
        "DIGITAL_BASELINES": ("digital_baselines", False),
        "REPORTS": ("reports", False),
    }
    inventory = (
        ROOT / "docs" / "current_implementation_inventory.md"
    ).read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    orchestrator = (ROOT / "scripts" / "run_main.sh").read_text(
        encoding="utf-8"
    )

    for constant, (directory, explicit_main_suffix) in declarations.items():
        paths, profiles = _catalog_profiles(
            directory,
            explicit_main_suffix=explicit_main_suffix,
        )
        assert profiles
        assert all(value == {"main", "smoke"} for value in profiles.values())

        main_paths = {
            relative
            for relative in paths
            if not Path(relative).stem.endswith("_smoke")
        }
        assert _plan_catalog_constant(constant) == main_paths
        assert all(relative in inventory for relative in paths)
        assert all(family in readme for family in profiles)
        assert all(relative in orchestrator for relative in main_paths)


def test_main_plan_audit_strictly_plans_every_declared_main_spec() -> None:
    watched = (
        ROOT / "configs",
        ROOT / "studies",
        ROOT / "reports",
        ROOT / "digital_baselines",
    )
    before = {
        path.relative_to(ROOT): (path.stat().st_size, path.stat().st_mtime_ns)
        for parent in watched
        for path in parent.rglob("*")
        if path.is_file()
    }
    completed = subprocess.run(
        [sys.executable, "scripts/plan_main.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    after = {
        path.relative_to(ROOT): (path.stat().st_size, path.stat().st_mtime_ns)
        for parent in watched
        for path in parent.rglob("*")
        if path.is_file()
    }
    audit = json.loads(completed.stdout)
    assert before == after
    assert audit["schema_version"] == "pol-production-plan-audit-v3"
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
        name: (item["case_count"], item["candidate_upper_bound"])
        for name, item in studies.items()
    } == {
        "heat_readout_calibration": (12, 12),
        "surrogate_parameter_time_coordinate_search": (2, 120),
        "surrogate_parameter_time_landscape": (3, 75),
        "dynamic_feature_baseline_comparison": (4, None),
        "readout_stability_noise": (1, None),
        "learning_curve": (6, None),
        "random_feature_seed_statistics": (1, None),
        "observation_output_budget": (40, None),
        "input_simulation_resolution": (32, None),
    }
    landscape = studies["surrogate_parameter_time_landscape"]["workload"]
    assert landscape["random_feature"][
        "eager_legacy_total_ridge_fit_count"
    ] == 41_250
    assert landscape["random_feature"]["ridge_fit_count"] == 40_500
    assert landscape["random_feature"][
        "selected_candidate_evaluation_member_fit_count"
    ] == 30
    unresolved = {
        name
        for name, item in studies.items()
        if item["workload"]["status"]
        == "unresolved_selection_dependency"
    }
    assert unresolved == {
        "dynamic_feature_baseline_comparison",
        "readout_stability_noise",
        "learning_curve",
        "random_feature_seed_statistics",
        "observation_output_budget",
        "input_simulation_resolution",
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
