from __future__ import annotations

import json
from pathlib import Path

from pol.cli import main
from tests.helpers import write_tiny_stack


def test_cli_validate_data_run_and_verify(tmp_path: Path, capsys) -> None:
    validation_path, dataset_path, study_path = write_tiny_stack(tmp_path)
    assert main(["validate", str(validation_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["status"] == "pass"

    assert main(["data", "build", str(dataset_path)]) == 0
    dataset = json.loads(capsys.readouterr().out)
    assert dataset["status"] == "pass"
    assert dataset["binding_kind"] == "foundation_only"
    assert dataset["binding_status"] == "pass"
    assert dataset["target_reference_validation_status"] == "not_claimed"
    assert dataset["binding_proof_hash"]

    assert main(["run", str(study_path), "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["filesystem_mutation"] is False

    assert main(["run", str(study_path)]) == 0
    run = json.loads(capsys.readouterr().out)
    assert run["status"] == "pass"
    assert (
        run["summary"]["dataset_target_reference_validation_status"]
        == "not_claimed"
    )

    assert main(["verify", run["path"]]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["kind"] == "study_run"


def test_cli_rejects_unknown_override_path(tmp_path: Path, capsys) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    code = main(["run", str(study_path), "--set", "base_trial.nope=1"])
    captured = capsys.readouterr()
    assert code == 2
    assert "override path does not exist" in captured.err
