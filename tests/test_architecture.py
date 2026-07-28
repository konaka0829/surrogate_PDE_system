from __future__ import annotations

import ast
import json
from pathlib import Path
import re


def test_core_package_has_no_publication_or_experiment_number_namespace() -> None:
    root = Path(__file__).resolve().parents[1] / "pol"
    forbidden_components = {"paper1", "workflow", "plots"}
    forbidden_tokens = re.compile(r"\b(?:E[0-9]+|e[0-9]+|paper1)\b")
    for path in root.rglob("*.py"):
        assert not (forbidden_components & set(path.parts)), path
        source = path.read_text(encoding="utf-8")
        assert forbidden_tokens.search(source) is None, path
        ast.parse(source)


def test_core_imports_do_not_depend_on_study_directory() -> None:
    root = Path(__file__).resolve().parents[1] / "pol"
    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "studies/" not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("studies") for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("studies")


def test_study_catalog_is_question_based_and_separates_dimension_maps() -> None:
    root = Path(__file__).resolve().parents[1]
    studies = root / "studies"
    assert all("paper1" not in path.parts for path in studies.rglob("*"))

    observation = json.loads(
        (studies / "observation_output_map_smoke.json").read_text(encoding="utf-8")
    )
    resolution = json.loads(
        (studies / "finite_surrogate_resolution_map_smoke.json").read_text(
            encoding="utf-8"
        )
    )
    assert [axis["path"] for axis in observation["global_axes"]] == [
        "feature.observation.J",
        "output.q",
    ]
    assert [axis["path"] for axis in resolution["global_axes"]] == [
        "input.n_tar",
        "feature.n_sur",
    ]
    assert observation["base_trial"]["input"]["n_tar"] == 32
    assert observation["base_trial"]["feature"]["n_sur"] == 32
    assert resolution["base_trial"]["feature"]["observation"]["J"] == 8
    assert resolution["base_trial"]["output"]["q"] == 9
