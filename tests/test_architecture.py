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


def _validation_import_target(
    node: ast.ImportFrom,
) -> str | None:
    if node.level == 1:
        suffix = "" if node.module is None else f".{node.module}"
        return f"pol.validation{suffix}"
    if node.level == 0 and node.module is not None:
        return node.module
    return None


def _imported_modules(path: Path) -> set[str]:
    imported: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            target = _validation_import_target(node)
            if target is not None:
                imported.add(target)
    return imported


def test_validation_check_modules_have_acyclic_dependency_direction() -> None:
    root = Path(__file__).resolve().parents[1] / "pol" / "validation"
    modules = {
        f"pol.validation.{path.stem}": path
        for path in root.glob("*.py")
        if path.name != "__init__.py"
    }
    graph: dict[str, set[str]] = {name: set() for name in modules}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                graph[name].update(
                    alias.name
                    for alias in node.names
                    if alias.name in modules
                )
            elif isinstance(node, ast.ImportFrom):
                target = _validation_import_target(node)
                if target in modules:
                    graph[name].add(target)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str, path: tuple[str, ...]) -> None:
        assert name not in visiting, " -> ".join((*path, name))
        if name in visited:
            return
        visiting.add(name)
        for dependency in graph[name]:
            visit(dependency, (*path, name))
        visiting.remove(name)
        visited.add(name)

    for module in graph:
        visit(module, ())

    target_modules = {
        "foundation_checks",
        "heat_reference",
        "reaction_diffusion_reference",
        "burgers_reference",
    }
    forbidden_dependencies = {
        "pol.validation.runner",
        "pol.validation.publication",
        "pol.artifacts",
        "pol.runtime.io",
        "pol.study",
    }
    for stem in target_modules:
        tree = ast.parse(
            (root / f"{stem}.py").read_text(encoding="utf-8")
        )
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = _validation_import_target(node)
                if target is not None:
                    imported.add(target)
        assert all(
            not any(
                dependency == forbidden
                or dependency.startswith(f"{forbidden}.")
                for forbidden in forbidden_dependencies
            )
            for dependency in imported
        ), stem


def test_validation_provenance_modules_have_single_semantic_owners() -> None:
    root = Path(__file__).resolve().parents[1] / "pol" / "validation"
    runner_imports = _imported_modules(root / "runner.py")
    assert runner_imports.isdisjoint(
        {
            "pol.validation.burgers_reference",
            "pol.validation.heat_reference",
            "pol.validation.reaction_diffusion_reference",
        }
    )

    certificate_imports = _imported_modules(root / "certificates.py")
    assert certificate_imports.isdisjoint(
        {
            "pol.validation.runner",
            "pol.validation.publication",
            "pol.validation.reference_convergence",
            "pol.systems",
        }
    )

    contract_imports = _imported_modules(root / "contracts.py")
    assert contract_imports.isdisjoint(
        {
            "pol.validation.runner",
            "pol.validation.certificates",
            "pol.validation.publication",
            "pol.artifacts",
            "pol.study",
        }
    )

    publication_imports = _imported_modules(root / "publication.py")
    assert publication_imports.isdisjoint(
        {
            "pol.validation.runner",
            "pol.validation.target_checks",
            "pol.validation.burgers_reference",
            "pol.validation.heat_reference",
            "pol.validation.reaction_diffusion_reference",
            "pol.validation.reference_convergence",
            "pol.systems",
            "torch",
        }
    )


def test_no_core_module_imports_runner_private_helpers() -> None:
    root = Path(__file__).resolve().parents[1] / "pol"
    runner = root / "validation" / "runner.py"
    for path in root.rglob("*.py"):
        if path == runner:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _validation_import_target(node)
            if target == "pol.validation.runner":
                assert all(
                    not alias.name.startswith("_")
                    for alias in node.names
                ), path


def test_study_catalog_is_question_based_and_separates_dimension_maps() -> None:
    root = Path(__file__).resolve().parents[1]
    studies = root / "studies"
    assert all("paper1" not in path.parts for path in studies.rglob("*"))

    observation = json.loads(
        (studies / "observation_output_budget_smoke.json").read_text(
            encoding="utf-8"
        )
    )
    resolution = json.loads(
        (studies / "input_simulation_resolution_smoke.json").read_text(
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
