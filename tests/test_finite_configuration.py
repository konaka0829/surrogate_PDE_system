from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

import pol
import pol.cli as cli
from pol.config.json_values import ensure_finite_json_value
from pol.config.loader import (
    load_dataset_spec,
    load_digital_baseline_spec,
    load_report_spec,
    load_study_spec,
    load_study_with_overrides,
    load_validation_spec,
)
from pol.config.models import (
    AffineRidgeReadoutSpec,
    BurgersSystemSpec,
    CoordinateAxisSpec,
    DomainSpec,
    EvolutionSpec,
    GRFSpec,
    HeatSystemSpec,
    RandomFeatureRidgeReadoutSpec,
    ReadoutStabilityNoiseDiagnosticSpec,
    SweepAxisSpec,
    VariantSpec,
)
from pol.config.report_models import PhaseDiagramReportSpec
from pol.digital_baselines.protocol import AdamSpec
from tests.helpers import write_tiny_stack


ROOT = Path(__file__).resolve().parents[1]
NONFINITE_VALUES = (float("nan"), float("inf"), float("-inf"))
NONFINITE_CONSTANTS = ("NaN", "Infinity", "-Infinity")


def _random_feature_with_scale(value: float) -> RandomFeatureRidgeReadoutSpec:
    return RandomFeatureRidgeReadoutSpec(
        id="random",
        widths=(4,),
        weight_scales=(value,),
        bias_scales=(0.1,),
        selection_seeds=(11,),
        evaluation_seeds=(21, 22),
        zetas=(1e-8,),
    )


def _phase_report_with_axis(value: float) -> PhaseDiagramReportSpec:
    return PhaseDiagramReportSpec(
        source_id="source",
        filename="phase",
        split="validation",
        metric="validation_field_relative_l2_mean",
        variant_id="heat",
        readout_id="affine",
        x="feature_nu",
        y="feature_time",
        x_values=(value,),
        y_values=(0.1,),
        x_label="x",
        y_label="y",
        metric_label="metric",
    )


PROGRAMMATIC_FACTORIES: tuple[tuple[str, Callable[[float], object]], ...] = (
    ("domain", lambda value: DomainSpec(length=value)),
    ("grf_tau", lambda value: GRFSpec(tau=value)),
    ("grf_mean", lambda value: GRFSpec(mean=value)),
    ("heat_nu", lambda value: HeatSystemSpec(nu=value)),
    (
        "burgers_dt",
        lambda value: BurgersSystemSpec(
            nu=0.1,
            dt=value,
            fine_dt=0.01,
        ),
    ),
    (
        "evolution_time",
        lambda value: EvolutionSpec(
            system=HeatSystemSpec(nu=0.1),
            time=value,
        ),
    ),
    (
        "affine_zeta",
        lambda value: AffineRidgeReadoutSpec(
            id="affine",
            zetas=(value,),
        ),
    ),
    ("random_feature_scale", _random_feature_with_scale),
    (
        "noise_level",
        lambda value: ReadoutStabilityNoiseDiagnosticSpec(
            levels=(0.0, value),
            repeats=2,
            seed=1,
        ),
    ),
    ("report_axis", _phase_report_with_axis),
    ("optimizer", lambda value: AdamSpec(learning_rate=value)),
)


@pytest.mark.parametrize("value", NONFINITE_VALUES)
@pytest.mark.parametrize(
    ("name", "factory"),
    PROGRAMMATIC_FACTORIES,
    ids=[name for name, _ in PROGRAMMATIC_FACTORIES],
)
def test_programmatic_strict_models_reject_nonfinite_floats(
    name: str,
    factory: Callable[[float], object],
    value: float,
) -> None:
    del name
    with pytest.raises(ValueError, match="finite|number"):
        factory(value)


@pytest.mark.parametrize("value", NONFINITE_VALUES)
@pytest.mark.parametrize("boundary", ("sweep", "coordinate", "variant_override"))
def test_nested_json_values_reject_nonfinite_numbers(
    boundary: str,
    value: float,
) -> None:
    nested = {"outer": [True, {"number": value}]}
    with pytest.raises(ValueError, match=r"\$.*\['outer'\]\[1\]\['number'\]"):
        if boundary == "sweep":
            SweepAxisSpec(path="feature.evolution.time", values=(nested,))
        elif boundary == "coordinate":
            CoordinateAxisSpec(
                path="feature.evolution.time",
                values=(nested,),
                anchor=nested,
            )
        else:
            VariantSpec(id="heat", overrides={"feature.evolution": nested})


def test_nested_json_finite_check_does_not_treat_bool_as_a_number() -> None:
    value = {"enabled": True, "nested": [False, 1, 0.5, None]}
    assert ensure_finite_json_value(value) is value


ConfigLoader = Callable[[Path], object]


def _validation_loader(path: Path) -> object:
    return load_validation_spec(path, repo_root=ROOT)


def _dataset_loader(path: Path) -> object:
    return load_dataset_spec(path, repo_root=ROOT)


def _study_loader(path: Path) -> object:
    return load_study_spec(path, repo_root=ROOT)


def _report_loader(path: Path) -> object:
    return load_report_spec(path, repo_root=ROOT)


def _digital_loader(path: Path) -> object:
    return load_digital_baseline_spec(path, repo_root=ROOT)


CONFIG_FAMILIES: tuple[
    tuple[str, Path, ConfigLoader, Callable[[dict[str, object], float], None]],
    ...,
] = (
    (
        "validation",
        ROOT / "configs/validation/foundation_smoke.json",
        _validation_loader,
        lambda raw, value: raw["domain"].__setitem__("length", value),  # type: ignore[union-attr]
    ),
    (
        "dataset",
        ROOT / "configs/datasets/burgers_smoke.json",
        _dataset_loader,
        lambda raw, value: raw.__setitem__("reference_nx", value),
    ),
    (
        "study",
        ROOT / "studies/surrogate_parameter_time_landscape_smoke.json",
        _study_loader,
        lambda raw, value: raw["selection"].__setitem__("tie_tolerance", value),  # type: ignore[union-attr]
    ),
    (
        "report",
        ROOT / "reports/surrogate_operator_summary_smoke.json",
        _report_loader,
        lambda raw, value: raw["reporters"][0]["x_values"].__setitem__(0, value),  # type: ignore[index,union-attr]
    ),
    (
        "digital_baseline",
        ROOT / "digital_baselines/fno1d_smoke.json",
        _digital_loader,
        lambda raw, value: raw["training"]["optimizer"].__setitem__(  # type: ignore[index,union-attr]
            "learning_rate", value
        ),
    ),
)


@pytest.mark.parametrize("constant", NONFINITE_CONSTANTS)
@pytest.mark.parametrize(
    ("family", "source", "loader", "set_nested"),
    CONFIG_FAMILIES,
    ids=[family for family, *_ in CONFIG_FAMILIES],
)
@pytest.mark.parametrize("location", ("top_level", "nested"))
def test_json_config_families_reject_nonfinite_constants(
    tmp_path: Path,
    family: str,
    source: Path,
    loader: ConfigLoader,
    set_nested: Callable[[dict[str, object], float], None],
    constant: str,
    location: str,
) -> None:
    del family
    raw = json.loads(source.read_text(encoding="utf-8"))
    value = {
        "NaN": float("nan"),
        "Infinity": float("inf"),
        "-Infinity": float("-inf"),
    }[constant]
    if location == "top_level":
        raw["nonfinite_probe"] = value
    else:
        set_nested(raw, value)
    path = tmp_path / source.name
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"invalid JSON.*non-finite JSON constant {constant!r}",
    ):
        loader(path)


@pytest.mark.parametrize("constant", NONFINITE_CONSTANTS)
def test_cli_set_rejects_nonfinite_constants_before_study_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    constant: str,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    calls = 0

    def unexpected_run(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal calls
        calls += 1
        raise AssertionError("study execution must not be reached")

    monkeypatch.setattr(cli, "run_study", unexpected_run)
    code = cli.main(
        [
            "run",
            str(study_path),
            "--set",
            f"base_trial.feature.evolution.time={constant}",
        ]
    )

    assert code == 2
    assert calls == 0
    assert "non-finite JSON constant" in capsys.readouterr().err


def test_invalid_file_fails_before_validation_or_artifact_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    raw["domain"]["length"] = float("inf")
    validation_path.write_text(json.dumps(raw), encoding="utf-8")
    calls = 0

    def unexpected_validation(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal calls
        calls += 1
        raise AssertionError("validation/PDE solving must not be reached")

    monkeypatch.setattr(cli, "ensure_validation", unexpected_validation)
    code = cli.main(["validate", str(validation_path)])

    assert code == 2
    assert calls == 0
    assert not (tmp_path / "artifacts").exists()
    assert "non-finite JSON constant" in capsys.readouterr().err


def test_nonfinite_digital_training_fails_before_training_or_source_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = ROOT / "digital_baselines/fno1d_smoke.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["output_root"] = str(tmp_path / "digital-output")
    raw["training"]["optimizer"]["learning_rate"] = float("nan")
    path = tmp_path / "digital.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    calls = 0

    def unexpected_run(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal calls
        calls += 1
        raise AssertionError("training/source access must not be reached")

    monkeypatch.setattr(cli, "run_digital_baseline", unexpected_run)
    code = cli.main(["digital-baseline", str(path)])

    assert code == 2
    assert calls == 0
    assert not (tmp_path / "digital-output").exists()
    assert "non-finite JSON constant" in capsys.readouterr().err


def test_all_checked_in_scientific_configs_strict_parse() -> None:
    catalogs = (
        ("configs/validation", load_validation_spec),
        ("configs/datasets", load_dataset_spec),
        ("studies", load_study_spec),
        ("reports", load_report_spec),
        ("digital_baselines", load_digital_baseline_spec),
    )
    parsed: list[Path] = []
    for directory, loader in catalogs:
        for path in sorted((ROOT / directory).glob("*.json")):
            loader(path, repo_root=ROOT)
            parsed.append(path)
    assert parsed


def test_valid_model_dumps_are_unchanged_and_version_is_patched() -> None:
    assert DomainSpec(length=1.0).model_dump(mode="json") == {"length": 1.0}
    assert pol.__version__ == "0.2.29"
