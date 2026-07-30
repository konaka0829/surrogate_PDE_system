"""Strict configuration and planning contract for digital baselines."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveFloat, PositiveInt, field_validator, model_validator

from pol.config.models import FiniteInputSpec, FourierOutputSpec, StrictModel
from pol.config.names import validate_safe_path_component


DIGITAL_BASELINE_SCHEMA_VERSION = "pol-digital-baseline-v3"


class FNO1dCandidateSpec(StrictModel):
    id: str = Field(min_length=1)
    modes: PositiveInt
    width: PositiveInt
    depth: PositiveInt

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        if not value.strip() or "/" in value or "\\" in value:
            raise ValueError("candidate id must be nonblank and path-safe")
        return value


class FNO1dModelSpec(StrictModel):
    kind: Literal["fno1d"] = "fno1d"
    activation: Literal["gelu"] = "gelu"
    coordinate_channel: Literal["none", "periodic_sin_cos"] = "none"
    candidates: tuple[FNO1dCandidateSpec, ...]

    @field_validator("coordinate_channel", mode="before")
    @classmethod
    def _reject_legacy_ramp(cls, value: object) -> object:
        if value == "unit_periodic":
            raise ValueError(
                "legacy unit_periodic ramp is not periodic; migrate explicitly "
                "to 'none' or 'periodic_sin_cos'"
            )
        return value

    @model_validator(mode="after")
    def _candidate_ids(self) -> "FNO1dModelSpec":
        if not self.candidates:
            raise ValueError("at least one FNO candidate is required")
        ids = [candidate.id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("FNO candidate ids must be unique")
        return self


class StandardScoreSpec(StrictModel):
    kind: Literal["train_standard_score"] = "train_standard_score"
    epsilon: PositiveFloat = 1e-12


class AdamSpec(StrictModel):
    kind: Literal["adam"] = "adam"
    learning_rate: PositiveFloat
    weight_decay: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _finite(self) -> "AdamSpec":
        if not math.isfinite(float(self.weight_decay)):
            raise ValueError("optimizer weight_decay must be finite")
        return self


class DigitalTrainingSpec(StrictModel):
    optimizer: AdamSpec
    epochs: PositiveInt
    batch_size: PositiveInt
    selection_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    checkpoint_metric: Literal["validation_field_relative_l2_mean"] = (
        "validation_field_relative_l2_mean"
    )
    checkpoint_tie_tolerance: float = Field(default=1e-12, ge=0.0)
    candidate_tie_tolerance: float = Field(default=1e-12, ge=0.0)
    candidate_tie_break: Literal["first_in_config_order"] = "first_in_config_order"

    @model_validator(mode="after")
    def _seeds_and_tolerances(self) -> "DigitalTrainingSpec":
        if len(self.selection_seeds) < 2:
            raise ValueError("at least two selection seeds are required")
        if len(self.evaluation_seeds) < 2:
            raise ValueError("at least two evaluation seeds are required")
        all_seeds = [*self.selection_seeds, *self.evaluation_seeds]
        if any(seed < 0 for seed in all_seeds):
            raise ValueError("training seeds must be nonnegative")
        if len(all_seeds) != len(set(all_seeds)):
            raise ValueError(
                "selection and evaluation training seeds must be unique and disjoint"
            )
        if not all(
            math.isfinite(float(value))
            for value in (
                self.checkpoint_tie_tolerance,
                self.candidate_tie_tolerance,
            )
        ):
            raise ValueError("training tie tolerances must be finite")
        return self


class PhysicalComparisonRowSpec(StrictModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    readout_id: str = Field(min_length=1)

    @field_validator("id", "label", "variant_id", "readout_id")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("physical comparison coordinates must not be blank")
        return value


class PhysicalComparisonSpec(StrictModel):
    source_study_spec: Path
    rows: tuple[PhysicalComparisonRowSpec, ...]

    @model_validator(mode="after")
    def _rows(self) -> "PhysicalComparisonSpec":
        if not self.rows:
            raise ValueError("at least one physical comparison row is required")
        ids = [row.id for row in self.rows]
        coordinates = [(row.variant_id, row.readout_id) for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("physical comparison row ids must be unique")
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("physical comparison coordinates must be unique")
        return self


class DigitalReportingSpec(StrictModel):
    primary_result: Literal["independent_training_seed_metric_summary"] = (
        "independent_training_seed_metric_summary"
    )
    confidence_level: Literal[0.95] = 0.95
    confidence_interval_method: Literal["student_t"] = "student_t"
    prediction_ensemble: Literal["separate_table"] = "separate_table"
    wall_clock_energy_comparison: Literal[
        "only_same_measurement_protocol"
    ] = "only_same_measurement_protocol"


class DigitalExecutionSpec(StrictModel):
    device: Literal["cpu"] = "cpu"
    torch_threads: PositiveInt = 1

    @field_validator("device", mode="before")
    @classmethod
    def _cpu_only(cls, value: object) -> object:
        if value != "cpu":
            raise ValueError("official digital baseline execution is CPU-only")
        return value


class DigitalBaselineSpec(StrictModel):
    schema_version: Literal["pol-digital-baseline-v3"] = (
        DIGITAL_BASELINE_SCHEMA_VERSION
    )
    name: str = Field(min_length=1)
    profile: Literal["test", "smoke", "main"]
    output_root: Path = Path("outputs/digital_baselines")
    dataset_spec: Path
    input: FiniteInputSpec
    output: FourierOutputSpec
    model: FNO1dModelSpec
    normalization: StandardScoreSpec
    training: DigitalTrainingSpec
    physical_comparison: PhysicalComparisonSpec
    reporting: DigitalReportingSpec = Field(default_factory=DigitalReportingSpec)
    execution: DigitalExecutionSpec = Field(default_factory=DigitalExecutionSpec)

    @field_validator("name", "profile")
    @classmethod
    def _safe_output_components(cls, value: str, info: object) -> str:
        field_name = str(getattr(info, "field_name", "path component"))
        return validate_safe_path_component(
            value,
            field=f"digital baseline {field_name}",
        )

    @model_validator(mode="after")
    def _interfaces(self) -> "DigitalBaselineSpec":
        n_tar = int(self.input.n_tar)
        q = int(self.output.q)
        if q > n_tar:
            raise ValueError("digital baseline output.q must be <= input.n_tar")
        available_modes = n_tar // 2 + 1
        if any(
            int(candidate.modes) > available_modes
            for candidate in self.model.candidates
        ):
            raise ValueError(
                "FNO modes must not exceed the finite n_tar RFFT mode count"
            )
        return self


def semantic_digital_baseline_spec(spec: DigitalBaselineSpec) -> dict[str, object]:
    """Return storage-location-independent configured scientific semantics."""
    payload = spec.model_dump(mode="json")
    payload.pop("output_root", None)
    payload.pop("dataset_spec", None)
    comparison = dict(payload["physical_comparison"])
    comparison.pop("source_study_spec", None)
    payload["physical_comparison"] = comparison
    return payload


def plan_digital_baseline(
    spec: DigitalBaselineSpec,
    *,
    n_train: int,
) -> dict[str, object]:
    """Pure upper-bound plan; it neither resolves nor executes source runs."""
    batches_per_epoch = math.ceil(int(n_train) / int(spec.training.batch_size))
    selection_models = (
        len(spec.model.candidates) * len(spec.training.selection_seeds)
    )
    evaluation_models = len(spec.training.evaluation_seeds)
    return {
        "schema_version": "pol-digital-baseline-plan-v2",
        "name": spec.name,
        "profile": spec.profile,
        "model_kind": spec.model.kind,
        "coordinate_channel": spec.model.coordinate_channel,
        "lifting_input_channels": (
            1 if spec.model.coordinate_channel == "none" else 3
        ),
        "candidate_count": len(spec.model.candidates),
        "selection_seed_count": len(spec.training.selection_seeds),
        "evaluation_seed_count": len(spec.training.evaluation_seeds),
        "selection_training_model_count": selection_models,
        "evaluation_training_model_count": evaluation_models,
        "total_training_model_count": selection_models + evaluation_models,
        "epochs_per_model": int(spec.training.epochs),
        "batches_per_epoch": batches_per_epoch,
        "optimizer_step_upper_bound": (
            (selection_models + evaluation_models)
            * int(spec.training.epochs)
            * batches_per_epoch
        ),
        "n_tar": int(spec.input.n_tar),
        "q": int(spec.output.q),
        "physical_comparison_row_count": len(spec.physical_comparison.rows),
        "filesystem_mutation": False,
        "main_execution": False,
    }


__all__ = [
    "DIGITAL_BASELINE_SCHEMA_VERSION",
    "DigitalBaselineSpec",
    "FNO1dCandidateSpec",
    "plan_digital_baseline",
    "semantic_digital_baseline_spec",
]
