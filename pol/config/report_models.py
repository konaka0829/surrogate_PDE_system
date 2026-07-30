from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import Field, PositiveInt, field_validator, model_validator

from .models import StrictModel


class ReportSourceSpec(StrictModel):
    id: str
    study_spec: Path

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        if (
            not value
            or Path(value).name != value
            or value.startswith(".")
            or any(character.isspace() for character in value)
        ):
            raise ValueError("report source id must be a safe nonempty name")
        return value


class PhaseDiagramReportSpec(StrictModel):
    kind: Literal["phase_diagram_report"] = "phase_diagram_report"
    source_id: str
    filename: str
    split: Literal["validation", "test"]
    metric: str
    variant_id: str
    readout_id: str
    x: Literal["feature_nu", "feature_time", "J", "q", "n_tar", "n_sur"]
    y: Literal["feature_nu", "feature_time", "J", "q", "n_tar", "n_sur"]
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    x_label: str
    y_label: str
    metric_label: str
    xscale: Literal["linear", "log"] = "linear"
    yscale: Literal["linear", "log"] = "linear"
    mark_selected: bool = False
    formats: tuple[Literal["png", "pdf"], ...] = ("png",)
    dpi: PositiveInt = 150

    @model_validator(mode="after")
    def _phase_diagram(self) -> "PhaseDiagramReportSpec":
        if self.x == self.y:
            raise ValueError("phase diagram axes must be distinct")
        if (
            not self.x_values
            or not self.y_values
            or len(set(self.x_values)) != len(self.x_values)
            or len(set(self.y_values)) != len(self.y_values)
        ):
            raise ValueError(
                "phase diagram axes must be nonempty and contain unique values"
            )
        if any(
            not value > 0.0
            for scale, values in (
                (self.xscale, self.x_values),
                (self.yscale, self.y_values),
            )
            if scale == "log"
            for value in values
        ):
            raise ValueError("log phase diagram axes require positive values")
        if self.mark_selected and self.split != "validation":
            raise ValueError(
                "selected-cell marking belongs to a validation phase diagram"
            )
        _safe_filename(self.filename)
        if not self.formats or len(set(self.formats)) != len(self.formats):
            raise ValueError("phase diagram formats must be nonempty and unique")
        return self


class BaselineTableRowSpec(StrictModel):
    id: str
    label: str
    variant_id: str
    readout_id: str

    @field_validator("id")
    @classmethod
    def _row_id(cls, value: str) -> str:
        if not value or Path(value).name != value or value.startswith("."):
            raise ValueError("baseline row id must be a safe nonempty name")
        return value


class BaselineSummaryTableSpec(StrictModel):
    kind: Literal["baseline_summary_table"] = "baseline_summary_table"
    source_id: str
    filename: str
    rows: tuple[BaselineTableRowSpec, ...]
    field_metric: Literal[
        "test_field_relative_l2_mean",
        "test_field_absolute_l2_mean",
    ] = "test_field_relative_l2_mean"
    data_metric: Literal[
        "test_data_field_relative_l2_mean",
        "test_data_field_absolute_l2_mean",
    ] = "test_data_field_relative_l2_mean"
    field_representation_floor_metric: Literal[
        "test_representation_floor_relative_l2_mean"
    ] = "test_representation_floor_relative_l2_mean"
    data_representation_floor_metric: Literal[
        "test_data_representation_floor_relative_l2_mean"
    ] = "test_data_representation_floor_relative_l2_mean"
    formatted_outputs: tuple[Literal["markdown", "latex"], ...] = (
        "markdown",
        "latex",
    )
    significant_digits: int = Field(default=4, ge=2, le=12)

    @model_validator(mode="after")
    def _baseline_table(self) -> "BaselineSummaryTableSpec":
        _safe_filename(self.filename)
        if not self.rows:
            raise ValueError("baseline table rows must not be empty")
        row_ids = [row.id for row in self.rows]
        coordinates = [
            (row.variant_id, row.readout_id) for row in self.rows
        ]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("baseline table row ids must be unique")
        if len(set(coordinates)) != len(coordinates):
            raise ValueError(
                "baseline table variant/readout coordinates must be unique"
            )
        if len(set(self.formatted_outputs)) != len(self.formatted_outputs):
            raise ValueError("formatted baseline outputs must be unique")
        return self


ReportItemSpec = Annotated[
    Union[PhaseDiagramReportSpec, BaselineSummaryTableSpec],
    Field(discriminator="kind"),
]


class ReportSpec(StrictModel):
    schema_version: Literal["pol-report-v1"] = "pol-report-v1"
    name: str
    profile: str
    output_root: Path = Path("outputs/reports")
    sources: tuple[ReportSourceSpec, ...]
    reporters: tuple[ReportItemSpec, ...]

    @model_validator(mode="after")
    def _report(self) -> "ReportSpec":
        _safe_filename(self.name)
        if len(self.sources) < 2:
            raise ValueError(
                "a cross-run report requires at least two source runs"
            )
        source_ids = [source.id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("report source ids must be unique")
        if not self.reporters:
            raise ValueError("reporters must not be empty")
        unknown_sources = {
            reporter.source_id for reporter in self.reporters
        } - set(source_ids)
        if unknown_sources:
            raise ValueError("reporter references an unknown source id")
        filenames = [reporter.filename for reporter in self.reporters]
        if len(set(filenames)) != len(filenames):
            raise ValueError("reporter filenames must be unique")
        return self


def _safe_filename(value: str) -> None:
    if (
        not value
        or Path(value).name != value
        or value.startswith(".")
        or Path(value).suffix
    ):
        raise ValueError(
            "report filename must be a safe extension-free basename"
        )


__all__ = [
    "BaselineSummaryTableSpec",
    "BaselineTableRowSpec",
    "PhaseDiagramReportSpec",
    "ReportItemSpec",
    "ReportSourceSpec",
    "ReportSpec",
]
