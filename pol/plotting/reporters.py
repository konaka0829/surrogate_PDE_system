from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pol.config.models import (
    HeatMultiplierComparisonReporterSpec,
    FourierErrorSpectraReporterSpec,
    LearningCurveReporterSpec,
    MetricMapReporterSpec,
    MetricCurveReporterSpec,
    RandomFeatureSeedDistributionReporterSpec,
    ReadoutStabilityReporterSpec,
    RepresentativePredictionFieldsReporterSpec,
    ReporterSpec,
)


def _save(fig, root: Path, filename: str, formats: Iterable[str], dpi: int) -> list[str]:
    names: list[str] = []
    for extension in formats:
        name = f"{filename}.{extension}"
        fig.savefig(root / name, dpi=dpi, bbox_inches="tight")
        names.append(name)
    plt.close(fig)
    return names


def expected_reporter_outputs(
    reporters: Iterable[ReporterSpec],
) -> list[str]:
    names = [
        f"{reporter.filename}.{extension}"
        for reporter in reporters
        for extension in reporter.formats
    ]
    if len(names) != len(set(names)):
        raise ValueError("configured reporters have duplicate output filenames")
    return names


def _table_for_split(
    split: str,
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return validation_rows if split == "validation" else test_rows


def _metric_curve(
    spec: MetricCurveReporterSpec,
    rows: list[dict[str, Any]],
    root: Path,
) -> list[str]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if spec.x in row and spec.metric in row and row[spec.x] is not None:
            groups[tuple(row.get(key) for key in spec.group_by)].append(row)
    if not groups:
        return []
    fig, ax = plt.subplots()
    for key, values in groups.items():
        ordered = sorted(values, key=lambda item: float(item[spec.x]))
        x = [float(item[spec.x]) for item in ordered]
        y = [float(item[spec.metric]) for item in ordered]
        label = ", ".join(
            f"{name}={value}" for name, value in zip(spec.group_by, key, strict=True)
        )
        ax.plot(x, y, marker="o", label=label)
    ax.set_xlabel(spec.x)
    ax.set_ylabel(spec.metric)
    ax.set_xscale(spec.xscale)
    ax.set_yscale(spec.yscale)
    if len(groups) > 1:
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, root, spec.filename, spec.formats, spec.dpi)


@dataclass(frozen=True)
class MetricMapData:
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    matrix: np.ndarray
    selected_cells: tuple[tuple[int, int], ...]
    missing_cells: tuple[tuple[int, int], ...]
    invalid_cells: tuple[tuple[int, int, str], ...]


def _skipped_axis_value(
    values: dict[str, Any],
    name: str,
) -> float | None:
    aliases = {
        "feature_nu": "nu",
        "feature_time": "time",
    }
    candidates = [
        value
        for path, value in values.items()
        if path == name
        or path.rsplit(".", maxsplit=1)[-1] == name
        or path.rsplit(".", maxsplit=1)[-1] == aliases.get(name)
    ]
    if len(candidates) != 1:
        return None
    try:
        result = float(candidates[0])
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _invalid_metric_map_cells(
    spec: MetricMapReporterSpec,
    skipped_rows: Iterable[dict[str, Any]],
    *,
    x_positions: dict[float, int],
    y_positions: dict[float, int],
) -> tuple[tuple[int, int, str], ...]:
    invalid: dict[tuple[int, int], str] = {}
    for item in skipped_rows:
        variant_id = item.get("variant_id")
        case_id = item.get("case_id")
        if (
            spec.variant_id is not None
            and variant_id not in (None, spec.variant_id)
        ):
            continue
        if (
            spec.variant_id is not None
            and isinstance(case_id, str)
            and not (
                case_id == spec.variant_id
                or case_id.startswith(f"{spec.variant_id}-")
            )
        ):
            continue
        values = item.get("global_values", item.get("overrides"))
        if not isinstance(values, dict):
            continue
        x = _skipped_axis_value(values, spec.x)
        y = _skipped_axis_value(values, spec.y)
        if x not in x_positions or y not in y_positions:
            continue
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason:
            reason = "invalid cell (no reason recorded)"
        invalid[(x_positions[x], y_positions[y])] = reason
    return tuple(
        (xi, yi, reason)
        for (xi, yi), reason in sorted(invalid.items())
    )


def build_metric_map_data(
    spec: MetricMapReporterSpec,
    rows: list[dict[str, Any]],
    *,
    skipped_rows: Iterable[dict[str, Any]] = (),
) -> MetricMapData | None:
    skipped_rows = tuple(skipped_rows)
    filtered = [
        row
        for row in rows
        if row.get("readout_id") == spec.readout_id
        and (
            spec.variant_id is None
            or row.get("variant_id") == spec.variant_id
        )
        and spec.x in row
        and spec.y in row
        and spec.metric in row
        and row[spec.x] not in ("", None)
        and row[spec.y] not in ("", None)
        and row[spec.metric] not in ("", None)
    ]
    if not filtered and not skipped_rows:
        return None
    if any(row.get("search_kind") == "coordinate" for row in filtered):
        raise ValueError("metric_map cannot consume coordinate-search rows")
    x_values = tuple(sorted(float(value) for value in spec.x_values))
    y_values = tuple(sorted(float(value) for value in spec.y_values))
    x_positions = {value: index for index, value in enumerate(x_values)}
    y_positions = {value: index for index, value in enumerate(y_values)}
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    selected_cells: list[tuple[int, int]] = []
    for row in filtered:
        x = float(row[spec.x])
        y = float(row[spec.y])
        if x not in x_positions or y not in y_positions:
            raise ValueError(
                "metric_map row lies outside the declared Cartesian axes"
            )
        xi = x_positions[x]
        yi = y_positions[y]
        if not np.isnan(matrix[yi, xi]):
            raise ValueError(
                f"metric_map has duplicate cell x={x:g}, y={y:g}"
            )
        metric = float(row[spec.metric])
        if not np.isfinite(metric):
            raise ValueError(
                f"metric_map has non-finite metric at x={x:g}, y={y:g}"
            )
        matrix[yi, xi] = metric
        selected_value = row.get("selected", False)
        is_selected = (
            selected_value
            if isinstance(selected_value, bool)
            else str(selected_value).lower() == "true"
        )
        if is_selected:
            selected_cells.append((xi, yi))
    if spec.mark_selected and len(selected_cells) != 1:
        raise ValueError(
            "metric_map with mark_selected=true requires exactly one "
            "validation-selected cell"
        )
    invalid_cells = _invalid_metric_map_cells(
        spec,
        skipped_rows,
        x_positions=x_positions,
        y_positions=y_positions,
    )
    invalid_positions = {(xi, yi) for xi, yi, _ in invalid_cells}
    missing_cells = tuple(
        (xi, yi)
        for yi in range(len(y_values))
        for xi in range(len(x_values))
        if np.isnan(matrix[yi, xi]) and (xi, yi) not in invalid_positions
    )
    return MetricMapData(
        x_values=x_values,
        y_values=y_values,
        matrix=matrix,
        selected_cells=tuple(selected_cells),
        missing_cells=missing_cells,
        invalid_cells=invalid_cells,
    )


def _metric_map(
    spec: MetricMapReporterSpec,
    rows: list[dict[str, Any]],
    root: Path,
    *,
    skipped_rows: Iterable[dict[str, Any]] = (),
) -> list[str]:
    data = build_metric_map_data(spec, rows, skipped_rows=skipped_rows)
    if data is None:
        return []
    fig, ax = plt.subplots()
    image = ax.imshow(data.matrix, origin="lower", aspect="auto")
    ax.set_xticks(
        range(len(data.x_values)),
        [f"{value:g}" for value in data.x_values],
    )
    ax.set_yticks(
        range(len(data.y_values)),
        [f"{value:g}" for value in data.y_values],
    )
    ax.set_xlabel(spec.x)
    ax.set_ylabel(spec.y)
    if spec.mark_selected:
        for xi, yi in data.selected_cells:
            ax.scatter(
                [xi],
                [yi],
                marker="*",
                s=140,
                facecolors="none",
                edgecolors="white",
                linewidths=1.5,
                label="validation selected",
            )
        ax.legend()
    if data.missing_cells:
        ax.scatter(
            [xi for xi, _ in data.missing_cells],
            [yi for _, yi in data.missing_cells],
            marker="s",
            s=70,
            facecolors="none",
            edgecolors="gray",
            linewidths=1.0,
            label="missing: no verified validation row",
        )
    if data.invalid_cells:
        ax.scatter(
            [xi for xi, _, _ in data.invalid_cells],
            [yi for _, yi, _ in data.invalid_cells],
            marker="x",
            s=70,
            color="red",
            linewidths=1.5,
            label="invalid: see cell reason",
        )
        for xi, yi, reason in data.invalid_cells:
            ax.annotate(
                reason,
                (xi, yi),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
                color="red",
            )
    if data.missing_cells or data.invalid_cells:
        ax.legend()
    fig.colorbar(image, ax=ax, label=spec.metric)
    return _save(fig, root, spec.filename, spec.formats, spec.dpi)


def _noise_curve(
    spec: ReadoutStabilityReporterSpec,
    rows: list[dict[str, Any]],
    root: Path,
) -> list[str]:
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            "noise_level" in row
            and spec.metric in row
            and row.get("result_kind")
            in {
                "single_model_repeat_summary",
                "independent_seed_primary_summary",
            }
        ):
            groups[(row.get("case_id"), row.get("readout_id"))].append(row)
    if not groups:
        return []
    fig, ax = plt.subplots()
    for key, values in groups.items():
        ordered = sorted(values, key=lambda item: float(item["noise_level"]))
        x = [float(item["noise_level"]) for item in ordered]
        y = [float(item[spec.metric]) for item in ordered]
        error = [
            float(
                item.get(
                    f"{spec.metric}_seed_std",
                    item.get(f"{spec.metric}_repeat_std", 0.0),
                )
            )
            for item in ordered
        ]
        ax.errorbar(
            x,
            y,
            yerr=error,
            marker="o",
            label=f"case={key[0]}, readout={key[1]}",
        )
    ax.set_xlabel("noise_level")
    ax.set_ylabel(spec.metric)
    ax.set_xscale("symlog", linthresh=1e-12)
    ax.set_yscale("log")
    if len(groups) > 1:
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, root, spec.filename, spec.formats, spec.dpi)


def _learning_curve(
    spec: LearningCurveReporterSpec,
    rows: list[dict[str, Any]],
    root: Path,
) -> list[str]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("n_train") not in ("", None)
            and row.get(spec.metric) not in ("", None)
        ):
            groups[tuple(row.get(key) for key in spec.group_by)].append(row)
    if not groups:
        return []
    fig, ax = plt.subplots()
    for key, values in groups.items():
        ordered = sorted(values, key=lambda item: int(item["n_train"]))
        x = np.asarray([int(item["n_train"]) for item in ordered])
        y = np.asarray([float(item[spec.metric]) for item in ordered])
        lower_field = f"{spec.metric}_seed_ci95_low"
        upper_field = f"{spec.metric}_seed_ci95_high"
        label = ", ".join(
            f"{name}={value}"
            for name, value in zip(spec.group_by, key, strict=True)
        )
        if all(
            item.get(lower_field) not in ("", None)
            and item.get(upper_field) not in ("", None)
            for item in ordered
        ):
            lower = np.asarray(
                [float(item[lower_field]) for item in ordered]
            )
            upper = np.asarray(
                [float(item[upper_field]) for item in ordered]
            )
            ax.errorbar(
                x,
                y,
                yerr=np.vstack((y - lower, upper - y)),
                marker="o",
                capsize=3,
                label=label,
            )
            ax.fill_between(x, lower, upper, alpha=0.12)
        else:
            ax.plot(x, y, marker="o", label=label)
    ax.set_xlabel("n_train")
    ax.set_ylabel(spec.metric)
    ax.set_xscale(spec.xscale)
    ax.set_yscale(spec.yscale)
    if len(groups) > 1:
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, root, spec.filename, spec.formats, spec.dpi)


def _random_seed_distribution(
    spec: RandomFeatureSeedDistributionReporterSpec,
    seed_rows: list[dict[str, Any]],
    primary_rows: list[dict[str, Any]],
    root: Path,
) -> list[str]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        if (
            row.get("test_result_kind") == "independent_seed_realization"
            and row.get(spec.metric) not in ("", None)
            and row.get("seed") not in ("", None)
        ):
            groups[tuple(row.get(key) for key in spec.group_by)].append(row)
    if not groups:
        return []
    primary_by_group = {
        tuple(row.get(key) for key in spec.group_by): row
        for row in primary_rows
        if row.get("test_result_kind") == "independent_seed_metric_summary"
    }
    if set(groups) - set(primary_by_group):
        raise ValueError(
            "random-feature seed reporter has no matching primary summary"
        )

    fig, ax = plt.subplots()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, (key, rows) in enumerate(sorted(groups.items())):
        ordered = sorted(rows, key=lambda item: int(item["seed"]))
        seeds = np.asarray([int(item["seed"]) for item in ordered])
        values = np.asarray([float(item[spec.metric]) for item in ordered])
        summary = primary_by_group[key]
        mean = float(summary[f"{spec.metric}_seed_mean"])
        lower = float(summary[f"{spec.metric}_seed_ci95_low"])
        upper = float(summary[f"{spec.metric}_seed_ci95_high"])
        label = ", ".join(
            f"{name}={value}"
            for name, value in zip(spec.group_by, key, strict=True)
        )
        color = colors[index % len(colors)]
        if spec.plot == "scatter":
            ax.scatter(seeds, values, color=color, label=label)
            ax.hlines(
                mean,
                float(seeds.min()),
                float(seeds.max()),
                color=color,
                linestyle="--",
                label=f"{label} mean",
            )
            ax.fill_between(
                [float(seeds.min()), float(seeds.max())],
                lower,
                upper,
                color=color,
                alpha=0.12,
                label=f"{label} Student-t 95% CI",
            )
        elif spec.plot == "box":
            position = index + 1
            ax.boxplot(
                [values],
                positions=[position],
                widths=0.5,
            )
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels(
                [
                    ", ".join(
                        f"{name}={value}"
                        for name, value in zip(
                            spec.group_by,
                            group_key,
                            strict=True,
                        )
                    )
                    for group_key in sorted(groups)
                ],
                rotation=15,
                ha="right",
            )
            ax.errorbar(
                [position],
                [mean],
                yerr=[[mean - lower], [upper - mean]],
                color=color,
                marker="o",
                capsize=4,
                label=f"{label} mean / Student-t 95% CI",
            )
        else:
            ordered_values = np.sort(values)
            empirical = np.arange(1, len(values) + 1) / len(values)
            ax.step(
                ordered_values,
                empirical,
                where="post",
                color=color,
                label=label,
            )
            ax.axvline(mean, color=color, linestyle="--")
            ax.axvspan(lower, upper, color=color, alpha=0.12)

    if spec.plot == "scatter":
        ax.set_xlabel("evaluation seed (independent realization; no line fit)")
        ax.set_ylabel(spec.metric)
        ax.set_yscale(spec.yscale)
    elif spec.plot == "box":
        ax.set_xlabel("random-feature realization group")
        ax.set_ylabel(spec.metric)
        ax.set_yscale(spec.yscale)
    else:
        ax.set_xlabel(spec.metric)
        ax.set_ylabel("empirical cumulative probability")
        ax.set_xscale(spec.yscale)
        ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, root, spec.filename, spec.formats, spec.dpi)


def _capture_label(entry: dict[str, Any]) -> str:
    semantics = str(entry["prediction_semantics"])
    if semantics == "independent_seed_realization":
        semantics = f"member seed={entry['seed']}"
    elif semantics == "prediction_ensemble":
        semantics = "prediction ensemble"
    return (
        f"{entry['case_id']}/{entry['readout_id']} "
        f"({semantics})"
    )


def _representative_prediction_fields(
    spec: RepresentativePredictionFieldsReporterSpec,
    capture: dict[str, Any],
    root: Path,
) -> list[str]:
    rows: list[tuple[dict[str, Any], int]] = []
    for entry in capture["entries"]:
        for sample_index in range(int(entry["sample_ids"].numel())):
            rows.append((entry, sample_index))
    if not rows:
        return []
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(12, max(2.4 * len(rows), 3.0)),
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    for row_index, (entry, sample_index) in enumerate(rows):
        n_ref = int(entry["n_ref"])
        domain_length = float(entry["domain_length"])
        x = np.arange(n_ref, dtype=float) * domain_length / n_ref
        target = entry["target_field_n_ref"][sample_index].numpy()
        prediction = entry["prediction_field_n_ref"][sample_index].numpy()
        error = prediction - target
        sample_id = int(entry["sample_ids"][sample_index])
        for column, (values, label) in enumerate(
            (
                (target, "target field on n_ref"),
                (
                    prediction,
                    f"q={entry['q']} projected prediction on n_ref",
                ),
                (error, "prediction - n_ref target"),
            )
        ):
            axes[row_index, column].plot(x, values)
            axes[row_index, column].set_title(
                f"{_capture_label(entry)}; sample={sample_id}\n{label}",
                fontsize=8,
            )
            axes[row_index, column].grid(True, alpha=0.25)
        axes[row_index, 0].set_ylabel("L2-orthonormal field value")
    for axis in axes[-1]:
        axis.set_xlabel("periodic coordinate x")
    fig.suptitle(
        "Predeclared representative predictions; shared x-axis and row y-scale",
        fontsize=10,
    )
    return _save(fig, root, spec.filename, spec.formats, spec.dpi)


def _fourier_error_spectra(
    spec: FourierErrorSpectraReporterSpec,
    capture: dict[str, Any],
    root: Path,
) -> list[str]:
    metric_field = f"test_{spec.metric}"
    fig, ax = plt.subplots()
    for entry in capture["entries"]:
        x_tensor = (
            entry["mode_indices"]
            if spec.x_axis == "mode_index"
            else entry["physical_wavenumbers"]
        )
        ax.plot(
            x_tensor.numpy(),
            entry[metric_field].numpy(),
            marker="o",
            label=_capture_label(entry),
        )
    ax.set_xlabel(
        "Fourier mode index"
        if spec.x_axis == "mode_index"
        else "physical angular wavenumber"
    )
    ax.set_ylabel(
        f"{spec.metric}; sample aggregate="
        f"{capture['spectrum_definition']['sample_aggregate']}"
    )
    ax.set_yscale(spec.yscale)
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, root, spec.filename, spec.formats, spec.dpi)


def _stability_scatter(
    spec: ReadoutStabilityReporterSpec,
    rows: list[dict[str, Any]],
    root: Path,
) -> list[str]:
    x_field = (
        "weight_frobenius_norm"
        if spec.plot == "error_vs_norm"
        else "readout_design_covariance_retained_rank_condition"
    )
    filtered = [
        row
        for row in rows
        if row.get("result_kind")
        in {
            "single_model_repeat_summary",
            "independent_seed_repeat_summary",
        }
        and row.get(x_field) not in ("", None)
        and row.get(spec.metric) not in ("", None)
    ]
    if not filtered:
        return []
    fig, ax = plt.subplots()
    for row in filtered:
        ax.scatter(
            float(row[x_field]),
            float(row[spec.metric]),
            label=(
                f"{row.get('case_id')}/{row.get('readout_id')}/"
                f"seed={row.get('seed')}/noise={row.get('noise_level')}"
            ),
        )
    ax.set_xlabel(x_field)
    ax.set_ylabel(spec.metric)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if len(filtered) <= 12:
        ax.legend(fontsize=6)
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, root, spec.filename, spec.formats, spec.dpi)


def _heat_multiplier_comparison(
    spec: HeatMultiplierComparisonReporterSpec,
    rows: list[dict[str, Any]],
    root: Path,
) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("readout_id") != spec.readout_id:
            continue
        if spec.q is not None and int(row.get("q", -1)) != int(spec.q):
            continue
        if (
            row.get("coefficient_index") in ("", None)
            or row.get("ideal_readout_multiplier") in ("", None)
            or row.get("effective_learned_diagonal") in ("", None)
        ):
            continue
        groups[str(row.get("variant_id", row.get("case_id")))].append(row)
    if not groups:
        return []
    fig, ax = plt.subplots()
    for variant_id, values in groups.items():
        ordered = sorted(
            values,
            key=lambda item: int(item["coefficient_index"]),
        )
        coefficient_indices = [
            int(item["coefficient_index"]) for item in ordered
        ]
        ax.plot(
            coefficient_indices,
            [float(item["ideal_readout_multiplier"]) for item in ordered],
            linestyle="--",
            label=f"{variant_id}: ideal",
        )
        ax.plot(
            coefficient_indices,
            [float(item["effective_learned_diagonal"]) for item in ordered],
            marker=".",
            label=f"{variant_id}: effective",
        )
    ax.set_xlabel("real-Fourier coefficient index")
    ax.set_ylabel("readout multiplier")
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, root, spec.filename, spec.formats, spec.dpi)


def generate_reporters(
    reporters: Iterable[ReporterSpec],
    *,
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    random_seed_rows: list[dict[str, Any]] = (),
    prediction_capture: dict[str, Any] | None = None,
    multiplier_rows: list[dict[str, Any]],
    noise_rows: list[dict[str, Any]],
    output_dir: Path,
    skipped_rows: Iterable[dict[str, Any]] = (),
) -> list[str]:
    reporters = tuple(reporters)
    expected_all = expected_reporter_outputs(reporters)
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for reporter in reporters:
        if isinstance(reporter, MetricCurveReporterSpec):
            rendered = _metric_curve(
                reporter,
                _table_for_split(reporter.split, validation_rows, test_rows),
                output_dir,
            )
        elif isinstance(reporter, MetricMapReporterSpec):
            rendered = _metric_map(
                reporter,
                _table_for_split(reporter.split, validation_rows, test_rows),
                output_dir,
                skipped_rows=skipped_rows,
            )
        elif isinstance(reporter, ReadoutStabilityReporterSpec):
            if reporter.plot == "noise_curve":
                rendered = _noise_curve(reporter, noise_rows, output_dir)
            else:
                rendered = _stability_scatter(
                    reporter,
                    noise_rows,
                    output_dir,
                )
        elif isinstance(reporter, LearningCurveReporterSpec):
            rendered = _learning_curve(
                reporter,
                _table_for_split(
                    reporter.split,
                    validation_rows,
                    test_rows,
                ),
                output_dir,
            )
        elif isinstance(
            reporter,
            RandomFeatureSeedDistributionReporterSpec,
        ):
            rendered = _random_seed_distribution(
                reporter,
                list(random_seed_rows),
                test_rows,
                output_dir,
            )
        elif isinstance(
            reporter,
            RepresentativePredictionFieldsReporterSpec,
        ):
            if prediction_capture is None:
                raise ValueError(
                    "representative prediction reporter requires a completed "
                    "prediction capture"
                )
            rendered = _representative_prediction_fields(
                reporter,
                prediction_capture,
                output_dir,
            )
        elif isinstance(reporter, FourierErrorSpectraReporterSpec):
            if prediction_capture is None:
                raise ValueError(
                    "Fourier spectrum reporter requires a completed "
                    "prediction capture"
                )
            rendered = _fourier_error_spectra(
                reporter,
                prediction_capture,
                output_dir,
            )
        elif isinstance(reporter, HeatMultiplierComparisonReporterSpec):
            rendered = _heat_multiplier_comparison(
                reporter,
                multiplier_rows,
                output_dir,
            )
        else:
            raise TypeError(f"unsupported reporter type: {type(reporter).__name__}")
        expected = [
            f"{reporter.filename}.{extension}"
            for extension in reporter.formats
        ]
        if rendered != expected:
            raise ValueError(
                f"reporter {reporter.kind!r} for {reporter.filename!r} "
                f"produced {rendered!r}; expected exactly {expected!r}"
            )
        for name in rendered:
            path = output_dir / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"reporter output is missing or unsafe: {name}")
        created.extend(rendered)
    actual = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    if actual != sorted(expected_all) or sorted(created) != sorted(expected_all):
        raise ValueError(
            "reporter output directory contains missing, duplicate, or "
            "unexpected files"
        )
    return created
