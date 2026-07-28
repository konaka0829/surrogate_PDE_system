from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pol.config.models import (
    MetricCurveReporterSpec,
    NoiseCurveReporterSpec,
    ReporterSpec,
    ResolutionMapReporterSpec,
)


def _save(fig, root: Path, filename: str, formats: Iterable[str], dpi: int) -> list[str]:
    names: list[str] = []
    stem = Path(filename).stem
    for extension in formats:
        name = f"{stem}.{extension}"
        fig.savefig(root / name, dpi=dpi, bbox_inches="tight")
        names.append(name)
    plt.close(fig)
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


def _resolution_map(
    spec: ResolutionMapReporterSpec,
    rows: list[dict[str, Any]],
    root: Path,
) -> list[str]:
    selected = [
        row
        for row in rows
        if row.get("readout_id") == spec.readout_id
        and spec.x in row
        and spec.y in row
        and spec.metric in row
    ]
    if not selected:
        return []
    x_values = sorted({float(row[spec.x]) for row in selected})
    y_values = sorted({float(row[spec.y]) for row in selected})
    buckets: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in selected:
        buckets[(float(row[spec.x]), float(row[spec.y]))].append(
            float(row[spec.metric])
        )
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for yi, y in enumerate(y_values):
        for xi, x in enumerate(x_values):
            values = buckets.get((x, y), [])
            if values:
                matrix[yi, xi] = float(np.mean(values))
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, origin="lower", aspect="auto")
    ax.set_xticks(range(len(x_values)), [f"{value:g}" for value in x_values])
    ax.set_yticks(range(len(y_values)), [f"{value:g}" for value in y_values])
    ax.set_xlabel(spec.x)
    ax.set_ylabel(spec.y)
    fig.colorbar(image, ax=ax, label=spec.metric)
    return _save(fig, root, spec.filename, spec.formats, spec.dpi)


def _noise_curve(
    spec: NoiseCurveReporterSpec,
    rows: list[dict[str, Any]],
    root: Path,
) -> list[str]:
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "noise_level" in row and spec.metric in row:
            groups[(row.get("case_id"), row.get("readout_id"))].append(row)
    if not groups:
        return []
    fig, ax = plt.subplots()
    for key, values in groups.items():
        ordered = sorted(values, key=lambda item: float(item["noise_level"]))
        ax.plot(
            [float(item["noise_level"]) for item in ordered],
            [float(item[spec.metric]) for item in ordered],
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


def generate_reporters(
    reporters: Iterable[ReporterSpec],
    *,
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    noise_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for reporter in reporters:
        if isinstance(reporter, MetricCurveReporterSpec):
            created.extend(
                _metric_curve(
                    reporter,
                    _table_for_split(reporter.split, validation_rows, test_rows),
                    output_dir,
                )
            )
        elif isinstance(reporter, ResolutionMapReporterSpec):
            created.extend(
                _resolution_map(
                    reporter,
                    _table_for_split(reporter.split, validation_rows, test_rows),
                    output_dir,
                )
            )
        elif isinstance(reporter, NoiseCurveReporterSpec):
            created.extend(_noise_curve(reporter, noise_rows, output_dir))
        else:
            raise TypeError(f"unsupported reporter type: {type(reporter).__name__}")
    return created
