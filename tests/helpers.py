from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_tiny_stack(
    root: Path,
    *,
    global_q_values: tuple[int, ...] = (9,),
    generate_plots: bool = False,
) -> tuple[Path, Path, Path]:
    artifact_root = root / "artifacts"
    output_root = root / "outputs"

    validation_path = write_json(
        root / "validation.json",
        {
            "schema_version": "pol-validation-v1",
            "name": "tiny_foundation",
            "artifact_root": str(artifact_root),
            "profile": "test",
            "domain": {"length": 1.0},
            "samples": {
                "total_samples": 12,
                "n_train": 8,
                "n_validation": 2,
                "n_test": 2,
                "seed": 17,
                "dtype": "float64",
                "device": "cpu",
                "initial_condition": {
                    "kind": "periodic_grf",
                    "gamma": 2.0,
                    "tau": 5.0,
                    "sigma": 0.5,
                    "mean": 0.0,
                },
                "preprocessing": "l2_scaling_only",
            },
            "reference_evolution": {
                "system": {
                    "kind": "burgers",
                    "nu": 0.05,
                    "advection_coefficient": 1.0,
                    "solver": "split_step",
                    "dt": 0.01,
                    "fine_dt": 0.0025,
                    "dealias": True,
                },
                "time": 0.02,
            },
            "calibration_sample_ids": [0, 1],
            "reference_nx_candidates": [16, 32],
            "time_candidates": [
                {
                    "solver": "split_step",
                    "dt": 0.01,
                    "fine_dt": 0.005,
                    "dealias": True,
                },
                {
                    "solver": "split_step",
                    "dt": 0.01,
                    "fine_dt": 0.0025,
                    "dealias": True,
                },
            ],
            "q_reference_check": 9,
            "reference_tolerances": {
                "mean_relative_l2": 1.0,
                "max_relative_l2": 1.0,
                "low_mode_relative_l2": 1.0,
            },
            "full_interface": {"n_tar": 16, "n_sur": 32, "J": 32, "q": 9},
            "reduced_observation": {"J": 8, "q": 5},
        },
    )

    dataset_path = write_json(
        root / "dataset.json",
        {
            "schema_version": "pol-dataset-v1",
            "name": "tiny_heat_dataset",
            "artifact_root": str(artifact_root),
            "validation_spec": str(validation_path),
            "reference_nx": 32,
            "target": {
                "system": {"kind": "heat", "nu": 0.1},
                "time": 0.1,
            },
            "batch_size": 12,
        },
    )

    global_axes = []
    if len(global_q_values) > 1 or global_q_values[0] != 9:
        global_axes = [{"path": "output.q", "values": list(global_q_values)}]

    study_path = write_json(
        root / "study.json",
        {
            "schema_version": "pol-study-v1",
            "name": "tiny_heat_study",
            "output_root": str(output_root),
            "artifact_root": str(artifact_root),
            "profile": "test",
            "dataset_spec": str(dataset_path),
            "base_trial": {
                "input": {"n_tar": 16, "resampling": "spectral"},
                "feature": {
                    "evolution": {
                        "system": {"kind": "heat", "nu": 0.05},
                        "time": 0.1,
                    },
                    "n_sur": 32,
                    "observation": {
                        "kind": "equispaced_points",
                        "J": 32,
                        "l2_scale": True,
                    },
                },
                "output": {"kind": "real_fourier", "q": 9},
                "readouts": [
                    {
                        "id": "direct",
                        "kind": "direct_fourier_decoder",
                        "display_name": "Direct decoder",
                    },
                    {
                        "id": "affine",
                        "kind": "affine_ridge",
                        "display_name": "Affine ridge",
                        "zetas": [0.0, 1e-8],
                        "tie_tolerance": 1e-12,
                        "tie_break": "largest_zeta",
                        "svd_rcond": None,
                    },
                    {
                        "id": "random",
                        "kind": "random_feature_ridge",
                        "display_name": "Random-feature ridge",
                        "activation": "tanh",
                        "widths": [2],
                        "weight_scales": [0.5],
                        "bias_scales": [0.1],
                        "selection_seeds": [11],
                        "evaluation_seeds": [21],
                        "zetas": [1e-8],
                        "tie_tolerance": 1e-12,
                        "svd_rcond": None,
                    },
                ],
            },
            "variants": [
                {
                    "id": "heat",
                    "display_name": "Heat features",
                    "overrides": {},
                    "search": {"kind": "static"},
                }
            ],
            "global_axes": global_axes,
            "selection": {
                "metric": "validation_field_relative_l2_mean",
                "tie_tolerance": 1e-12,
                "tie_break": "first_in_config_order",
                "representative_readout": "affine",
                "freeze_before_test": True,
            },
            "diagnostics": [{"kind": "heat_multiplier", "identifiable_variance_floor": 1e-14}],
            "reporters": (
                [
                    {
                        "kind": "metric_curve",
                        "filename": "validation_error_vs_q",
                        "x": "q",
                        "metric": "validation_field_relative_l2_mean",
                        "split": "validation",
                        "group_by": ["variant_id", "readout_id"],
                        "xscale": "linear",
                        "yscale": "log",
                        "formats": ["png"],
                        "dpi": 80,
                    }
                ]
                if generate_plots
                else []
            ),
            "execution": {
                "torch_threads": 1,
                "batch_size": 3,
                "invalid_trial_policy": "error",
                "cache_states": True,
                "generate_plots": generate_plots,
            },
        },
    )
    return validation_path, dataset_path, study_path
