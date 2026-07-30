#!/usr/bin/env bash
set -euo pipefail

python3 -m pol validate configs/validation/foundation_smoke.json
python3 -m pol validate configs/validation/heat_smoke.json
python3 -m pol validate configs/validation/reaction_diffusion_smoke.json
python3 -m pol data build configs/datasets/heat_smoke.json
python3 -m pol data build configs/datasets/burgers_smoke.json
python3 -m pol run studies/heat_readout_calibration_smoke.json
python3 -m pol run studies/surrogate_parameter_time_coordinate_search_smoke.json
python3 -m pol run studies/surrogate_parameter_time_landscape_smoke.json
python3 -m pol selection inspect studies/surrogate_parameter_time_landscape_smoke.json
python3 -m pol selection verify studies/dynamic_feature_baseline_comparison_smoke.json
python3 -m pol run studies/dynamic_feature_baseline_comparison_smoke.json
python3 -m pol digital-baseline digital_baselines/fno1d_smoke.json
python3 -m pol selection verify studies/readout_stability_noise_smoke.json
python3 -m pol run studies/readout_stability_noise_smoke.json
python3 -m pol selection verify studies/learning_curve_smoke.json
python3 -m pol run studies/learning_curve_smoke.json
python3 -m pol selection verify studies/random_feature_seed_statistics_smoke.json
python3 -m pol run studies/random_feature_seed_statistics_smoke.json
python3 -m pol selection verify studies/observation_output_budget_smoke.json
python3 -m pol selection verify studies/input_simulation_resolution_smoke.json
python3 -m pol run studies/observation_output_budget_smoke.json
python3 -m pol run studies/input_simulation_resolution_smoke.json
python3 -m pol report reports/surrogate_operator_summary_smoke.json
