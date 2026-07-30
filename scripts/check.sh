#!/usr/bin/env bash
set -euo pipefail

python3 -m compileall -q pol tests
pytest -q
python3 scripts/plan_main.py >/dev/null
python3 -m pol run studies/heat_readout_calibration_smoke.json --plan >/dev/null
python3 -m pol run studies/surrogate_parameter_time_coordinate_search_smoke.json --plan >/dev/null
python3 -m pol run studies/surrogate_parameter_time_landscape_smoke.json --plan >/dev/null
python3 -m pol run studies/dynamic_feature_baseline_comparison_smoke.json --plan >/dev/null
python3 -m pol digital-baseline digital_baselines/fno1d_smoke.json --plan >/dev/null
python3 -m pol run studies/readout_stability_noise_smoke.json --plan >/dev/null
python3 -m pol run studies/learning_curve_smoke.json --plan >/dev/null
python3 -m pol run studies/random_feature_seed_statistics_smoke.json --plan >/dev/null
python3 -m pol run studies/observation_output_budget_smoke.json --plan >/dev/null
python3 -m pol run studies/input_simulation_resolution_smoke.json --plan >/dev/null
python3 -c "from pathlib import Path; from pol.config.loader import load_report_spec; root=Path.cwd(); load_report_spec('reports/surrogate_operator_summary_smoke.json', repo_root=root); load_report_spec('reports/surrogate_operator_summary.json', repo_root=root)"
