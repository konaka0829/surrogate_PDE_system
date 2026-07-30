#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'MSG'
Usage: scripts/run_main.sh --stage STAGE

Read-only stage:
  audit

Main execution stages (one stage per invocation):
  validation
  datasets
  heat-calibration
  parameter-time-search
  parameter-time-landscape
  baseline-comparison
  digital-baseline
  readout-stability
  learning-curve
  random-feature-seeds
  observation-output
  input-simulation
  cross-run-report

Every main execution stage requires:
  POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage STAGE

There is deliberately no "all" stage. See docs/production_runbook.md for
dependency order, checkpoints, verification, resume, force, and recovery.
MSG
}

if [[ $# -ne 2 || "$1" != "--stage" ]]; then
  usage
  exit 2
fi

stage="$2"
if [[ "$stage" == "audit" ]]; then
  python3 scripts/plan_main.py
  exit 0
fi

case "$stage" in
  validation|datasets|heat-calibration|parameter-time-search|parameter-time-landscape|baseline-comparison|digital-baseline|readout-stability|learning-curve|random-feature-seeds|observation-output|input-simulation|cross-run-report)
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ "${POL_CONFIRM_MAIN:-}" != "YES" ]]; then
  echo "Refusing main execution: set POL_CONFIRM_MAIN=YES for this one stage." >&2
  exit 3
fi

case "$stage" in
  validation)
    python3 -m pol validate configs/validation/foundation_main.json
    python3 -m pol validate configs/validation/heat_main.json
    python3 -m pol validate configs/validation/reaction_diffusion_main.json
    ;;
  datasets)
    python3 -m pol data build configs/datasets/heat_main.json
    python3 -m pol data build configs/datasets/burgers_main.json
    ;;
  heat-calibration)
    python3 -m pol run studies/heat_readout_calibration.json
    ;;
  parameter-time-search)
    python3 -m pol run studies/surrogate_parameter_time_coordinate_search.json
    ;;
  parameter-time-landscape)
    python3 -m pol run studies/surrogate_parameter_time_landscape.json
    python3 -m pol selection inspect studies/surrogate_parameter_time_landscape.json
    ;;
  baseline-comparison)
    python3 -m pol selection verify studies/dynamic_feature_baseline_comparison.json
    python3 -m pol run studies/dynamic_feature_baseline_comparison.json
    ;;
  digital-baseline)
    python3 -m pol digital-baseline digital_baselines/fno1d.json
    ;;
  readout-stability)
    python3 -m pol selection verify studies/readout_stability_noise.json
    python3 -m pol run studies/readout_stability_noise.json
    ;;
  learning-curve)
    python3 -m pol selection verify studies/learning_curve.json
    python3 -m pol run studies/learning_curve.json
    ;;
  random-feature-seeds)
    python3 -m pol selection verify studies/random_feature_seed_statistics.json
    python3 -m pol run studies/random_feature_seed_statistics.json
    ;;
  observation-output)
    python3 -m pol selection verify studies/observation_output_budget.json
    python3 -m pol run studies/observation_output_budget.json
    ;;
  input-simulation)
    python3 -m pol selection verify studies/input_simulation_resolution.json
    python3 -m pol run studies/input_simulation_resolution.json
    ;;
  cross-run-report)
    python3 -m pol report reports/surrogate_operator_summary.json
    ;;
esac
