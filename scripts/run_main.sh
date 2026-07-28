#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
Main profiles are computationally expensive. This script runs high-resolution
foundation validation, builds the reference datasets, and executes every
configured publication study. Press Ctrl-C now to stop.
MSG
sleep 5

python -m pol validate configs/validation/foundation_main.json
python -m pol data build configs/datasets/heat_main.json
python -m pol data build configs/datasets/burgers_main.json
python -m pol run studies/heat_readout_calibration.json
python -m pol run studies/surrogate_parameter_time.json
python -m pol run studies/observation_output_map.json
python -m pol run studies/finite_surrogate_resolution_map.json
