#!/usr/bin/env bash
set -euo pipefail

python -m pol validate configs/validation/foundation_smoke.json
python -m pol data build configs/datasets/heat_smoke.json
python -m pol data build configs/datasets/burgers_smoke.json
python -m pol run studies/heat_readout_calibration_smoke.json
python -m pol run studies/surrogate_parameter_time_smoke.json
python -m pol run studies/observation_output_map_smoke.json
python -m pol run studies/finite_surrogate_resolution_map_smoke.json
