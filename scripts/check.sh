#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q pol tests
pytest -q
python -m pol run studies/heat_readout_calibration_smoke.json --plan >/dev/null
python -m pol run studies/surrogate_parameter_time_smoke.json --plan >/dev/null
python -m pol run studies/observation_output_map_smoke.json --plan >/dev/null
python -m pol run studies/finite_surrogate_resolution_map_smoke.json --plan >/dev/null
