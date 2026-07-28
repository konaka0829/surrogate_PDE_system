# Migration from experiment-specific runners

The former repository treated publication labels and numbered experimental
questions as software architecture. Separate configuration classes, recipes,
runners, artifact schemas, plotting registries, and sweep plugins existed for
each experiment. A top-level dispatcher selected among them, so the apparent
unified command still executed separate applications internally.

The refactor replaces that organization as follows.

| Former responsibility | Current responsibility |
|---|---|
| numbered foundation experiment | `pol.validation` and ordinary unit tests |
| experiment-specific target archive | `pol.data` explicitly validation-bound reference dataset |
| heat calibration runner | generic study + heat diagnostic |
| surrogate parameter/time runner | generic study + coordinate search |
| scalar runner | one-cell `StudySpec` |
| matrix runner and plugin | `global_axes`, variants, and generic search |
| experiment-specific plot recipe | generic reporter over long-form tables |
| publication model numbers | readout display metadata |
| experiment-specific state cache | content-addressed `feature_states` artifact |
| experiment-specific artifact versions | semantic artifact-kind schemas |

## What remains study-specific

The `studies/` directory records declarative combinations needed to answer
scientific questions. It is deliberately outside `pol/`. The package does not
import this directory, and moving or copying a study file does not change the
implementation path.

The former mixed resolution/observation sweep has been split into two studies:

```text
observation_output_map
    J × q, with n_tar and n_sur fixed

finite_surrogate_resolution_map
    n_tar × n_sur, with J and q fixed
```

This prevents observation budget, output bandwidth, finite input information,
and surrogate numerical resolution from changing in the same phase diagram.

## Current command mapping

```text
foundation validation
    python -m pol validate configs/validation/foundation_smoke.json

heat readout calibration
    python -m pol run studies/heat_readout_calibration_smoke.json

surrogate parameter/readout-time study
    python -m pol run studies/surrogate_parameter_time_smoke.json

observation/output phase diagram
    python -m pol run studies/observation_output_map_smoke.json

finite-data/surrogate-resolution phase diagram
    python -m pol run studies/finite_surrogate_resolution_map_smoke.json
```

Old compatibility commands are intentionally not retained. Maintaining two
execution paths would reintroduce the ambiguity that the refactor removes.
