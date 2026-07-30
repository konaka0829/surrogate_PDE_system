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
| surrogate parameter/time runner | generic study + separate coordinate-search and Cartesian-landscape configurations |
| scalar runner | one-cell `StudySpec` |
| matrix runner and plugin | `global_axes`, variants, and generic search |
| experiment-specific plot recipe | generic reporter over long-form tables |
| in-memory post-run figure path | verified-run read-only report transaction |
| manually combined publication figures/tables | `pol-report-v1` over verified completed runs |
| neural operator forced into a physical readout registry | independent `pol.digital_baselines` adapter sharing dataset/split/metrics |
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
observation_output_budget
    J × q, with n_tar and n_sur fixed

input_simulation_resolution
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

surrogate parameter/readout-time coordinate search
    python -m pol run studies/surrogate_parameter_time_coordinate_search_smoke.json

surrogate parameter/readout-time Cartesian landscape
    python -m pol run studies/surrogate_parameter_time_landscape_smoke.json

inspect the completed landscape selection without execution
    python -m pol selection inspect studies/surrogate_parameter_time_landscape_smoke.json

verify the downstream completed-selection binding without execution
    python -m pol selection verify studies/observation_output_budget_smoke.json

independent random-feature seed statistics
    python -m pol run studies/random_feature_seed_statistics_smoke.json

regenerate figures from an existing verified run only
    python -m pol run studies/random_feature_seed_statistics_smoke.json --plots-only

observation/output budget phase diagram
    python -m pol run studies/observation_output_budget_smoke.json

finite-data/surrogate-resolution phase diagram
    python -m pol run studies/input_simulation_resolution_smoke.json

cross-run phase diagrams and baseline table
    python -m pol report reports/surrogate_operator_summary_smoke.json

verify a cross-run report
    python -m pol report verify outputs/reports/<report>/<profile-hash>

plan/run the independent digital FNO baseline
    python -m pol digital-baseline digital_baselines/fno1d_smoke.json --plan
    python -m pol digital-baseline digital_baselines/fno1d_smoke.json

verify a completed digital baseline
    python -m pol digital-baseline verify outputs/digital_baselines/<name>/<profile-hash>
```

Old compatibility commands are intentionally not retained. Maintaining two
execution paths would reintroduce the ambiguity that the refactor removes.

Study configurations through `pol-study-v6` are accepted. A variant that consumes
a validation-selected feature condition declares `completed_study_selection`
with a source study spec, source variant/readout, and allowlisted import paths.
`pol-study-v2` runs do not contain this provenance and are rejected rather than
silently accepted as selection-bound results.

Cross-run reporting is intentionally not another study runner. Report sources
are resolved from strict study specifications to their exact existing
content-addressed runs, verified, and then read as immutable CSV/JSON inputs.
The report identity includes the semantic report declaration, source
run/scientific/manifest hashes, and software environment, while excluding
source and destination storage paths. A rendering failure is confined to a
staging transaction and cannot replace an existing verified report or mutate
a source run.

The FNO path is intentionally not a `StudySpec` readout. It resolves a verified
physical comparison source, then reuses the exact dataset/split and metric
primitives while owning neural-network optimization, validation checkpoint
selection, frozen checkpoint read-back, independent training-seed evaluation,
and its own transactional artifact. DeepONet remains unimplemented.
