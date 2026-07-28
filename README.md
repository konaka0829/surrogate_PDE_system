# Surrogate-Dynamics Operator Learning for PDEs

This repository implements operator learning in which the continuous-time
evolution of a surrogate PDE is used as a dynamic feature generator. A target
PDE dataset supplies finite initial fields and future states; the surrogate
system evolves only the finite input available to the model; an observation
operator produces features; and a fixed or learned readout predicts a finite
Fourier representation of the target state.

The software architecture is organized around reusable scientific components
and a single `StudyRunner`. Publication labels and figure numbers do not appear
in the executable core. A scalar calculation, a parameter sweep, a resolution
map, and a comparison between different surrogate systems all use the same
trial and study execution path.

## Design principles

- `n_ref`, `n_tar`, `n_sur`, `J`, and `q` are independent quantities with
  separate owners.
- There is no general `n_tar <= J` assumption. The implemented interface only
  requires the relevant representability constraints, including `J <= n_sur`
  and `q <= n_tar`.
- The surrogate initial state is constructed from the finite `n_tar` input.
  Discarded reference-grid modes are never recovered or exposed to the feature
  generator.
- Scientific configuration is strict: unknown keys are rejected with their
  JSON path.
- Hyperparameter and system selection use the training and validation splits
  only.
- The selection record and frozen evaluation plan are written, hashed, and read
  back before the first test state solve or test metric.
- Validation products, datasets, feature states, and completed studies are
  content addressed and verified by exact byte manifests.
- Directory publication is transactional: incomplete staging trees never
  replace a valid artifact or study run.

## Repository layout

```text
pol/
├── config/       strict, discriminated configuration models and loaders
├── math/         periodic grids, spectral resampling, real Fourier maps
├── systems/      heat, Burgers, and reaction-diffusion evolution systems
├── data/         validated reference datasets and finite-resolution views
├── learning/     observations, fixed decoder, ridge, random features, metrics
├── validation/   independent algebraic and numerical foundation validation
├── study/        unified trial, search, selection, convergence, freeze/test flow
├── artifacts/    content-addressed artifact store
├── plotting/     generic reporters over long-form result tables
├── numerics/     import-light numerical kernels
└── runtime/      atomic I/O, hashing, and directory transactions

configs/
├── validation/   foundation-validation profiles
└── datasets/     target-dataset profiles

studies/
└── *.json        question-based combinations of reusable components
```

`studies/` contains only declarative combinations of reusable components. The
core package does not import it and contains no publication- or
experiment-number control flow.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e '.[test]'
```

For an execution-only installation:

```bash
python -m pip install -e .
```

## Command-line interface

There are four commands.

```bash
pol validate SPEC
pol data build SPEC
pol run STUDY
pol verify PATH
```

The equivalent module form is available when the console script has not yet
been installed:

```bash
python -m pol --help
```

### 1. Validate the numerical foundation

```bash
python -m pol validate configs/validation/foundation_smoke.json
```

The validation layer is independent of prediction studies. It checks periodic
resampling and Nyquist handling, the real Fourier projector, finite-input/no-
leak behavior, valid use of `n_tar < J`, fixed-decoder behavior and aliasing,
and Burgers reference convergence. A passing certificate and master initial-
condition archive are published under `artifacts/validations/<content-hash>/`.

### 2. Build or reuse a target dataset

```bash
python -m pol data build configs/datasets/burgers_smoke.json
```

A dataset is built only from a passing validation certificate. Its split IDs,
reference inputs, reference targets, target-solver metadata, and tensor hashes
are stored under `artifacts/datasets/<content-hash>/`.

### 3. Run a study

A single condition is a one-cell study. Sweeps are produced by `global_axes`,
variant overrides, or a variant search specification; they are not separate
runner types.

```bash
python -m pol run studies/heat_readout_calibration_smoke.json
python -m pol run studies/surrogate_parameter_time_smoke.json
python -m pol run studies/observation_output_map_smoke.json
python -m pol run studies/finite_surrogate_resolution_map_smoke.json
```

Inspect the expansion without creating artifacts or output directories:

```bash
python -m pol run studies/surrogate_parameter_time_smoke.json --plan
```

Override an existing field without editing JSON:

```bash
python -m pol run studies/surrogate_parameter_time_smoke.json \
  --set base_trial.feature.observation.J=8 \
  --set base_trial.output.q=9
```

Only existing paths can be overridden, and the resolved study is validated
again before execution.

Useful execution modes:

```bash
# Recompute and transactionally replace the same content-addressed run.
python -m pol run STUDY.json --force

# Recreate figures from a verified completed run with the same study identity.
python -m pol run STUDY.json --plots-only
```

### 4. Verify an artifact or study run

```bash
python -m pol verify artifacts/validations/<hash>
python -m pol verify outputs/studies/<study>/<profile-hash>
```

Verification rejects symlinks, missing or extra files, and byte changes relative
to the manifest.

## Study semantics

A `TrialSpec` defines one finite operator-learning pipeline:

```text
finite input at n_tar
    -> feature-state encoding at n_sur
    -> optional surrogate-PDE evolution
    -> observation in J dimensions
    -> fixed / affine-ridge / random-feature-ridge readout
    -> q real Fourier coefficients
    -> reconstructed field and metrics
```

The unqualified `field_*` metrics reconstruct the prediction on the
convergence-validated `n_ref` grid and use that grid as quadrature for the
continuous periodic `L2` norm. `data_field_*` metrics are also reported on the
finite `n_tar` target grid. This distinction prevents a change in `n_tar` from
silently changing the meaning of the principal error metric.

A `StudySpec` adds:

- publication-independent variants;
- global sweep axes;
- static, grid, or coordinate search;
- validation-only selection;
- optional surrogate-resolution convergence;
- diagnostics such as heat multipliers or observation noise;
- generic reporters over result tables.

The implemented feature-generator kinds are:

- `pde_dynamics`: evolve the finite input with a configured surrogate PDE;
- `static_input`: observe the encoded finite input without evolution, for a
  no-dynamics baseline through the same trial and study path.

The implemented readout kinds are:

- `direct_fourier_decoder`: fixed, untrained decoding;
- `affine_ridge`: centered affine ridge regression;
- `random_feature_ridge`: skip-connected random nonlinear features followed by
  an affine ridge readout.

Display labels such as “Model 1”, “Model 2”, and “Model 3” live only in study
metadata and do not define implementation branches.

## Output contract

A completed study is written under:

```text
outputs/studies/<study-name>/<profile>-<content-hash>/
```

Important files include:

- `resolved_study.json`
- `dataset_reference.json`
- `validation_trials.csv`
- `selection_record.json`
- `convergence.csv`
- `frozen_models.pt`
- `frozen_evaluation_plan.json`
- `test_metrics.csv`
- diagnostic CSV files
- `events.json`
- `run_summary.json`
- `figures/`
- `manifest.json`

`events.json` records the durability boundary. In a valid run,
`freeze_read_back` appears before `first_test_state_solve` and
`first_test_metric`.

## Included profiles

Smoke profiles are intentionally small and are used for integration checks:

- `heat_readout_calibration_smoke.json`
- `surrogate_parameter_time_smoke.json`
- `observation_output_map_smoke.json`
- `finite_surrogate_resolution_map_smoke.json`

The two phase diagrams are deliberately separate. The observation/output map
varies `J × q` while holding `n_tar` and `n_sur` fixed. The finite/surrogate
resolution map varies `n_tar × n_sur` while holding `J` and `q` fixed.

The corresponding non-smoke profiles retain the high-resolution research
settings. They can require substantial CPU time, storage, and memory and are
not run by the test suite.

## Tests

```bash
pytest -q
```

The suite covers Fourier and Nyquist behavior, finite-input information
isolation, strict configuration, dimension independence, fixed and learned
readouts, artifact tamper detection, foundation validation, dataset splitting,
unified scalar/sweep planning, selection freezing, test-order enforcement,
plot regeneration, CLI behavior, and the absence of publication-number
namespaces from the core package.

For a complete local smoke sequence:

```bash
./scripts/run_smoke.sh
```

Additional architectural details are in
[`docs/architecture.md`](docs/architecture.md), and the mapping from the former
experiment-specific layout is documented in
[`docs/migration.md`](docs/migration.md).
