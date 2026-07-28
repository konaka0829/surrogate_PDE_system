# Current implementation inventory

This inventory describes the code present at Phase P0-00. It is a
characterization of the current implementation, not a claim that every
implemented behavior is scientifically final.

## Validation

`pol.validation.runner` implements one content-addressed foundation-validation
flow. A passing certificate currently covers:

- periodic spectral mode transfer, even-grid Nyquist handling, and discard of
  an unrepresentable high mode;
- real Fourier analysis/synthesis recovery on a band-limited field;
- finite-interface shapes, construction of `n_sur` input through `n_tar`, and
  a synthetic no-high-frequency-leak check;
- an explicit valid case with `n_tar <= J`, demonstrating that the dimensions
  are independent;
- fixed Fourier decoding for full and reduced band-limited observations, plus
  an aliasing counterexample;
- Burgers spatial, temporal, and joint reference-convergence comparisons on
  configured calibration samples.

The validation artifact contains the resolved specification, detailed checks,
a passing certificate, convergence rows, and a master initial-condition
archive. Exact-byte manifests and transactional publication are implemented by
the artifact layer.

## Data

The implemented data path includes:

- deterministic periodic Gaussian-random-field initial conditions;
- a content-addressed master initial-condition archive;
- deterministic, disjoint train/validation/test ID splits;
- reference datasets evolved by a registered target system;
- finite `n_tar` views derived by periodic spectral resampling;
- `q`-dimensional real Fourier training targets;
- separate finite targets and `n_ref` reference targets;
- construction of the `n_sur` feature input only from the finite `n_tar`
  tensor.

Checked-in dataset profiles currently provide heat and viscous Burgers targets,
each in smoke and main sizes. The generic system schema can also describe a
reaction-diffusion target, but there is no checked-in reaction-diffusion target
dataset profile.

## Evolution systems and feature generators

Registered systems are:

| Semantic kind | Current implementation |
|---|---|
| `heat` | exact discrete spectral heat flow |
| `burgers` | split-step/semi-implicit alias and Fourier pseudospectral ETDRK4/`etdrk4` alias |
| `reaction_diffusion` | semi-implicit spectral Euler with optional two-thirds nonlinear filtering |

Feature-generator kinds are `pde_dynamics` and `static_input`. The static
baseline uses the same finite-input, cache, observation, readout, and
evaluation path, although no standalone checked-in study JSON currently
selects it.

The implemented observation is L2-scaled equispaced point observation of the
source-grid trigonometric interpolant.

## Readouts

The unified trial engine implements:

- `direct_fourier_decoder`: an untrained fixed decoder (displayed as Model 1
  in some study metadata);
- `affine_ridge`: centered affine ridge regression with validation-selected
  regularization and an SVD minimum-norm path at zero regularization (Model 2);
- `random_feature_ridge`: skip-connected random nonlinear features followed
  by centered affine ridge regression, with separate selection and evaluation
  seed sets (Model 3).

Display names do not control dispatch.

## Studies and scientific questions

All scalar and swept calculations use `pol.study.runner.StudyRunner`'s unified
execution path (implemented by `run_study`); a scalar calculation is a one-cell
study.

| Study JSON family | Scientific question currently represented |
|---|---|
| `heat_readout_calibration` | For an analytically transparent heat target, how do fixed, affine, and random-feature readouts behave as output bandwidth changes, including effective heat multipliers and observation-noise sensitivity? |
| `surrogate_parameter_time` | Which surrogate system parameters and readout time are selected on validation data, and how do readout classes compare after surrogate-resolution convergence checking? |
| `observation_output_map` | How does performance vary over the distinct `J x q` observation/output budget at fixed `n_tar` and `n_sur`? |
| `finite_surrogate_resolution_map` | How does performance vary over the distinct `n_tar x n_sur` finite-data/surrogate-resolution budget at fixed `J` and `q`? |

Each family has a smoke and a main profile. Only smoke/tiny profiles belong in
automated checks or maintenance work.

Supported search modes are static, Cartesian grid, and two-axis coordinate
search. Implemented diagnostics are heat-multiplier inspection,
surrogate-resolution convergence, and observation-noise robustness. Generic
reporters provide metric curves, resolution maps, and noise curves.

## Metrics

The prediction metric set currently stores:

- coefficient MSE;
- samplewise coefficient relative L2, aggregated by mean, median, and maximum;
- reference-field absolute and relative periodic L2, each aggregated by mean,
  median, and maximum;
- finite-data-field absolute and relative periodic L2, each aggregated by
  mean, median, and maximum;
- reference-field and finite-data representation-floor relative L2, each
  aggregated by mean, median, and maximum.

Validation selection can currently use
`validation_field_relative_l2_mean` or `validation_coefficient_mse`.
Convergence rows separately report terminal-state, observed-feature, and
prediction relative errors (mean and maximum). Heat-multiplier and
noise-robustness diagnostics have their own long-form tables.

## Implemented artifact and evaluation controls

- Strict Pydantic models reject unknown scientific configuration keys.
- Scientific identities exclude storage roots and include environment and
  upstream artifact identities.
- Validation products, datasets, feature states, and study runs are
  content-addressed and exact-byte verified.
- Publication is transactional.
- Selection uses train/validation IDs only.
- The selection record, frozen model archive, and frozen evaluation plan are
  written, hashed, and read back before test feature-state access.
- Event ordering and test-table bindings are checked when a completed study is
  verified.
- Plot-only regeneration requires an existing verified run and is
  transactional.

## P0-00 characterization baseline

The following small deterministic tests fix the behaviors requested for this
phase. No solver or metric implementation was changed to establish them.

| Contract | Characterization coverage |
|---|---|
| periodic resampling identity, shared low modes, and Nyquist transfer | `tests/test_math.py::test_spectral_resampling_is_identity_at_the_same_resolution`, `test_spectral_resampling_preserves_shared_modes_and_even_nyquist`, and `test_spectral_downsampling_discards_unrepresentable_high_modes` |
| no discarded `n_ref` mode reaches the `n_sur` input | `tests/test_validation_data.py::test_discarded_reference_modes_do_not_reach_feature_initial_state`, plus the foundation validation's `no_high_frequency_leak` check |
| real Fourier analysis/synthesis projection | `tests/test_math.py::test_real_fourier_roundtrip_for_bandlimited_fields` and `test_real_fourier_field_projection_is_idempotent` |
| Model 1 consistency for matched target/surrogate dynamics | `tests/test_learning.py::test_model1_is_consistent_for_matched_bandlimited_heat_dynamics` |
| split ID separation and full coverage | `tests/test_validation_data.py::test_reference_dataset_reuses_validation_and_has_disjoint_splits` and dataset load-time validation |
| no test access before persisted freeze-plan read-back | `tests/test_study.py::test_study_freezes_selection_before_any_test_evaluation`, completed-run semantic verification, and event-order checks |

## Missing or incomplete at P0-00

The following are not implemented or are incomplete; they are not repaired in
this phase:

- primary Model 3 reporting does not yet aggregate independent per-seed
  metrics with standard deviation and a confidence interval;
- GRF spatial frequencies are not parameterized by `domain_length`;
- the validation certificate's selected solver/reference conditions are not
  strongly checked against each dataset's target solver and `reference_nx`;
- device selection is not propagated as an end-to-end dataset/study execution
  policy;
- the fixed decoder silently zero-pads requested coefficients outside the
  point-observation bandwidth;
- there are no checked-in study specifications corresponding to the currently
  undescribed E5, E6, or E7 questions;
- FNO and DeepONet readouts are not implemented.

See `docs/known_scientific_risks.md` for code-level evidence and impact.

## Publication-label correspondence (reference only)

The current repository contains no authoritative legacy E0--E7 mapping. The
following rows for E0--E4 are inferred from `docs/migration.md` and the
question-based study catalog; they are **not yet verified** against a legacy
source tree. They must not be used for Python names or dispatch.

| Publication label | Current semantic responsibility | Status |
|---|---|---|
| E0 | independent foundation validation in `pol.validation` | inferred, current implementation present |
| E1 | heat readout calibration study | inferred, smoke/main JSON present |
| E2 | surrogate parameter/readout-time study | inferred, smoke/main JSON present |
| E3 | `J x q` observation/output map | inferred, smoke/main JSON present |
| E4 | `n_tar x n_sur` finite/surrogate resolution map | inferred, smoke/main JSON present |
| E5 | not identifiable from the current repository | not implemented; not yet verified |
| E6 | not identifiable from the current repository | not implemented; not yet verified |
| E7 | not identifiable from the current repository | not implemented; not yet verified |

Figure-number correspondence is likewise absent and is intentionally not
invented.
