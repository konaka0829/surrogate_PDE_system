# Current implementation inventory

This inventory describes the code present at Phase P0-05. It is a
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
- explicit fixed-decoder characterization with requested `q` above the
  observable bandwidth, including coefficient/mode zero-fill ranges,
  observable-prefix correctness, and exact-zero suffix;
- Burgers spatial, temporal, and joint reference-convergence comparisons on
  configured calibration samples.

The validation artifact contains the resolved specification, detailed checks,
a passing certificate, convergence rows, and a master initial-condition
archive. Exact-byte manifests and transactional publication are implemented by
the artifact layer. The certificate now separates:

- a foundation contract containing domain, the GRF sampler's actual domain and
  physical-wavenumber semantics, dtype, sample/split counts, seed, the complete
  initial-condition specification, all general-check statuses, and the master
  archive's tensor hashes and identity hash;
- a Burgers target-reference contract containing invariant PDE parameters,
  evolution time, ordered spatial and time candidate lists, selected values
  and indices, selection policy, and machine-readable exact validated suffixes.

Certificate loading reconstructs these contracts from the content-addressed
resolved specification, checks, and master archive. Missing or contradictory
fields, inconsistent selected indices, altered allowed suffixes, and legacy
certificate/archive revisions are rejected.

## Execution device policy

The public first-paper artifact workflow is CPU-only. Validation schema
`pol-validation-v3` accepts only `samples.device="cpu"` and rejects CUDA,
automatic selection, and unknown devices before execution, independently of
`torch.cuda.is_available()`.

`pol.runtime.device` is the single source of truth for
`execution_device_policy="cpu_only"` and `compute_device="cpu"`. These values
are recorded in the numerical environment, validation identity/certificate
and foundation/master contracts, dataset binding proof/identity/metadata,
feature-state identity/metadata/archive, frozen model and evaluation records,
and study identity/summary. They participate in content hashes and are checked
on read-back. `torch_cuda_version`, when present, describes only the installed
PyTorch build.

Official boundaries reject a non-CPU tensor rather than silently moving it:
validation checks and reference solves, master publication, dataset target
batches and loaded tensors, feature-state solves/cache, readout fitting,
frozen-model publication/read-back, test evaluation, convergence and
diagnostics. Existing `detach().cpu()` calls remain the serialization contract
after the producing boundary has already proved CPU placement.

The low-level Fourier, resampling, solver, observation, and learning algebra
continues to preserve input-device generality where implemented. It is not an
end-to-end GPU workflow and carries no GPU artifact or reproducibility
guarantee.

## Data

The implemented data path includes:

- deterministic periodic Gaussian-random-field initial conditions whose
  physical angular wavenumber is `2 pi m / L`;
- a content-addressed master initial-condition archive;
- deterministic, disjoint train/validation/test ID splits;
- reference datasets evolved by a registered target system;
- finite `n_tar` views derived by periodic spectral resampling;
- `q`-dimensional real Fourier training targets;
- separate finite targets and `n_ref` reference targets;
- construction of the `n_sur` feature input only from the finite `n_tar`
  tensor.

Every dataset has an explicit validation binding. `validated_reference`
requires exact target-system/PDE/time/domain/dtype equality and exact
membership of `reference_nx` and the solver/time dictionary in the certificate
suffixes. `foundation_only` reuses only the initial-condition foundation,
requires a reason, and reports target-reference status `not_claimed`. A pure
binding evaluator constructs a canonical proof before target evolution.

The proof and proof hash participate in dataset identity and are stored in
`resolved_spec.json`, `metadata.json`, `dataset.pt`, and `ReferenceDataset`.
The proof also binds the GRF sampler's actual `domain_length` to the
certificate and dataset condition. Dataset loading cross-checks all copies.
Burgers smoke explicitly proves the selected-32 to candidate-64 relation. Heat
smoke and main remain buildable as foundation-only datasets and make no heat
convergence claim.

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
  in some study metadata). Its directly observable real-Fourier prefix has
  `observable_q=J` for odd `J` and `J-1` for even `J`; requested coefficients
  after `min(q, observable_q)` are structurally zero-filled and explicitly
  diagnosed;
- `affine_ridge`: centered affine ridge regression with validation-selected
  regularization and an SVD minimum-norm path at zero regularization (Model 2);
- `random_feature_ridge`: skip-connected random nonlinear features followed
  by centered affine ridge regression, with separate selection and evaluation
  seed sets (Model 3). Evaluation requires at least two distinct seeds.

Display names do not control dispatch.

The fixed-decoder bandwidth is not a generic interface constraint. In
particular, `q > J` remains valid: affine and random-feature readouts learn
`q` outputs from a `J`-dimensional feature. No `q <= J` or `n_tar <= J`
constraint is imposed.

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

For a selected random-feature readout, the primary canonical test metrics are
means of per-seed metrics. The same row records Bessel-corrected sample
standard deviations and two-sided 95% Student-t confidence intervals.
Per-seed rows and prediction-ensemble rows are separate tables, and ensemble
metrics use `test_ensemble_*` names. Deterministic readouts remain single-model
results and do not receive placeholder seed uncertainty.

## Implemented artifact and evaluation controls

- Strict Pydantic models reject unknown scientific configuration keys.
- Scientific identities exclude storage roots and include environment and
  upstream artifact identities.
- Dataset identities include the complete validation binding proof and hash;
  changing a proof changes the artifact ID.
- Validation products, datasets, feature states, and study runs are
  content-addressed and exact-byte verified.
- Publication is transactional.
- Selection uses train/validation IDs only.
- The selection record, frozen model archive, and frozen evaluation plan are
  written, hashed, and read back before test feature-state access.
- Event ordering and test-table bindings are checked when a completed study is
  verified.
- Completed-run verification checks frozen random-feature seed membership,
  recomputes primary mean/standard-deviation/confidence-interval fields from
  per-seed rows, and checks the separate ensemble row and member count.
- Direct-decoder diagnostics are recomputed from `J/q` and bound across
  validation rows, direct inner-selection records, frozen models, frozen
  plans, test rows, and run-summary counts. Learned readouts must have empty
  union-CSV diagnostic cells and no frozen/selection diagnostic payload.
- P0-05 study-run identity, manifest, summary, selection, frozen-plan, and
  frozen-model schemas use `v5`; `dataset_reference.json` uses `v3`.
  Validation identity/certificate use `v5`, the foundation contract uses
  `v4`, initial-condition/dataset artifacts remain at `v4`,
  foundation/master bindings and dataset binding proofs use `v3`, and
  feature-state identity/archive/metadata use `v2`.
- Numerical-environment schema `pol-numerical-environment-v2` and package
  version `0.2.5` join these artifact revisions in preventing pre-P0-05 runs
  and validation certificates from being interpreted as decoder-diagnostic
  complete. The earlier sampler, domain, CPU, target-reference binding, and
  frozen-test proof semantics remain content-hash inputs.
- Plot-only regeneration requires an existing verified run and is
  transactional.

## P0-00 characterization baseline and P0-01--P0-05 extensions

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
| independent-seed primary statistics and separate prediction ensemble | `tests/test_random_feature_evaluation.py` and the random-feature artifact/verification tests in `tests/test_study.py` |
| certificate/dataset target-reference binding | `tests/test_dataset_binding.py`, including exact candidate suffix membership, all target-condition mismatch classes, pre-evolution rejection, heat foundation-only status, and dataset/study tamper checks |
| physical-domain GRF covariance and `L=1` regression | `tests/test_numerics.py::test_grf_unit_domain_output_matches_pre_p0_03_regression`, `test_grf_mode_amplitudes_follow_physical_domain_scaling`, and `test_grf_even_grid_nyquist_uses_physical_wavenumber` |
| GRF domain provenance and tamper rejection | `tests/test_validation_data.py::test_nonunit_domain_is_bound_across_grf_archive_and_certificate` and the sampler-domain proof/certificate tamper tests in `tests/test_dataset_binding.py` |
| CPU-only configuration, boundary placement, provenance, and tamper rejection | `tests/test_device_policy.py`, covering CPU/default acceptance; CUDA/auto rejection under both mocked availability states; no CUDA availability query in resolution; CPU master/dataset/feature/frozen tensors; environment, certificate, proof, identity, and summary policy copies; binding-proof and study-summary policy tamper; and a non-CPU boundary error |
| P0-04 CPU numerical preservation | `tests/test_device_policy.py::test_p0_04_cpu_deterministic_archive_regression`, which fixes exact tensor hashes for the pre-P0-04 tiny seed/configuration; the pre-existing unit-domain regression and feature-state batching-invariance test remain in force |
| P0-05 fixed-decoder bandwidth formulas and numerical preservation | parameterized odd/even and below/equal/above-bandwidth tests in `tests/test_learning.py`, including invalid-input rejection, observable-prefix recovery, exact suffix zeros, and exact equality with the pre-diagnostic tensor construction |
| P0-05 decoder artifact binding and tamper rejection | `tests/test_study.py::test_direct_decoder_diagnostic_is_bound_across_study_artifacts`, diagnostic-tamper cases, and the frozen-mismatch pre-test guard |
| P0-05 `J x q` independence and foundation characterization | `tests/test_study.py::test_checked_in_observation_output_plan_keeps_q_greater_than_J_cells` and `tests/test_validation_data.py::test_foundation_validation_publishes_passing_certificate` |

## Missing or incomplete after P0-05

The following are not implemented or are incomplete; they are not repaired in
this phase:

- heat profiles have no target-specific convergence certificate and therefore
  remain explicitly `foundation_only`;
- end-to-end CUDA/GPU execution, GPU artifact provenance, CPU/GPU numerical
  equivalence, mixed precision, and distributed execution are not implemented;
- the fixed decoder still uses structural zero-fill outside the directly
  observable point-observation bandwidth; this is now explicit and
  artifact-bound, but it is not learned extrapolation;
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
