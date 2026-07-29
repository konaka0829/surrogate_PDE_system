# Current implementation inventory

This inventory describes the code present through Phase 2-05B after
Review Gate A/B and the study responsibility refactors. It is a
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
- a profile-independent matched Model 1 pipeline suite that passes the actual
  finite restriction, feature encoding, registered PDE evolution, point
  observation, and fixed decoder for heat, split-step/ETDRK4 Burgers, and
  reaction-diffusion, with independent target/surrogate solves, odd/even
  cases, a discarded-reference-mode isolation pair, and a time-mismatch
  negative control;
- a profile-independent periodic field-quadrature suite covering analytic
  constant, sine/cosine, and orthonormal-multimode norms on odd/even and
  non-unit grids, plus fixed-prediction reference-grid refinement,
  field-wrapper agreement, fixed-data invariance, and separated
  reference/data representation floors;
- explicit fixed-decoder characterization with requested `q` above the
  observable bandwidth, including coefficient/mode zero-fill ranges,
  observable-prefix correctness, and exact-zero suffix;
- Burgers spatial, temporal, and independently evaluated joint
  reference-convergence comparisons on configured non-test calibration
  samples, within one canonical solver family and one dealias policy;
- optional Burgers split-step/ETDRK4 supporting validation, with independent
  same-family self-convergence rows followed by a symmetric finest-condition
  discrepancy on the same finest grid and initial tensor;
- independent heat Fourier-flow checks for the zero mode, sine/cosine modes,
  linear superposition, odd/even grids including unpaired Nyquist, non-unit
  domains, and float32/float64;
- heat spatial-truncation convergence across configured reference resolutions,
  with analytic-exact temporal status and no artificial time-step sweep.
- independent reaction-diffusion zero/constant/equilibrium/linear-mode
  characterization plus spatial, temporal, and selected-versus-finest joint
  convergence under one fixed solver and nonlinear-filter condition.

The validation artifact contains the resolved specification, detailed checks,
a passing certificate, convergence rows, and a master initial-condition
archive. Exact-byte manifests and transactional publication are implemented by
the artifact layer. The certificate now separates:

- a foundation contract containing domain, the GRF sampler's actual domain and
  physical-wavenumber semantics, dtype, sample/split counts, seed, the complete
  initial-condition specification, all general-check statuses, and the master
  archive's tensor hashes and identity hash. It also contains the explicitly
  configured calibration IDs, their deterministic train/validation
  membership, zero test overlap, split policy/version, and canonical split
  hash. It also binds the matched Model 1 case summaries, detailed-case hash,
  check hash, tolerance, expected outcomes, and q-projection/representation-
  floor distinction. It also binds the field-quadrature norm convention,
  analytic/convergence hashes, selected reference grid and allowed suffix,
  tolerance, wrapper status, fixed-data invariance, and representation-floor
  status;
- a system-agnostic target-reference contract containing system kind,
  invariant PDE parameters, evolution time/domain/dtype, ordered reference
  candidates, canonical numerical-method validation, actual-step refinement
  proofs, long-form pair rows and hashes, selected/finest values and indices,
  selection policy, and machine-readable exact validated suffixes. Burgers
  uses `candidate_refinement`; heat uses the sole
  `{"solver":"spectral_exact"}` condition with `analytic_exact` temporal
  status; reaction-diffusion uses candidate refinement over exact
  `solver`/`dt`/`nonlinear_filter` tuples.
- a separate optional cross-solver supporting-evidence block containing both
  family proofs/rows/runtime step metadata and a symmetric discrepancy. It is
  not part of the generic target-reference contract or its allowed suffix.

Certificate loading reconstructs these contracts from the content-addressed
resolved specification, checks, and master archive. Missing or contradictory
fields, inconsistent selected indices, altered candidate order/effective step
metadata/row hashes/allowed suffixes, CSV disagreement, and legacy
certificate/archive revisions are rejected. The shared convergence CSV schema
is `pol-reference-convergence-csv-v3`; heat step fields are empty, split-step
stores requested and effective steps separately, and ETDRK4 stores null
requested fine steps. Reaction-diffusion rows store `dt` and
`nonlinear_filter` directly and leave Burgers-specific step fields empty.

## Execution device policy

The public first-paper artifact workflow is CPU-only. Validation schema
`pol-validation-v5` accepts only `samples.device="cpu"` and rejects CUDA,
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
membership of `reference_nx` and the canonical numerical condition in the
certificate suffixes. `foundation_only` reuses only the initial-condition
foundation,
requires a reason, and reports target-reference status `not_claimed`. A pure
binding evaluator constructs a canonical proof before target evolution.

The proof and proof hash participate in dataset identity and are stored in
`resolved_spec.json`, `metadata.json`, `dataset.pt`, and `ReferenceDataset`.
The proof also binds the GRF sampler's actual `domain_length` to the
certificate and dataset condition. The reusable `pol.data.splits` primitive
owns CPU `torch.randperm` partitioning and the established canonical split
hash. Validation calibration and dataset construction both use it; binding
checks the policy/version, counts, seed, split hash, and calibration
provenance before target evolution. Dataset loading reconstructs the exact
split IDs and cross-checks all copies. Burgers smoke explicitly proves the
selected-32 to candidate-64 relation. Heat smoke and main now bind to their
heat-specific analytic/spatial certificates and exact `spectral_exact`
numerical condition.

Checked-in dataset profiles currently provide heat and viscous Burgers targets,
each in smoke and main sizes. The generic system schema can also describe a
reaction-diffusion target, but there is no checked-in reaction-diffusion target
dataset profile. A temporary integration dataset proves that validated
reaction-diffusion references bind before target evolution.

## Evolution systems and feature generators

Registered systems are:

| Semantic kind | Current implementation |
|---|---|
| `heat` | exact discrete spectral heat flow |
| `burgers` | split-step/semi-implicit alias and Fourier pseudospectral ETDRK4/`etdrk4` alias |
| `reaction_diffusion` | semi-implicit spectral Euler with optional two-thirds nonlinear filtering |

The Burgers split-step kernel now has an explicit real-grid-length contract.
Its Fourier nonlinear and outer-step helpers require `nx`, verify that state,
wavenumber, and any supplied dealias mask all have RFFT width
`nx // 2 + 1`, and reject a forcing coefficient tensor whose shape differs
from the state coefficients. Every inverse RFFT receives `n=nx`; no helper
recovers a real-grid length from the coefficient width.

Before Phase 2-01, the split-step nonlinear helper used
`2 * (rfft_width - 1)` as the inverse-transform length. That expression is
correct for even grids but maps, for example, an `nx=15` field with width 8
onto a length-14 grid. Phase 2-01 corrects odd-grid nonlinear and trajectory
semantics. A test-local reproduction of the pre-correction even-grid algorithm
checks exact tensor equality with the corrected implementation in the same
runtime for `nx=16`, both with and without the current two-thirds filter.
Independent float64 mathematical references directly write the explicit-`nx`
conservative nonlinearity and each heat/nonlinear/Euler/filter substep for
`nx=15,16`, providing the separate odd-grid correctness check. Focused ETDRK4
parity tests pass without an ETDRK4 implementation change because that solver
already uses the real-field length in every inverse RFFT.

`pol.systems.burgers.step_metadata` is also the single step-semantics owner
for configuration validation, runtime solver metadata, certificates, and
dataset binding. For split-step it records the requested outer/fine steps,
`ceil(dt/fine_dt)`, and the actual quotient step. For ETDRK4 it requires null
`fine_dt` and uses requested `dt` as the effective step. Candidate validation
rejects reversed/duplicate-actual sequences, time misalignment, solver-family
mixing, and dealias mixing before any PDE solve.

Reaction-diffusion time candidates use only
`semi_implicit_spectral_euler`, keep `nonlinear_filter` fixed, align every
strictly decreasing unique `dt` with final time, and require the reference
evolution to equal the finest candidate. `none` and `two_thirds` are different
method conditions, not adjacent refinement levels. The generic nonlinear-PDE
convergence engine performs the Burgers and reaction-diffusion spatial,
temporal, stable-suffix, and independent joint comparisons. Every solve is
checked for finite output; a non-finite state publishes a transactional,
exact-byte-verified failure diagnostic.

Feature-generator kinds are `pde_dynamics` and `static_input`. The static
baseline uses the same finite-input, cache, observation, readout, and
evaluation path, although no standalone checked-in study JSON currently
selects it.

The implemented observation is L2-scaled equispaced point observation of the
source-grid trigonometric interpolant.

## Readouts

`pol.study.readouts` owns the fitting and frozen-prediction lifecycle for:

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

`pol.study.trial.TrialEngine` coordinates finite views, cached feature states,
train/validation tensors, and test feature preparation for one validated
trial. Readout-kind dispatch is confined to `pol.study.readouts`; the trial
coordinator does not implement ridge or random-feature algorithms.

## Study orchestration and protocol

All scalar and swept calculations use the unified
`pol.study.runner.run_study` execution path; a scalar calculation is a
one-cell study. The runner is the high-level façade that prepares the dataset,
invokes case expansion and trial/search work, enforces convergence, crosses
the persisted freeze boundary, evaluates tests, and transactionally publishes
the completed run.

Supporting responsibilities have one semantic owner:

- `pol.study.cases` performs pure case/axis expansion, invalid-trial planning,
  and candidate upper-bound planning;
- `pol.study.protocol` constructs selection and frozen payloads and performs
  exact/hash read-back before test evaluation is permitted;
- `pol.study.results` owns CSV field order, table/summary serialization,
  manifest construction, and loading completed reporter inputs;
- `pol.study.verification` verifies completed runs read-only, including exact
  manifest bytes, identity and frozen bindings, event order, independent-seed
  summaries/ensemble membership, and direct-decoder diagnostics.

None of these support modules imports `pol.study.runner`; there is no duplicate
scalar, sweep, verification, or plots-only execution path.

## Studies and scientific questions

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

`pol.study.evaluation` owns the coefficient/data-field/reference-field metric
wrappers, representation-floor wrapper, metric prefixing, independent-seed
statistics, immutable candidate/test/prediction results, and pure
validation/test row construction. It performs no readout fitting, feature
solve, test-ID request, or artifact I/O.

The prediction metric set currently stores:

- coefficient MSE;
- samplewise coefficient relative L2, aggregated by mean, median, and maximum;
- reference-field absolute and relative periodic L2, each aggregated by mean,
  median, and maximum;
- finite-data-field absolute and relative periodic L2, each aggregated by
  mean, median, and maximum;
- reference-field and finite-data representation-floor relative L2, each
  aggregated by mean, median, and maximum.

The periodic field norm remains
`sqrt((L/n) * sum_j value_j^2)`. Phase 2-05B independently validates this
formula against continuous orthogonality/Parseval results and adds explicit
rejection of a missing/empty spatial axis, non-floating values, and non-finite
or non-positive domain length. Samplewise relative error retains its existing
target-norm machine-epsilon clamp.

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
  Phase 2-05B validation identity/certificate use `v12`, the foundation
  contract uses `v8`, the field quadrature check is
  `pol-field-quadrature-check-v1`, and the matched pipeline check remains
  `pol-matched-model1-pipeline-check-v1`. The generic target-reference
  contract remains `v4`, and the nested cross-solver spec/check remain
  `v1`/`v2`. Dataset binding proofs use `v7`. Dataset identity, metadata,
  archive, and resolved-spec families use `v5`; initial-condition archives
  remain at `v4`, foundation/master bindings at `v3`, and feature-state
  identity/archive/metadata at `v2`.
- Numerical-environment schema `pol-numerical-environment-v2` and package
  version `0.2.13` are content-hash inputs. Validation artifacts without the
  field-quadrature and matched-pipeline suites, actual-step refinement proof,
  and reconstructable convergence rows cannot share an identity with current
  results.
  Package version `0.2.6` previously separated post-Phase-2-01 odd-grid
  results from version `0.2.5`. Study, metric, readout, initial-condition, and
  dataset artifact schemas are unchanged. The earlier sampler, domain, CPU,
  target-reference binding, decoder-diagnostic, split IDs, and frozen-test
  proof semantics remain content-hash inputs.
- Plot-only regeneration requires an existing verified run and is
  transactional.

## Characterization and correction coverage through Phase 2-05B

The P0 rows below characterize or extend the previously established contracts.
The final row records the focused Phase 2-01 scientific correction and its
numerical-preservation evidence. Phase 2-05B preserves metric formulas and
adds only independently justified input validation.

| Contract | Characterization coverage |
|---|---|
| periodic resampling identity, shared low modes, and Nyquist transfer | `tests/test_math.py::test_spectral_resampling_is_identity_at_the_same_resolution`, `test_spectral_resampling_preserves_shared_modes_and_even_nyquist`, and `test_spectral_downsampling_discards_unrepresentable_high_modes` |
| no discarded `n_ref` mode reaches the `n_sur` input | `tests/test_validation_data.py::test_discarded_reference_modes_do_not_reach_feature_initial_state`, plus the foundation validation's `no_high_frequency_leak` check |
| real Fourier analysis/synthesis projection | `tests/test_math.py::test_real_fourier_roundtrip_for_bandlimited_fields` and `test_real_fourier_field_projection_is_idempotent` |
| Model 1 consistency for matched target/surrogate dynamics | `pol.validation.model1_consistency` and focused tests in `tests/test_learning.py` cover odd/even equal-grid heat, unequal `n_tar/n_sur` heat, discarded high-mode isolation, odd/even split-step Burgers through the explicit-`nx` kernel, ETDRK4, reaction-diffusion, separate solve calls, q-projection versus representation floor, observable-band case selection, general `q > J`, and a time-mismatch negative control. `tests/test_validation_data.py` covers foundation/certificate binding, detailed and summarized case/tolerance/status tamper rejection, and all three checked-in smoke validations. |
| Phase 2-05B periodic L2 and field quadrature | `pol.validation.quadrature`, `tests/test_math.py`, `tests/test_learning.py`, and `tests/test_validation_data.py` cover constant and resolved sine/cosine norms, orthonormal Parseval on odd/even and non-unit grids with float32/float64 batch axes, known absolute/relative errors, the zero-target epsilon clamp, coefficient/field Parseval agreement, an intentionally aliased `n_ref=8`, the selected stable suffix `[15,16,31,32]`, direct-wrapper agreement, fixed `data_field_*` invariance, separated reference/data representation floors, certificate binding, and grid/order/selection/suffix/tolerance/status tamper rejection. |
| split ID separation and full coverage | `tests/test_validation_data.py::test_reference_dataset_reuses_validation_and_has_disjoint_splits` and dataset load-time validation |
| no test access before persisted freeze-plan read-back | `tests/test_study.py::test_study_freezes_selection_before_any_test_evaluation`, completed-run semantic verification, and event-order checks |
| independent-seed primary statistics and separate prediction ensemble | `tests/test_random_feature_evaluation.py` and the random-feature artifact/verification tests in `tests/test_study.py` |
| certificate/dataset target-reference binding | `tests/test_dataset_binding.py`, including exact candidate suffix membership, heat and Burgers cross-binding rejection, all target-condition mismatch classes, pre-evolution rejection, heat `validated_reference` status, and proof/certificate tamper checks |
| Phase 2-02 heat analytic and spatial validation | `tests/test_learning.py`, `tests/test_validation_data.py`, and `tests/test_config.py`, covering independent constant/sine/cosine/multimode formulas, odd/even and Nyquist behavior, non-unit domains, float32/float64 tolerances, malformed inputs, semantic-union strictness, analytic-exact temporal status, spatial suffix selection, and absence of fake time candidates |
| Phase 2-03A Burgers same-family convergence certificate | `tests/test_config.py`, `tests/test_validation_data.py`, `tests/test_dataset_binding.py`, and `tests/test_numerics.py`, covering alias canonicalization, requested/effective split-step refinement, ETDRK4 null fine steps, reversed/duplicate/family/dealias/alignment rejection, long-form spatial/temporal/joint rows, odd/even reference grids, finest-pair failure, exact suffix reconstruction, CSV and step/candidate/selection tampering, mixed-field pseudo-candidate rejection, and pre-evolution binding failure |
| Phase 2-03B Burgers cross-solver supporting evidence | `tests/test_config.py`, `tests/test_validation_data.py`, `tests/test_dataset_binding.py`, and `tests/test_numerics.py`, covering independent family specs/proofs, disabled no-solve behavior, self-before-cross ordering, requested/effective rows, symmetric field/low-mode metrics on odd/even grids, deterministic reruns, evidence/hash/condition/status/runtime-metadata tampering, and exclusion from dataset allowed suffixes |
| Phase 2-04 reaction-diffusion characterization and convergence | `tests/test_numerics.py`, `tests/test_config.py`, `tests/test_validation_data.py`, and `tests/test_dataset_binding.py`, covering independent scalar recurrence, zero and applicable nonzero equilibria, `beta=0` Fourier multiplier, odd/even and non-unit domains, both nonlinear filters, fixed-filter decreasing-`dt` refinement, spatial/temporal/joint rows, stable suffix reconstruction, finite-failure artifacts, exact condition binding, cross-system misuse, and tamper rejection |
| physical-domain GRF covariance and `L=1` regression | `tests/test_numerics.py::test_grf_unit_domain_output_matches_pre_p0_03_regression`, `test_grf_mode_amplitudes_follow_physical_domain_scaling`, and `test_grf_even_grid_nyquist_uses_physical_wavenumber` |
| GRF domain provenance and tamper rejection | `tests/test_validation_data.py::test_nonunit_domain_is_bound_across_grf_archive_and_certificate` and the sampler-domain proof/certificate tamper tests in `tests/test_dataset_binding.py` |
| CPU-only configuration, boundary placement, provenance, and tamper rejection | `tests/test_device_policy.py`, covering CPU/default acceptance; CUDA/auto rejection under both mocked availability states; no CUDA availability query in resolution; CPU master/dataset/feature/frozen tensors; environment, certificate, proof, identity, and summary policy copies; binding-proof and study-summary policy tamper; and a non-CPU boundary error |
| P0-04 CPU numerical preservation | `tests/test_device_policy.py::test_p0_04_cpu_deterministic_archive_regression`, which fixes exact tensor hashes for the pre-P0-04 tiny seed/configuration; the pre-existing unit-domain regression and feature-state batching-invariance test remain in force |
| P0-05 fixed-decoder bandwidth formulas and numerical preservation | parameterized odd/even and below/equal/above-bandwidth tests in `tests/test_learning.py`, including invalid-input rejection, observable-prefix recovery, exact suffix zeros, and exact equality with the pre-diagnostic tensor construction |
| P0-05 decoder artifact binding and tamper rejection | `tests/test_study.py::test_direct_decoder_diagnostic_is_bound_across_study_artifacts`, diagnostic-tamper cases, and the frozen-mismatch pre-test guard |
| P0-05 `J x q` independence and foundation characterization | `tests/test_study.py::test_checked_in_observation_output_plan_keeps_q_greater_than_J_cells` and `tests/test_validation_data.py::test_foundation_validation_publishes_passing_certificate` |
| Phase 2-01 split-step real-grid length and parity | `tests/test_numerics.py`, covering RFFT-width ambiguity, required explicit `nx`, independent nonlinear and short-trajectory mathematical references for `nx=15,16` with filtering off/on, same-runtime exact equality against a test-local pre-correction even-grid algorithm, spectral/mask/forcing mismatch rejection, and ETDRK4 parity characterization |
| Review Gate B calibration/test isolation | `tests/test_validation_data.py`, `tests/test_config.py`, and `tests/test_dataset_binding.py`, covering exact preservation of pre-gate split IDs/hash, disjoint full coverage, checked-in smoke/main membership, pre-compute overlap rejection, certificate provenance, certificate/proof tamper rejection, and dataset binding/load reconstruction |

## Missing or incomplete after Phase 2-05B

The following are not implemented or are incomplete; they are not repaired in
this task:

- Burgers main-profile convergence has been statically validated but not run
  or retrospectively certified during maintenance;
- the heat main profile was statically validated but not run during
  maintenance;
- the cross-solver result is established only for the checked-in smoke
  diagnostic; the main profile declares it disabled and has not been
  calibrated or run;
- the reaction-diffusion main profile is declaration-only and was not run;
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
