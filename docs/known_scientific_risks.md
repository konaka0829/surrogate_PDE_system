# Known scientific risks through Phase 2-05B

Status terms in this document have strict meanings:

- **confirmed**: the behavior follows directly from the current code;
- **suspected**: there is code evidence for concern, but the end-to-end effect
  has not been demonstrated;
- **not yet verified**: the required historical result or runtime evidence is
  absent.
- **resolved**: the earlier behavior is retained for history, but the current
  implementation and focused tests enforce the stated correction.

Resolved items remain recorded with their resolution and residual caveats.

## R1. Model 3 primary test metrics used a prediction ensemble

**Status: resolved in P0-01.**

Resolution:

- `pol/study/trial.py::predict_frozen`
- `pol/study/trial.py::TrialEngine.evaluate_test`
- `pol/study/runner.py::run_study` and `verify_study_run`

The frozen prediction API now distinguishes a deterministic prediction from
per-seed predictions and requires an explicit call to form a prediction
ensemble. For `random_feature_ridge`, `TrialEngine.evaluate_test` computes
metrics independently for every frozen evaluation seed. `test_metrics.csv`
stores their arithmetic mean, Bessel-corrected sample standard deviation, and
two-sided 95% Student-t interval as the primary result.

Per-seed realizations remain in `random_feature_seed_metrics.csv`. The metric
of the seed-prediction average is now labeled `prediction_ensemble`, uses
`test_ensemble_*` metric names, and is stored only in
`random_feature_ensemble_metrics.csv`. Completed-run verification recomputes
the primary seed summaries and checks frozen seed membership and ensemble row
cardinality.

**Residual caveat:** a finite seed count estimates variability only over the
configured random-feature initialization distribution. Student-t intervals
describe uncertainty in the arithmetic mean under the usual independent-seed
assumption; they do not quantify dataset-sampling uncertainty. Ensemble rows
remain scientifically different models and must not be substituted for the
primary rows.

## R2. GRF frequencies do not use `domain_length`

**Status: resolved in P0-03; current checked-in profiles all use length one.**

Earlier behavior:

The numerical sampler constructed frequencies with
`torch.fft.rfftfreq(nx, d=1.0 / nx)`. It had no `domain_length` argument.
`generate_grf_archive` accepted and recorded `domain_length`, but did not pass
it to the sampler. Consequently, on `L != 1`, the GRF covariance spectrum used
unit-domain wavenumbers while PDE solvers and Fourier maps used the configured
physical length.

P0-03 resolution:

- `pol/numerics/initial_conditions.py::sample_gaussian_random_field_initial_conditions`
  now requires `domain_length`, rejects non-finite/non-positive values, and
  constructs `rfftfreq(nx, d=L/nx)`, so `k_m = 2*pi*m/L`.
- `pol/data/initial_conditions.py::generate_grf_archive` passes the configured
  length and records the sampler length and semantics.
- validation archive/certificate/foundation and dataset binding revisions
  cross-check that the actual sampler length, resolved contract length,
  serialized archive length, and dataset condition agree. P0-02 artifacts are
  rejected under the new semantics.
- focused tests verify the analytic amplitude ratio
  `sqrt(lambda_m(L2) / lambda_m(L1))` on odd and even grids, including the
  even-grid Nyquist wavenumber `pi*nx/L`, while retaining the constant mode.
  A fixed unit-domain regression is bitwise equal to the pre-P0-03 output.

**Residual caveat:** the correction deliberately changes the GRF distribution
for `L != 1`. No checked-in `L=1` profile changes numerically, and no main
profile was executed during P0-03.

## R3. Validation and dataset solver/reference binding is scientifically weak

**Status: resolved for explicit target-reference claims in P0-02, heat
target-specific validation in Phase 2-02, same-family Burgers refinement
certificates in Phase 2-03A, and small supporting cross-solver validation in
Phase 2-03B, and reaction-diffusion characterization/convergence in
Phase 2-04.**

Earlier behavior:

The P0-01 dataset identity was strongly bound to a validation artifact ID, but
the builder only checked that `reference_nx` did not exceed the master archive
resolution. It did not require the target solver/time/PDE condition to match
the Burgers convergence certificate. The checked-in heat dataset reused that
artifact, and smoke used dataset `reference_nx=64` after certificate selection
at 32 without recording why that refinement was allowed. Content provenance
was exact while the target-validation claim was under-specified.

P0-02 resolution:

- `pol/data/dataset.py::ensure_dataset`
- `pol/validation/binding.py::evaluate_dataset_binding`
- `pol/validation/runner.py::load_validation_certificate`
- `pol/config/models.py::DatasetSpec`

Certificates now have separate foundation and Burgers target-reference
contracts. Dataset schema `pol-dataset-v2` requires either
`validated_reference` or a reason-bearing `foundation_only` binding. The
former checks exact system kind, invariant parameters, evolution time, dtype,
and domain, and accepts only exact members of the recorded spatial and
time-candidate suffixes. It does not infer safety from larger resolution or
smaller time step. The latter fixes target-reference status to `not_claimed`
and binds only the general foundation and master archive.

Binding is evaluated before target evolution. The canonical proof/hash/status
are part of dataset identity and are cross-checked across dataset and study
payloads. Burgers smoke's 32-to-64 use is now accepted specifically because 64
is the next recorded candidate; an unlisted 48 is rejected.

Phase 2-02 resolution:

- validation configuration now uses a strict target-reference union within the
  existing validation runner: Burgers owns candidate refinement, while heat
  owns analytic and spatial-truncation checks;
- heat exact flow is checked independently against
  `exp(-nu*(2*pi*m/L)^2*t)` for constant, sine, cosine, multimode, odd/even,
  non-unit-domain, float64, float32, and even-Nyquist cases;
- heat reference candidates restrict the same finite master GRF, apply the
  exact heat flow, compare spatial truncations, and select the coarsest stable
  suffix; temporal status is `analytic_exact` and no fake `dt` exists;
- checked-in heat datasets are now `validated_reference` and bind exact
  PDE/time/domain/dtype, allowed `n_ref`, and the canonical
  `spectral_exact` condition before target evolution.

Phase 2-03A resolution:

- split-step refinement is ordered by both requested outer step and the actual
  quotient step `dt/ceil(dt/fine_dt)`; ETDRK4 uses strictly decreasing `dt`
  and null `fine_dt`;
- solver aliases are canonicalized, and solver-family/dealias mixing,
  duplicate actual conditions, reversed order, and final-time misalignment are
  rejected before a PDE solve;
- spatial, temporal, and selected-versus-finest joint comparisons are separate
  hashed long-form rows with explicit common grids and complete canonical step
  conditions;
- certificate and binding verification reconstruct selected/allowed suffixes
  from ordered candidates and rows, and reject effective-step, substep-count,
  order, selected-index, CSV, and mixed-field pseudo-candidate tampering.

Phase 2-03B resolution:

- the smoke Burgers diagnostic validates independent split-step and ETDRK4
  time-refinement sequences on the same finest grid and non-test initial
  tensor before comparing their finest solutions;
- field and low-mode relative discrepancies use the symmetric
  `2*norm(a-b)/(norm(a)+norm(b))` definition, so neither solver is a reference;
- the certificate stores family proofs, requested/effective runtime step
  metadata, hashed self-convergence rows, finest conditions, symmetric
  metrics, and a versioned evidence hash;
- the supporting block is excluded from the primary exact allowed suffix, and
  binding tamper coverage rejects injecting its ETDRK4 condition.

Phase 2-04 resolution:

- zero fields, positive/negative constant scalar recurrences, applicable
  `+-sqrt(alpha/beta)` equilibria, and `beta=0` single-mode multipliers are
  independently characterized on odd/even and non-unit grids with both
  nonlinear-filter settings;
- reaction-diffusion time refinement keeps
  `semi_implicit_spectral_euler` and `nonlinear_filter` fixed while `dt`
  strictly decreases and aligns with final time; switching filters is rejected
  as a method change;
- the generic convergence path records spatial, temporal, and actual
  selected-versus-finest joint rows, reconstructs the coarsest passing suffix,
  and checks finite output after every solve;
- canonical binding checks `nu`, `alpha`, `beta`, final time, solver, `dt`,
  nonlinear filter, resolution, domain, and dtype before target evolution;
  finite instability diagnostics and passing certificates remain
  transactional and exact-byte verified.

**Residual caveat:** main heat, Burgers, and reaction-diffusion validation profiles were not
executed or retrospectively certified during maintenance. The cross-solver
claim is limited to the small smoke configuration; the checked-in main
diagnostic is disabled.

## R4. Device configuration is not end-to-end

**Status: resolved for the first-paper workflow in P0-04 by narrowing the
validated scope to CPU-only; GPU support remains unimplemented.**

Earlier behavior:

- `pol/config/models.py::SampleSpec`
- `pol/data/initial_conditions.py::resolve_device` and
  `generate_grf_archive`
- `pol/data/dataset.py::_build_dataset` and `load_dataset`
- `pol/study/cache.py::FeatureStateCache.get_or_solve`
- `pol/validation/runner.py::_resampling_checks`

`SampleSpec.device` controls initial-condition generation during validation.
Published master and dataset tensors are detached to CPU, and dataset loading
uses `map_location="cpu"`. Dataset and study specifications have no
end-to-end device policy, so feature-state solves subsequently operate on the
CPU-loaded tensors. Some algebraic validation checks also explicitly construct
CPU tensors regardless of `SampleSpec.device`.

The resulting risk was that requesting CUDA did not mean that validation, target
dataset generation, feature generation, fitting, and evaluation all use CUDA
consistently. Device-dependent reproducibility/performance claims and metadata
could therefore be misleading. Checked-in profiles requested CPU.

P0-04 resolution:

- `SampleSpec.device` now defaults to and accepts only `"cpu"` under public
  validation schema `pol-validation-v5`. `"cuda"`, `"auto"`, and unknown
  values fail at configuration loading regardless of CUDA availability.
- `resolve_device` maps only `"cpu"` and contains no availability-dependent
  branch.
- `pol.runtime.device` defines the canonical
  `execution_device_policy="cpu_only"` and `compute_device="cpu"` contract.
  Validation, dataset binding and metadata, feature-state artifacts, frozen
  models, and study identity/summary store hashed copies. Verifiers reject a
  missing or altered policy.
- Natural high-level boundaries validate CPU tensors before and after
  validation solves, dataset target batches, feature solves, readout fitting,
  frozen-model read-back, test evaluation, and diagnostics. Publication keeps
  established `detach().cpu()` serialization only after checking that the
  upstream official result is already CPU.
- Validation, dataset, feature-state, and study artifact schemas were revised,
  and package version `0.2.4` plus numerical-environment schema `v2` prevent
  older artifacts from being reused as though they carried this guarantee.
- A fixed tiny CPU archive regression confirms unchanged values and Fourier
  coefficients for the pre-P0-04 seed and configuration.

**Residual caveat:** P0-04 does not implement end-to-end GPU execution,
CPU/GPU equivalence testing, GPU cache/serialization provenance, mixed
precision, or distributed execution. Low-level kernels may still preserve the
device of directly supplied tensors, but that behavior is outside the official
artifact workflow and must not be described as validated GPU support. A
non-null `torch_cuda_version` identifies the PyTorch build only; it does not
change the recorded CPU compute policy.

## R5. Fixed Fourier decoder silently zero-pads unobservable coefficients

**Status: resolved in P0-05 by making the retained bandwidth and structural
zero-fill explicit without changing the numerical prediction; Phase 2-05A
adds an artifact-bound matched-pipeline consistency suite.**

Relevant code:

- `pol/learning/direct.py::decode_point_observation_to_real_fourier`
- `pol/learning/direct.py::fixed_fourier_decoder_bandwidth`
- `pol/study/trial.py::TrialEngine`
- `pol/study/runner.py::verify_study_run`
- `pol/validation/runner.py::_decoder_checks`
- `pol/validation/model1_consistency.py`

Earlier behavior:

The decoder defined `q_observable` from `J`, analyzed only
`min(q, q_observable)` coefficients, and concatenated zeros when the requested
odd `q` was larger. That numerical behavior had unit coverage, but result
tables and frozen evaluation artifacts did not distinguish estimated
coefficients from structural zeros.

P0-05 resolution:

- one immutable helper now defines `observable_q`, `retained_q`, requested and
  observable maximum modes, both zero-fill counts, the application flag, and
  a versioned decoder policy;
- `observable_q = J` for odd `J` and `J-1` for even `J`; the even-grid Nyquist
  term is not counted as an observable pair in the odd-`q` real basis;
- direct validation rows, inner selection records, frozen models, frozen
  plans, and test rows carry the same `decoder_*` fields;
- frozen read-back recomputes the diagnostic from `J/q` before any test
  feature solve or metric, and completed-run verification rejects formula
  mismatches, cross-artifact mismatches, inconsistent flags/counts, and false
  diagnostics on learned readouts;
- foundation validation now records an explicit `q > observable_q`
  characterization with the zero-filled coefficient/mode ranges, correctness
  of the observable prefix, and exact zero of the suffix;
- the separate Phase 2-05A foundation check passes synthetic fields through
  the actual finite restriction, encoding, registered PDE evolution,
  observation, and fixed decoder for matched heat, Burgers, and
  reaction-diffusion cases, with independent target/surrogate calls,
  information-isolation evidence, q-projected field metrics, a separate
  representation floor, and a mismatched-time negative control;
- an exact tensor regression fixes the pre-P0-05 numerical output.

**Residual caveat:** zero-fill remains a structural Model 1 policy, not learned
extrapolation. A high-mode comparison may therefore remain unfavorable, but
the result row now states why. The general trial schema continues to permit
`q > J`: Model 2/3 learn `q` outputs from `J` features, so neither `q <= J` nor
`n_tar <= J` is a valid general interface constraint.

## R6. Burgers split-step inferred an even real-grid length from RFFT width

**Status: resolved in Phase 2-01 for the split-step kernel.**

Earlier behavior:

`pol/numerics/burgers.py::burgers_nonlinear_hat` and the implicit-mask path
reconstructed the real-grid length as `2 * (u_hat.shape[-1] - 1)`. An RFFT
width does not determine parity: width `m` can represent either
`nx=2*(m-1)` or `nx=2*m-1`. Consequently an odd `nx=15` state of width 8 was
inverse transformed with `n=14`, so its conservative nonlinear term and
subsequent trajectory were computed on the wrong collocation grid. Existing
odd/even tests covered only shape, finiteness, and determinism and therefore
did not expose the numerical error.

Phase 2-01 resolution:

- the split-step nonlinear and outer-step APIs require explicit keyword-only
  `nx`, and every inverse RFFT uses that length;
- state coefficients, wavenumbers, and any supplied dealias mask must have
  width `nx // 2 + 1`; forcing coefficients must exactly match the state
  coefficient shape;
- implicit masks are built from explicit `nx`, retaining the existing
  `mode_index <= nx // 3` policy;
- independent float64 references verify the conservative nonlinear term and
  every substep of short `nx=15,16` trajectories with dealiasing off and on;
- a test-local reproduction of the pre-correction `nx=16` algorithm verifies
  same-runtime exact tensor equality for the nonlinear term and short
  trajectory in both filter modes, while the independent mathematical
  references separately establish odd-grid correctness;
- ETDRK4 odd/even nonlinear-length and finite-trajectory characterization
  passes. Its implementation was not changed because it already supplies the
  real-field length to its inverse RFFTs.

Package version `0.2.6` enters the existing numerical-environment fingerprint,
so artifacts computed with version `0.2.5` cannot reuse the same
content-addressed numerical identity. Structural artifact schemas were not
changed.

**Residual caveat:** this correction establishes local odd/even consistency;
it does not complete all PDE validation work. Main-profile
Burgers/heat/reaction-diffusion execution remains separate work. The
profile-independent reference-field metric quadrature is covered by R8.

## R7. Validation calibration samples could overlap the future test split

**Status: resolved in Review Gate B.**

Earlier behavior:

`ValidationSpec` checked only that `calibration_sample_ids` were unique and
inside `[0, total_samples)`. Dataset construction independently applied the
configured CPU `torch.randperm` split. Under the checked-in main counts and
seed, calibration IDs `[0,1,2,3]` classified IDs 2 and 3 as test, so
finite-interface validation and target-reference convergence could consume
future test samples.

Review Gate B resolution:

- `pol.data.splits` is the single deterministic split owner and preserves the
  established CPU generator, `torch.randperm` ordering, exact IDs, and
  canonical split-hash payload;
- validation classifies explicit calibration IDs before initial-condition
  generation, any PDE solve, or artifact publication and rejects any test
  overlap with the offending IDs and complete split condition;
- the main calibration IDs are explicitly `[0,1,4,6]`, all train under its
  unchanged seed/counts; smoke IDs `[0,1]` remain train;
- validation identity/foundation/certificate store the IDs, train/validation
  membership, zero overlap, split policy/version, counts, seed, and split hash;
- certificate read-back, dataset binding, binding-proof verification, and
  dataset loading reconstruct and compare the same provenance;
- validation identity/certificate advance to `v6`, foundation contract to
  `v5`, dataset binding proof to `v4`, and package version to `0.2.7`.
  Study, metric, readout, and dataset split schemas are unchanged.

**Residual caveat:** calibration IDs remain a deliberate configuration choice.
The contract proves their split membership and excludes test; it does not
claim that a particular train/validation calibration subset is statistically
optimal.

## R8. Reference-grid quadrature could be mistaken for exact field error

**Status: resolved at the mathematical-foundation level in Phase 2-05B.**

Earlier gap:

The production formula
`periodic_l2_norm(v)^2 = (L/n) * sum_j v_j^2` and the separate `field_*` /
`data_field_*` outputs were in the correct form, but validation did not bind
their analytic normalization or demonstrate how the reference-grid
quadrature behaves when an error field is under-resolved. A finest numerical
grid could therefore be informally mistaken for analytic truth, and
reference-grid convergence was not independently certified.

Phase 2-05B resolution:

- independent unit tests prove constant and sine/cosine formulas and
  orthonormal real-Fourier Parseval identities on odd/even and non-unit grids;
- known coefficient pairs characterize absolute/relative errors and the
  existing dtype-epsilon clamp for a zero target;
- a pure synthetic check fixes one continuous target, prediction, and
  `n_tar=16` data target, then varies only `n_ref` over
  `[8,15,16,31,32]`;
- continuous error and representation-floor norms are computed analytically
  from Fourier coefficients, not inferred from the finest grid;
- the alias-prone `n_ref=8` case differs, while the complete suffix
  `[15,16,31,32]` agrees with the analytic values; the selected grid is 15;
- `fourier_prediction_metrics.field_*` agrees with direct quadrature at every
  candidate, while fixed-grid `data_field_*` is invariant;
- reference and data representation floors remain separate, with only the
  former depending on `n_ref`;
- the candidate order, row hashes, selected grid, allowed suffix, tolerances,
  statuses, and consistency hashes are bound into validation identity,
  certificate, and foundation contract and revalidated on load.

The production metric formula and study metric field names are unchanged.
Only invalid input handling was tightened for empty spatial axes,
non-floating values, and invalid domain lengths. Validation identity and
certificate are `v12`, the foundation contract is `v8`, the field-quadrature
check is `pol-field-quadrature-check-v1`, and the package version is `0.2.13`.

**Residual caveat:** the fixed synthetic suite certifies the mathematical
quadrature convention and its stated mode band. It is not a substitute for
running target-specific main profiles or for demonstrating convergence of
arbitrary unresolved production targets.

## Items not yet verified

- The authoritative historical mapping from E0--E7 or Figure numbers to the
  current question-based studies is absent from this repository.
- No main profile was run in P0-05. The fixed-decoder artifact contract is
  established with focused tests, foundation smoke validation, and checked-in
  smoke studies rather than new production-resolution results.
- No main profile was run in Phase 2-01. Its numerical correction is checked
  by focused unit/reference regressions and the checked-in smoke workflow.
- No main profile was run in Review Gate B. Main calibration membership was
  checked by deterministic split reconstruction only.
- Main-profile heat/Burgers certificates were not produced during maintenance.
  The reaction-diffusion main profile was also parse/static validated only.
  Main-scale cross-solver behavior and target-specific production
  reference-grid convergence remain unestablished.
- This work does not establish cross-device numerical equivalence.
