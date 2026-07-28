# Known scientific risks at Phase P0-04

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

**Status: resolved for explicit target-reference claims in P0-02; heat remains
intentionally foundation-only.**

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

**Residual caveat:** checked-in heat smoke and main profiles are
`foundation_only`. Their initial conditions, domain, dtype, splits, and master
archive are provenance-bound to a passing foundation, but heat
target-reference convergence has not been established and is never claimed.
P0-02 does not add an analytic or numerical heat convergence test. Existing
main artifacts were not executed or retrospectively certified in this phase.

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
  validation schema `pol-validation-v3`. `"cuda"`, `"auto"`, and unknown
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

**Status: confirmed and characterized by an existing test.**

Relevant code:

- `pol/learning/direct.py::decode_point_observation_to_real_fourier`
- `tests/test_learning.py::test_fixed_decoder_zero_pads_unobservable_output_modes`

The decoder defines `q_observable` from `J`, analyzes only
`min(q, q_observable)` coefficients, and concatenates zeros when the requested
odd `q` is larger. The general trial schema permits this because `q` belongs to
the finite output interface and is not constrained by `J`.

**Potential impact:** the fixed decoder's output can contain structural zeros
without an explicit result flag. Comparisons at `q > q_observable` may be
interpreted as learned or observed predictions for coefficients that the fixed
decoder never estimated. This is a Model 1 limitation, not a reason to add a
general `q <= J` or `n_tar <= J` constraint.

## Items not yet verified

- The authoritative historical mapping from E0--E7 or Figure numbers to the
  current question-based studies is absent from this repository.
- No main profile was run in P0-04, so this phase establishes the CPU execution
  contract using focused tests and checked-in smoke profiles rather than new
  production-resolution results.
- This inventory does not establish cross-device numerical equivalence.
