# Known scientific risks at Phase P0-01

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

**Status: confirmed for domains with length other than one; current checked-in
profiles all use length one.**

Relevant code:

- `pol/numerics/initial_conditions.py::sample_gaussian_random_field_initial_conditions`
- `pol/data/initial_conditions.py::generate_grf_archive`

The numerical sampler constructs frequencies with
`torch.fft.rfftfreq(nx, d=1.0 / nx)`. It has no `domain_length` argument.
`generate_grf_archive` accepts and records `domain_length`, but does not pass it
to the sampler.

**Potential impact:** when `domain_length != 1`, the GRF covariance spectrum is
defined using unit-domain wavenumbers while the PDE solvers and Fourier maps
use the configured physical length. Correlation scales and the relative
meaning of `tau` can therefore be inconsistent. No current length-one result
is affected by this specific discrepancy.

## R3. Validation and dataset solver/reference binding is scientifically weak

**Status: confirmed as an allowed configuration; the numerical impact of any
particular main artifact is not yet verified.**

Relevant code:

- `pol/data/dataset.py::ensure_dataset`
- `pol/data/dataset.py::_build_dataset`
- `pol/validation/runner.py::run_validation`
- `pol/config/models.py::DatasetSpec`

The dataset identity is strongly bound to the validation artifact ID, and the
dataset builder checks that `reference_nx` does not exceed the master archive
resolution. It does not require `DatasetSpec.reference_nx` to equal the
certificate's `selected_reference_nx`, and it does not require the dataset
target solver/time configuration to match the validated reference evolution
or selected time candidate. The checked-in heat dataset intentionally reuses a
Burgers foundation-validation archive, illustrating that the certificate
currently acts partly as an initial-condition/numerical-foundation certificate
rather than a target-specific solver certificate. The P0-00 smoke run also
confirmed a certificate-selected reference resolution of 32 while the
checked-in smoke datasets were built at `reference_nx=64`; this is permitted by
the present maximum-master-resolution check.

**Potential impact:** artifact provenance can be exact while the scientific
claim “this target field uses the convergence-validated reference conditions”
is under-specified. A dataset may use a resolution or target evolution whose
convergence was not established by the bound certificate.

## R4. Device configuration is not end-to-end

**Status: confirmed.**

Relevant code:

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

**Potential impact:** requesting CUDA does not mean that validation, target
dataset generation, feature generation, fitting, and evaluation all use CUDA
consistently. Device-dependent reproducibility/performance claims and metadata
can therefore be misleading. Current checked-in profiles request CPU.

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
- No main profile was run in P0-00, so the realized magnitude of any risk on
  production-resolution results was not measured.
- This inventory does not establish cross-device numerical equivalence.
