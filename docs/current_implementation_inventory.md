# Current implementation inventory

This inventory separates the active implementation from historical
transitions and work that has not yet been executed or verified. It is a
characterization of package `0.2.29`, not a claim that checked-in main plans
have produced scientific results.

## Current active implementation

The active write-side contracts are:

| Artifact responsibility | Active contract |
|---|---|
| package/environment | package `0.2.29`; `pol-numerical-environment-v2` |
| validation | config through `pol-validation-v6`; identity/certificate `v12`; foundation contract `v8` |
| dataset | `pol-dataset-v3`; binding proof `v7`; identity/metadata/archive/resolved families `v5` |
| study plan and identity | `pol-study-plan-v4`; run identity `v15`; workload plan/case `v1` |
| study freeze and results | selection `v9`; frozen archive/plan `v10`; result row `v3`; event `v1` |
| completed study | run manifest/summary `v16`; dataset reference `v3`; prediction capture `v1` |
| cross-run report | `pol-report-v1` and corresponding report artifact `v1` families |
| digital FNO adapter | config `v3`; run identity/manifest/summary `v4`; selection/checkpoint/plan `v3`; source reference/fairness row `v3` |

Older schemas accepted by read-only selection-source verification are
migration inputs, not active-write versions.

Maintenance validation for this release completed with Python 3 compilation,
646 tests, `scripts/check.sh`, the full smoke workflow, the read-only main plan
audit, and wheel construction passing. The smoke workflow exercised all three
validations, both datasets, nine studies, the optional FNO adapter, and the
four-source report. This is execution-path evidence only: no main profile was
run and no main scientific result is claimed.

### Checked-in configuration catalog

Every family below has a smoke profile for maintenance and a main profile that
is declaration-only until an operator explicitly passes the production gates.

| Kind | Main | Smoke |
|---|---|---|
| validation | `configs/validation/foundation_main.json` | `configs/validation/foundation_smoke.json` |
| validation | `configs/validation/heat_main.json` | `configs/validation/heat_smoke.json` |
| validation | `configs/validation/reaction_diffusion_main.json` | `configs/validation/reaction_diffusion_smoke.json` |
| dataset | `configs/datasets/heat_main.json` | `configs/datasets/heat_smoke.json` |
| dataset | `configs/datasets/burgers_main.json` | `configs/datasets/burgers_smoke.json` |
| study | `studies/heat_readout_calibration.json` | `studies/heat_readout_calibration_smoke.json` |
| study | `studies/surrogate_parameter_time_coordinate_search.json` | `studies/surrogate_parameter_time_coordinate_search_smoke.json` |
| study | `studies/surrogate_parameter_time_landscape.json` | `studies/surrogate_parameter_time_landscape_smoke.json` |
| study | `studies/dynamic_feature_baseline_comparison.json` | `studies/dynamic_feature_baseline_comparison_smoke.json` |
| study | `studies/readout_stability_noise.json` | `studies/readout_stability_noise_smoke.json` |
| study | `studies/learning_curve.json` | `studies/learning_curve_smoke.json` |
| study | `studies/random_feature_seed_statistics.json` | `studies/random_feature_seed_statistics_smoke.json` |
| study | `studies/observation_output_budget.json` | `studies/observation_output_budget_smoke.json` |
| study | `studies/input_simulation_resolution.json` | `studies/input_simulation_resolution_smoke.json` |
| digital baseline | `digital_baselines/fno1d.json` | `digital_baselines/fno1d_smoke.json` |
| cross-run report | `reports/surrogate_operator_summary.json` | `reports/surrogate_operator_summary_smoke.json` |

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
`pol-validation-v6` accepts only `samples.device="cpu"` and rejects CUDA,
automatic selection, and unknown devices before execution, independently of
`torch.cuda.is_available()`; existing v5 heat/Burgers configurations remain
accepted migration inputs.

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
evaluation path. The `dynamic_feature_baseline` comparison contract prevents
variant-level changes to the information budget or readout candidates and
requires verified Phase-3 selection sources for all dynamic families. Static
resolution rows are explicitly labeled as encoding-consistency checks and do
not claim PDE convergence.

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
  seed sets (Model 3). Selection caches each seeded map and its lifted
  train/validation tensors across ridge values. Candidate results carry a
  recipe; evaluation members are fitted only for the study-selected
  case/readout, from train/validation data, before the persisted freeze
  boundary. Evaluation requires at least two distinct seeds.

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
- `pol.study.prediction_capture` owns predeclared test/member capture,
  capacity-conscious all-test Fourier aggregates, tensor content identities,
  and internal capture verification;
- `pol.study.verification` verifies completed runs read-only, including exact
  manifest bytes, identity and frozen bindings, event order, independent-seed
  summaries/ensemble membership, and direct-decoder diagnostics.
- `pol.reporting` owns cross-run source verification, phase-diagram extraction,
  baseline-table normalization, deterministic rendering, report identity, and
  transactional exact-byte publication. It imports no study runner and has no
  validation, dataset-build, feature-solve, fit, or test-inference path.

None of these support modules imports `pol.study.runner`; there is no duplicate
scalar, sweep, verification, or plots-only execution path.

## Studies and scientific questions

| Study JSON family | Scientific question currently represented |
|---|---|
| `heat_readout_calibration` | For an analytically transparent heat target, how closely do direct and affine readouts match the ideal heat multiplier over `q` under under-diffusive, matched, and more-diffusive feature conditions? Random features remain a separately evaluated nonlinear model with an explicit non-applicable multiplier status. |
| `surrogate_parameter_time_coordinate_search` | Which surrogate parameter/readout-time condition is found by alternating two-axis validation search? This remains an efficient search aid, not a complete phase map. |
| `surrogate_parameter_time_landscape` | How does validation error vary over the complete predeclared `nu_tilde x T_tilde` Cartesian product for heat, Burgers, and reaction-diffusion features, and which validation-selected affine condition is the representative downstream feature condition? |
| `dynamic_feature_baseline_comparison` | At one shared information budget and fixed validation-selected feature conditions, how do static input, heat, Burgers, and reaction-diffusion features compare for the same three readout contracts? |
| `readout_stability_noise` | For read-back frozen readouts, how do clean/noisy errors, learned-map norms, centered covariance conditioning, independent random-feature seeds, and a separately labeled prediction ensemble relate under common random noise draws? |
| `learning_curve` | How do validation/test generalization and readout stability change over strict nested prefixes of the unchanged canonical train split, with every size frozen before test access and feature-state solves reused across sizes? |
| `random_feature_seed_statistics` | For the fixed Phase-3-selected Burgers feature condition, what is the distribution of independent random-feature model performance over disjoint evaluation seeds, with Student-t uncertainty for the mean, separate descriptive quartiles, auditable map/member hashes, and a separately labeled prediction ensemble? |
| `observation_output_budget` | How does performance vary over the distinct `J x q` observation/output budget at fixed `n_tar` and `n_sur`, for the Phase 3-selected Burgers and reaction-diffusion systems and each direct/affine/random readout? |
| `input_simulation_resolution` | How does performance vary over the distinct `n_tar x n_sur` finite-data/surrogate-resolution budget at fixed `J` and `q`, for the Phase 3-selected Burgers and reaction-diffusion systems and each direct/affine/random readout? |

The current documentation groups these semantic responsibilities as follows:

| Phase | Semantic responsibility |
|---|---|
| Phase 3 | validation-only surrogate-parameter/readout-time coordinate search, complete Cartesian landscape, representative-condition selection, and downstream selection-source binding |
| Phase 4 | separate `J x q` observation/output and `n_tar x n_sur` finite/surrogate-resolution studies |
| Phase 5 | static-versus-dynamic feature comparison at one shared information budget and validation-selected dynamic conditions |
| Phase 6 | readout stability under feature noise, nested-prefix learning curves, and independent random-feature seed statistics |
| Phase 7 | predeclared prediction capture, verified-run single-study reporters, and verified-source cross-run phase diagrams/baseline tables |

These are current semantic groupings, not a correspondence to publication
labels E3--E7 or to figure numbers.

Each family has a smoke and a main profile. Only smoke/tiny profiles belong in
automated checks or maintenance work.

Supported search modes are static, Cartesian grid, and two-axis coordinate
search. Grid selection artifacts preserve the declared Cartesian count,
config-order candidate sequence, evaluated/skipped cells and reasons, and the
validation-selected representative feature condition before test access.
Implemented diagnostics are analytic heat-multiplier inspection,
surrogate-resolution convergence, and formal readout stability under
observation noise. The stability diagnostic consumes the persisted/read-back
archive, binds every row to its model key and freeze hashes, preserves
per-seed/per-repeat rows, and separates independent-seed primary summaries
from prediction ensembles. Rank-deficient covariance reports raw and
retained-rank conditions with an explicit cutoff. Noise is a separate study
and is not configured by the parameter/time studies. Generic reporters provide
metric curves, validation-only metric maps, error-bar noise curves,
error-versus-norm and condition-versus-error plots, random-feature seed
scatter/box/empirical-CDF views with mean and Student-t CI, and read-only
ideal/effective heat-multiplier comparisons. Representative prediction-field
and Fourier-error-spectrum reporters consume only
`pol-prediction-capture-v1` from a verified completed run. New numerical runs
publish and verify before the optional report transaction; a report failure
does not invalidate the numerical publication.

The separate `pol-report-v1` layer reads two or more exact verified completed
runs. Its `phase_diagram_report` consumes explicitly named validation or test
tables and records every declared cell as valid, missing, or invalid before
rendering physical/log axes. Its `baseline_summary_table` emits unrounded CSV
plus optional Markdown/LaTeX, keeps field/data spaces and representation
floors separate, and normalizes random-feature primary seed statistics
without consulting the ensemble table. `pol report verify` checks the
content-addressed identity and every artifact byte. The smoke report is in the
ordered smoke workflow; the main report remains an unexecuted declaration
over unexecuted main sources.

The learning-curve contract is owned by the trial/study selection layer rather
than the dataset. `training_subset.n_train` is the sole global axis, arbitrary
ID injection is rejected by the strict schema, and runtime preflight rejects
sizes above the canonical train count before feature fitting. Selection and
result artifacts preserve the ordered subset IDs, subset and parent hashes,
policy/version, and fixed validation binding. The completed-table reporter
uses a log training-size axis and displays the random-feature Student-t
interval without feeding test values back into selection.

The smoke landscape has been executed in maintenance and verified across all
three feature systems. The main 75-cell landscape has only been strict-loaded
and planned; no main numerical result is claimed.

## Digital baseline adapter

`pol.digital_baselines` implements one CPU-only 1D FNO without adding a
physical readout kind or a second physical StudyRunner. `protocol.py` owns the
strict `pol-digital-baseline-v3` schema and pure budget plan, `datasets.py`
owns train/validation/test finite-view boundaries and train-only
standardization, `fno1d.py` owns spectral/local layers and real-scalar
parameter counting, `evaluation.py` owns deterministic training and
validation checkpoint selection, and `runner.py` owns physical-source
preflight, freeze/read-back, test statistics, fairness tables, transactions,
and completed-run verification.

The checked-in smoke/main configurations use the exact Burgers
dynamic-feature baseline as a predeclared physical source and list all 12
physical variant/readout rows before evaluation. The periodic Burgers
baseline supplies only the finite `n_tar` field and uses no absolute
coordinate channel, preserving circular-shift equivariance. The optional
periodic sin/cos policy is explicit and the removed ramp is rejected.
Predictions are `q` real Fourier coefficients and use the existing
`fourier_prediction_metrics`/representation-floor implementation.

Selection seeds choose the candidate architecture and epoch using validation
only. A disjoint evaluation-seed set produces independently trained frozen
models. The primary row reports per-seed mean, Bessel-corrected standard
deviation, and Student-t 95% interval; a prediction-average ensemble is a
separate table. The selection record, checkpoint archive, and evaluation plan
are persisted, byte/tensor hashed, and read back before `build_test_view`.

The fairness table binds the exact dataset/split, input/output dimensions,
per-independent-realization trainable, fixed-random, and total stored
real-scalar counts, pre/post-lift feature dimensions, all-realization storage,
validation metrics, reference/data metrics and floors, seed statistics,
source/condition hashes, and the physical-versus-digital inference path.
Counts are reconstructed from frozen tensors and exclude the physical
dynamics and hardware resources. Digital training time is recorded, but
energy is unmeasured and
wall-clock/energy comparison is disabled because the physical source lacks a
common measurement protocol.

The FNO smoke path and tiny/focused tests are executed. The main declaration
plans 20 trained models and at most 64,000 optimizer steps; it has not been
executed and makes no performance claim. DeepONet, GPU training, mixed
precision, distributed training, and energy instrumentation are not
implemented.

Production readiness is operationally fixed by
`docs/production_runbook.md`, `scripts/plan_main.py`, and the stage-gated
`scripts/run_main.sh`. Audit schema `pol-production-plan-audit-v3`
strict-parses three validations, two datasets, nine studies, one digital
baseline, and one cross-run report. It reports exact map/lift/ridge/SVD and
selected-only evaluation-member fit budgets, shape/convergence/diagnostic
budgets, and missing/completed dependency state. Unresolved selection-bound
studies report null workload counts. It performs no main execution. Every
mutating main stage requires `POL_CONFIRM_MAIN=YES`, no all-in-one stage
exists, and the workload block requires human capacity approval before the
first main stage. At this maintenance checkpoint, all downstream main
dependencies remain intentionally missing because no main profile has been
run.

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
prediction relative errors (mean and maximum). Heat-multiplier coefficients
and case/readout summaries have separate long-form tables. Noise robustness
has its own table only when configured.

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
- Completed-study selection sources resolve an exact verified run without
  executing it, import only allowlisted feature evolution fields, reject
  profile/data/split/cycle/tamper failures before downstream feature work, and
  bind source run/selection/freeze identities into downstream identity and
  freeze artifacts. Pure plans expose missing dependency status without a
  placeholder scientific value.
- The active study contracts are plan `v4`, identity `v15`, selection `v9`,
  frozen archive/plan `v10`, result row `v3`, event `v1`, and completed
  manifest/summary `v16`. Random-feature lifecycle fields bind selected-only
  evaluation-member materialization and its operation counts across those
  artifacts. `dataset_reference.json` remains `v3`.
- Validation identity/certificate remain `v12`, the foundation contract
  remains `v8`, dataset binding proofs remain `v7`, and the field quadrature
  and matched Model 1 checks remain independently versioned. The numerical
  environment is `v2` and records active package version `0.2.29`.
- Plot-only regeneration requires an existing verified run and is
  transactional.
- Scientific JSON and CLI overrides reject non-finite constants at parse/model
  boundaries, including nested values. Output directory components and
  reporter filenames are path-safe basenames, configured reporters must
  produce their exact declared file set, and transactional failure preserves
  a prior verified run.
- The digital baseline verifies source provenance without parsing physical
  test values before its own freeze/read-back boundary. Fairness parameter
  counts are tensor-cross-checked trainable/fixed/total real scalars per
  realization, with all-realization storage reported separately.
- Backend numerical regression tests use analytic or independent same-runtime
  references and tolerances rather than fixed FFT/SVD/solve byte hashes.

The smoke heat calibration has been exercised during maintenance. No main
heat-calibration profile or production-resolution result is claimed here.

## Historical phase and schema transitions

The table records why older versions may appear in migration code or archived
artifacts. It does not redefine the active versions above.

| Release/transition | Historical responsibility |
|---|---|
| `0.2.6`--`0.2.16` | odd-grid Burgers correction; validated split/binding, quadrature, heat diagnostics, Cartesian landscape, and selection-source contracts |
| `0.2.23` | first independent digital-baseline adapter and its initial artifact families |
| `0.2.24` | strict finite scientific JSON/model/override boundaries |
| `0.2.25` | path-safe artifact components and fail-closed reporter output sets |
| `0.2.26` | pre-freeze digital physical-source test isolation |
| `0.2.27` | tensor-cross-checked real-scalar fairness parameter accounting |
| `0.2.28` | periodic FNO coordinate policy: default `none`, optional `periodic_sin_cos`, rejected legacy ramp |
| `0.2.29` | selected-only random-feature evaluation-member materialization and workload plan/audit v3 |

The portable numerical-regression audit changed tests only; it did not change
package, artifact, or scientific schema versions.

## Characterization and correction coverage

The rows below point to focused characterization or correction coverage for
the reusable numerical and artifact contracts. Later Phase 3--7 workflow
coverage is exercised by study, reporting, digital-baseline, and production
readiness tests in addition to the smoke workflow.

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
| CPU GRF numerical semantics and portability | `tests/test_device_policy.py` compares seeded coefficients and inverse FFT output against an independently constructed odd/even, float32/float64, non-unit-domain reference, with explicit DC/Nyquist handling and a mutation check; no backend FFT byte hash is treated as a cross-version numerical oracle |
| P0-05 fixed-decoder bandwidth formulas and numerical preservation | parameterized odd/even and below/equal/above-bandwidth tests in `tests/test_learning.py`, including invalid-input rejection, observable-prefix recovery, exact suffix zeros, and exact equality with the pre-diagnostic tensor construction |
| P0-05 decoder artifact binding and tamper rejection | `tests/test_study.py::test_direct_decoder_diagnostic_is_bound_across_study_artifacts`, diagnostic-tamper cases, and the frozen-mismatch pre-test guard |
| P0-05 `J x q` independence and foundation characterization | `tests/test_study.py::test_checked_in_observation_output_budget_keeps_q_greater_than_J_cells` and `tests/test_validation_data.py::test_foundation_validation_publishes_passing_certificate` |
| Phase 2-01 split-step real-grid length and parity | `tests/test_numerics.py`, covering RFFT-width ambiguity, required explicit `nx`, independent nonlinear and short-trajectory mathematical references for `nx=15,16` with filtering off/on, same-runtime exact equality against a test-local pre-correction even-grid algorithm, spectral/mask/forcing mismatch rejection, and ETDRK4 parity characterization |
| Review Gate B calibration/test isolation | `tests/test_validation_data.py`, `tests/test_config.py`, and `tests/test_dataset_binding.py`, covering exact preservation of pre-gate split IDs/hash, disjoint full coverage, checked-in smoke/main membership, pre-compute overlap rejection, certificate provenance, certificate/proof tamper rejection, and dataset binding/load reconstruction |
| finite scientific configuration | `tests/test_finite_configuration.py` covers programmatic fields, strict file JSON, CLI overrides, nested JSON values, all config families, and pre-compute/pre-publication failure |
| artifact names and fail-closed reporters | `tests/test_artifact_names_and_reporters.py` covers component policy, filename/format collisions, exact expected outputs, empty/missing/extra output rejection, transactional preservation, and completed-run tamper checks |
| digital source isolation, parameter accounting, and coordinates | focused digital-baseline tests cover provenance-only preflight before freeze, delayed physical test parsing, tensor-cross-checked trainable/fixed/total counts, default coordinate-free circular equivariance, periodic sin/cos construction, and legacy-ramp rejection |
| selected-only random-feature materialization and workload planning | `tests/test_random_feature_lazy.py` and `tests/test_production_readiness.py` cover fit/lift counters, cached maps/lifts, non-selection audit metrics, eager/lazy same-runtime equivalence, operation formulas, unresolved dependencies, and full main catalog enumeration |
| numerical-hash classification | `tests/test_numerical_hash_inventory.py` binds the complete hard-coded hash inventory to `docs/numerical_hash_inventory.md`; random-feature and GRF tests use direct semantics or independent references instead of backend tensor byte goldens |

## Not yet executed or verified

The following limitations are current and must not be mistaken for completed
main results:

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
- the read-only main workload audit requires human CPU-memory, wall-time, and
  storage approval before any production stage; no such approval or main
  execution is recorded in this repository;
- DeepONet is not implemented. FNO1d is implemented only as the independent
  digital baseline adapter; it is intentionally not a physical readout.

See `docs/known_scientific_risks.md` for code-level evidence and impact.

## Publication-label correspondence (reference only)

The current repository contains no authoritative legacy E0--E7 or Figure
mapping. Earlier documentation inferred E0--E4 from partial migration context,
but those inferences are not an authoritative correspondence and are not
repeated as active claims here. In particular, the absence of an E5--E7
mapping says nothing about implementation: the current semantic Phase 5--7
responsibilities are present in the study/report catalog above. Publication
labels and figure numbers must remain documentation-only and must not control
Python names, dispatch, schemas, or artifact kinds.
