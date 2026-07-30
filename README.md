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
  and `q <= n_tar`. There is likewise no general `q <= J` constraint: learned
  readouts map a `J`-dimensional feature to `q` outputs.
- The surrogate initial state is constructed from the finite `n_tar` input.
  Discarded reference-grid modes are never recovered or exposed to the feature
  generator.
- Every validation artifact includes a fixed synthetic matched-dynamics
  diagnostic that independently evolves the finite target and encoded
  surrogate states through heat, Burgers, and reaction-diffusion pipelines,
  checks Model 1 coefficients and q-projected fields, and records the separate
  full-field representation floor.
- Every validation artifact also includes a fixed analytic field-quadrature
  diagnostic. It verifies the periodic trapezoidal `L2` normalization on
  odd/even and non-unit grids, selects a stable reference-grid suffix against
  exact Fourier-orthogonality values, and separately checks fixed-`n_tar`
  data-space metrics and representation floors.
- Scientific configuration is strict: unknown keys are rejected with their
  JSON path.
- Hyperparameter and system selection use the training and validation splits
  only.
- Numerical-foundation and reference-convergence calibration IDs are bound to
  the same deterministic CPU `torch.randperm` split used by datasets. Test
  overlap is rejected before validation computation or artifact publication.
- Random-feature evaluation seeds are independent model realizations. Primary
  test results summarize per-seed metrics; prediction averaging is reported
  separately as an ensemble model.
- The selection record and frozen evaluation plan are written, hashed, and read
  back before the first test state solve or test metric.
- Validation products, datasets, feature states, and completed studies are
  content addressed and verified by exact byte manifests.
- The validated artifact, dataset, feature-state, and study workflow is
  CPU-only. Public validation configuration accepts only `device="cpu"` and
  records that policy independently of whether the installed PyTorch build
  contains CUDA libraries.
- Content-addressed provenance is distinct from target-reference convergence.
  Every dataset declares either a `validated_reference` binding to an exact
  validated candidate suffix or a reason-bearing `foundation_only` binding
  that explicitly makes no target-reference validation claim.
- Directory publication is transactional: incomplete staging trees never
  replace a valid artifact or study run.

## Periodic-domain and GRF convention

Every endpoint-free grid represents the physical periodic domain `[0, L)`,
with spacing `L / nx`. The Gaussian-random-field sampler requires `L`
explicitly and constructs Fourier cycles with
`torch.fft.rfftfreq(nx, d=L / nx)`. Thus integer mode `m` has physical angular
wavenumber

```text
k_m = 2 pi m / L
```

and covariance eigenvalue
`lambda_m = sigma^2 (k_m^2 + tau^2)^(-gamma)`. The PDE solvers, Fourier maps,
GRF archive, and validation binding all use the same configured `L`. The
constant-mode and `(-1)^m` half-domain-shift conventions are independent of
`L`; the latter represents a physical shift by `L / 2`.

## Burgers split-step real-grid contract

The split-step Fourier kernel requires the real collocation length `nx`
explicitly wherever an RFFT coefficient tensor is interpreted. State
coefficients, wavenumbers, and a supplied dealias mask must each have final
length `nx // 2 + 1`; forcing coefficients must have exactly the state
coefficient shape. Every inverse RFFT uses `n=nx`. The two-thirds policy
continues to retain mode indices through `nx // 3`.

Before Phase 2-01, this kernel inferred the inverse-transform length as
`2 * (rfft_width - 1)`, which silently assumes an even grid. Since both
`nx=14` and `nx=15` have RFFT width 8, an odd-grid state was transformed on a
different real grid. Phase 2-01 removes that inference. Independent float64
tests cover direct conservative nonlinear terms and short trajectories on
`nx=15,16`, with dealiasing disabled and enabled. Odd-grid correctness is
covered by this independent mathematical reference. Separately, a test-local
reproduction of the pre-correction even-grid algorithm checks exact tensor
equality with the corrected implementation in the same runtime, for both the
nonlinear coefficients and a short trajectory.

ETDRK4 was not changed: focused parity tests confirm that its real-space
nonlinear inputs and Fourier coefficient lengths remain consistent on odd and
even grids, and that short trajectories preserve shape and finiteness.

Phase 2-04 subsequently adds target-specific reaction-diffusion
characterization and smoke-scale spatial/time/joint convergence. Phase 2-05A
adds the profile-independent matched Model 1 pipeline suite, including
discarded-mode information isolation and a time-mismatch negative control.
Phase 2-05B adds analytic periodic-`L2` characterization and a
profile-independent reference-grid quadrature convergence certificate while
holding the continuous prediction and target fixed. Main-scale heat, Burgers,
and reaction-diffusion certificates remain future validation work.

## Repository layout

```text
pol/
├── config/       strict, discriminated configuration models and loaders
├── digital_baselines/ independent FNO optimization, freeze, evaluation, verification
├── math/         periodic grids, spectral resampling, real Fourier maps
├── systems/      heat, Burgers, and reaction-diffusion evolution systems
├── data/         validation-bound reference datasets and finite-resolution views
├── learning/     observations, fixed decoder, ridge, random features, metrics
├── validation/   independent algebraic and numerical foundation validation
├── study/        cases, trial/search, readouts, freeze protocol, results, verification
├── artifacts/    content-addressed artifact store
├── plotting/     generic reporters over long-form result tables
├── numerics/     import-light numerical kernels
└── runtime/      atomic I/O, hashing, and directory transactions

configs/
├── validation/   foundation-validation profiles
└── datasets/     target-dataset profiles

studies/
└── *.json        question-based combinations of reusable components

digital_baselines/
└── *.json        digital neural-operator baseline profiles
```

`studies/` contains only declarative combinations of reusable components. The
core package does not import it and contains no publication- or
experiment-number control flow.

Within `pol/study`, `cases.py` owns pure planning, `trial.py` coordinates one
validated trial, `readouts.py` and `evaluation.py` own model and metric
primitives, `protocol.py` owns the persisted selection/freeze boundary,
`results.py` owns table/summary serialization, and `verification.py` is the
read-only completed-run verifier. `runner.py` is the public high-level
orchestrator over those responsibilities.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e '.[test]'
```

For an execution-only installation:

```bash
python -m pip install -e .
```

## Execution device policy

The first-paper scientific workflow is formally CPU-only. This scope includes
validation checks and reference convergence, master initial-condition
publication, dataset target generation and loading, feature-state caching,
readout fitting and freezing, test evaluation, diagnostics, and study
publication/verification.

`samples.device` defaults to `"cpu"` and is the only accepted value in the
latest `pol-validation-v6` schema. `"cuda"`, `"auto"`, and unknown values fail during
configuration loading; they are never silently converted to CPU. A
CUDA-enabled PyTorch wheel may still report a non-null `torch_cuda_version` in
the numerical environment fingerprint. That field describes the installed
build, while `execution_device_policy="cpu_only"` and
`compute_device="cpu"` describe this workflow's execution contract.

Low-level tensor algebra may continue to preserve the device of tensors passed
directly to it. That generality is not public end-to-end GPU support. CUDA
dataset generation, solver/cache/readout execution, serialization, diagnostics,
and cross-device reproducibility remain unimplemented.

## Command-line interface

The main workflow commands are:

```bash
pol validate SPEC
pol data build SPEC
pol run STUDY
pol selection inspect|verify STUDY
pol report REPORT
pol digital-baseline SPEC
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
python -m pol validate configs/validation/heat_smoke.json
python -m pol validate configs/validation/reaction_diffusion_smoke.json
```

The validation layer is independent of prediction studies. It checks periodic
resampling and Nyquist handling, the real Fourier projector, finite-input/no-
leak behavior, valid use of `n_tar < J`, fixed-decoder behavior and aliasing,
and target-specific reference validation selected by the strict
`target_reference` union. `burgers_convergence` owns Burgers spatial/time
candidate refinement. Its optional cross-solver diagnostic first checks
independent split-step and ETDRK4 self-convergence sequences on the finest
grid, then stores a symmetric finest-solution discrepancy without naming
either method as truth. That supporting block never extends the primary
dataset-binding suffix. `heat_analytic` checks the exact spectral Fourier
multiplier independently on constant, sine, cosine, multimode, odd/even,
non-unit-domain, and float32/float64 cases, then measures spatial truncation
convergence over reference-resolution candidates. Heat records
`temporal_status=analytic_exact` and has no time-step candidates.
`reaction_diffusion_convergence`
constructs independent expectations for zero/constant/nonzero-equilibrium
fields and the `beta=0` one-step Fourier multiplier, then reuses the generic
spatial, time-candidate, and actual selected-versus-finest joint convergence
path. Its time candidates keep
`semi_implicit_spectral_euler` and `nonlinear_filter` fixed while `dt`
strictly decreases; switching between `none` and `two_thirds` is rejected.
A passing certificate and master initial-condition archive are published under
`artifacts/validations/<content-hash>/`.
The certificate records separate machine-readable foundation and generic
target-reference contracts, selected candidate indices, complete ordered
candidate lists, exact allowed suffixes, and master-archive tensor hashes.
The P0-03 foundation contract additionally binds the actual GRF sampler
semantics and sampler `domain_length` to the resolved domain, serialized
master archive, and downstream dataset proof. P0-04 adds a hashed CPU-only
execution policy to the numerical environment, validation identity,
certificate, foundation contract, and master archive. Publication and loading
verify that all master tensors are on CPU.
P0-05 adds a certificate-bound case with requested `q` above the directly
observable fixed-decoder bandwidth. It records the requested/observable
bandwidth, half-open zero-filled coefficient range, zero-filled mode range,
observable-prefix error, and an exact-zero check for the filled suffix.
Review Gate B additionally binds the configured calibration IDs to the
dataset split policy, counts, seed, and canonical split hash. The certificate
records each calibration ID's train/validation membership and a zero test-
overlap count; loading reconstructs this provenance and rejects tampering.

### 2. Build or reuse a target dataset

```bash
python -m pol data build configs/datasets/burgers_smoke.json
```

A dataset is built only from a passing validation certificate and an explicit
`pol-dataset-v3` binding:

- `validated_reference` requires the target system, invariant PDE parameters,
  evolution time, dtype, and domain to match the certificate exactly.
  `reference_nx` and the canonical numerical condition must be exact members
  of their respective validated candidate suffixes. Heat binds to
  `{"solver":"spectral_exact"}`. Burgers binds the complete canonical
  solver condition: requested outer/fine steps, outer-step count, actual
  effective substep, substeps per outer step, and dealias policy.
  Reaction-diffusion binds `nu`, `alpha`, `beta`, final time, domain, dtype,
  `reference_nx`, and the exact canonical
  `solver`/`dt`/`nonlinear_filter` tuple.
- `foundation_only` reuses only the checked initial-condition foundation and
  master archive. It requires a nonempty reason and records
  `target_reference_validation_status=not_claimed`.

The binding proof is evaluated before target evolution. It verifies the same
split policy/version, counts, seed, canonical split hash, and calibration
provenance recorded by validation. Its proof hash, split IDs, reference inputs,
reference targets, target-solver metadata, and tensor hashes are stored under
`artifacts/datasets/<content-hash>/`. For example, the Burgers smoke dataset
may use `reference_nx=64` after selection at 32 because 64 is an actual member
of the certificate's validated suffix; 48 would be rejected even though it is
larger than 32.

### 3. Run a study

A single condition is a one-cell study. Sweeps are produced by `global_axes`,
variant overrides, or a variant search specification; they are not separate
runner types.

```bash
python -m pol run studies/heat_readout_calibration_smoke.json
python -m pol run studies/surrogate_parameter_time_coordinate_search_smoke.json
python -m pol run studies/surrogate_parameter_time_landscape_smoke.json
python -m pol run studies/observation_output_budget_smoke.json
python -m pol run studies/input_simulation_resolution_smoke.json
```

Inspect the expansion without creating artifacts or output directories:

```bash
python -m pol run studies/surrogate_parameter_time_landscape_smoke.json --plan
```

Override an existing field without editing JSON:

```bash
python -m pol run studies/surrogate_parameter_time_landscape_smoke.json \
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

# Build a content-addressed report from existing verified study runs only.
python -m pol report reports/surrogate_operator_summary_smoke.json

# Verify the report artifact and every recorded output byte.
python -m pol report verify outputs/reports/<report>/<profile-hash>
```

### 4. Run or verify the digital FNO baseline

The digital baseline shares the validated dataset, canonical split, finite
`n_tar` input, `q` output, and field/data metrics with the physical studies,
but it has its own optimizer/checkpoint adapter:

```bash
python -m pol digital-baseline digital_baselines/fno1d_smoke.json --plan
python -m pol digital-baseline digital_baselines/fno1d_smoke.json
python -m pol digital-baseline verify outputs/digital_baselines/<name>/<profile-hash>
```

The exact physical baseline source must already exist and verify; the command
never starts that source implicitly. Architecture/checkpoint selection is
validation-only. Frozen evaluation-seed checkpoints and the evaluation plan
are hashed and read back before the test finite view is requested. Primary
test results summarize independent training-seed metrics; prediction averaging
is stored separately. The fairness CSV records distinct physical and digital
inference paths and forbids wall-clock/energy claims without a common
measurement protocol.

### 5. Verify an artifact or study run

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

The unqualified `field_*` metrics reconstruct the prediction on the dataset
`n_ref` grid and use that grid as quadrature for the continuous periodic `L2`
norm. Whether that target-reference condition is convergence validated is
reported separately by the dataset binding status. `data_field_*` metrics are
also reported on the finite `n_tar` target grid. This distinction prevents a
change in `n_tar` from silently changing the meaning of the principal error
metric and keeps the dataset's target-specific validation claim explicit.
For endpoint-free values `v_j`, the implemented norm is
`sqrt((L/n) * sum_j v_j^2)`. The validation foundation checks it against exact
constant, trigonometric-mode, and orthonormal-Fourier Parseval formulas, then
varies only `n_ref` over `[8,15,16,31,32]`; the fixed-`n_tar=16`
`data_field_*` quantities must remain unchanged.

A `StudySpec` adds:

- publication-independent variants;
- global sweep axes;
- static, grid, or coordinate search;
- validation-only selection;
- optional surrogate-resolution convergence;
- diagnostics such as heat multipliers or readout stability under observation
  noise;
- generic reporters over result tables.

The implemented feature-generator kinds are:

- `pde_dynamics`: evolve the finite input with a configured surrogate PDE;
- `static_input`: observe the encoded finite input without evolution, for a
  no-dynamics baseline through the same trial and study path.

`studies/dynamic_feature_baseline_comparison*.json` fixes one shared
`n_tar/n_sur/J/q`, observation, and readout-candidate contract across
`static_input`, heat, Burgers, and reaction-diffusion features. Dynamic
conditions are imported from a verified completed validation selection;
static input carries an explicit no-evolution marker. The completed run writes
`selected_comparison.csv` by joining validation-selected rows to primary
frozen-model test rows. Random-feature ensembles remain in their separate
table.

`studies/readout_stability_noise*.json` consumes only the persisted and
read-back frozen models selected by the parameter/time study. It records
clean and noisy test metrics in per-repeat long form, learned readout norms,
centered base-feature and readout-design covariance spectra/ranks/condition
numbers, and explicit relative-global-feature-RMS noise coordinates.
Deterministic repeat uncertainty, independent random-feature seed uncertainty,
and prediction-ensemble noise results are written with distinct semantics and
tables. The checked-in main profile is a plan only; no production result is
claimed.

`studies/learning_curve*.json` uses the canonical train split order to form
strict nested prefixes. Each configured `n_train` is an experimental
condition, not a test-selected candidate. The subset IDs, content hash,
parent-train hash, policy, and version are bound into validation/test rows,
selection records, frozen plans, and frozen model entries. Feature states are
computed for the canonical train-plus-validation sample request and reused
across prefix sizes; only readout fitting sees the shorter prefix. All sizes
are frozen before any test access. The direct decoder is retained as an
explicit training-size-invariant control, and random-feature seed statistics
and prediction ensembles keep their existing separate semantics. The main
sizes are `50,100,200,400,800,1000`, matching the configured 1000-sample
canonical train split; this profile has not been executed.

`studies/random_feature_seed_statistics*.json` fixes the Phase-3-selected
Burgers feature condition and chooses random-feature width, scales, bias, and
ridge coefficient using training/validation data only. Selection seeds and
evaluation seeds are disjoint. The smoke profile has three independent
evaluation realizations; the planned, unexecuted main profile has 32. Per-seed
rows bind the random-map and complete frozen-member content hashes, evaluation-
seed validation metrics that are explicitly marked as not used for selection,
and learned-readout norms. The random-map hash joins those rows to the
conditioning information in `readout_stability_models.csv`. Scatter, box, and
empirical-CDF reporters read the completed CSV tables and do not interpolate
the seed axis.

The same study predeclares representative test sample IDs and explicit
random-feature member seeds in `prediction_capture`. After the frozen archive
has been read back and the canonical test pass has produced predictions,
`prediction_capture.pt` stores finite inputs, finite/reference targets,
target/prediction coefficients, `n_tar`/`n_ref` reconstructions, coefficient
errors, and content identities. To avoid storing every full test prediction,
it stores full fields only for the predeclared samples and stores
per-coefficient/per-mode aggregates over the complete canonical test split.
The aggregate definition, DC/cos/sin ordering, physical wavenumbers, and
machine-epsilon zero-denominator policy are artifact fields.

The implemented readout kinds are:

- `direct_fourier_decoder`: fixed, untrained decoding. For `J` equispaced
  observations it directly retains `J` real coefficients when `J` is odd and
  `J-1` when `J` is even; a larger requested odd `q` remains valid and is
  completed by structural zero-fill;
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
- nested-training subset IDs and hashes within the selection/freeze records
- `convergence.csv`
- `frozen_models.pt`
- `frozen_evaluation_plan.json`
- `test_metrics.csv`
- `random_feature_seed_metrics.csv`
- `random_feature_ensemble_metrics.csv`
- optional `prediction_capture.pt`
- `readout_stability_models.csv`
- `readout_stability_noise_repeats.csv`
- `readout_stability_noise_summary.csv`
- separately labeled readout-stability prediction-ensemble CSV files
- other diagnostic CSV files
- `events.json`
- `run_summary.json`
- `figures/`
- `manifest.json`

Numerical tables and tensor captures are transactionally published and
verified first. If plots are requested, a second transaction reads that
verified completed run, produces figures, updates only report metadata and
the manifest, and verifies the result again. Automatic post-run reporting and
`--plots-only` use this same read-only path. A failed report transaction leaves
the verified numerical run intact and does not restart solvers or readout
inference.

Cross-run reporting is a separate read-only artifact family. A strict
`pol-report-v1` specification names at least two source study specifications
and predeclares each split, metric, variant, readout, axis, and baseline row.
Every exact content-addressed source run is verified before rendering.
`phase_diagram_report` writes a cell-status table that distinguishes valid,
missing, and reason-bearing invalid cells and rejects numerical NaN/Inf.
`baseline_summary_table` keeps reference-field and finite-data errors in
separate columns, carries both representation floors and all five dimensions,
and reports Model 3 seed mean/standard deviation/Student-t interval without
placing prediction-ensemble metrics in the primary columns. Reports are
published under `outputs/reports/` with resolved specification, source
references, unrounded machine-readable CSV, optional Markdown/LaTeX, figures,
summary, and an exact-byte manifest. Source storage paths are excluded from
the report identity.

Production operation is documented in
[`docs/production_runbook.md`](docs/production_runbook.md). Main execution is
split into explicit stages, each guarded by `POL_CONFIRM_MAIN=YES`; there is no
all-in-one main stage. `python3 scripts/plan_main.py` strict-parses and plans
every main validation, dataset, study, and report without executing or
publishing one.

`events.json` records the durability boundary. In a valid run,
`freeze_read_back` appears before `first_test_state_solve` and
`first_test_metric`.

`test_metrics.csv` is the primary comparison table. Deterministic readouts have
`test_result_kind=single_model`. For `random_feature_ridge`, every canonical
`test_*` metric is the arithmetic mean of metrics computed independently for
each frozen evaluation seed, and the row has
`test_result_kind=independent_seed_metric_summary`. Each canonical metric is
accompanied by:

```text
<metric>_seed_mean
<metric>_seed_std
<metric>_seed_ci95_low
<metric>_seed_ci95_high
<metric>_seed_q25
<metric>_seed_median
<metric>_seed_q75
```

The standard deviation uses Bessel's correction (`ddof=1`), and the two-sided
95% interval for the arithmetic mean uses a Student-t quantile. Seed count,
confidence level, and interval method are recorded in the primary row. The
linearly interpolated quartiles are descriptive distribution summaries and
are explicitly not labeled as an uncertainty interval.

`random_feature_seed_metrics.csv` contains one
`independent_seed_realization` row per frozen evaluation seed, including its
random-map hash, frozen-member hash, validation metric, selected structural
hyperparameters and ridge coefficient, and readout norm fields.
`random_feature_ensemble_metrics.csv` contains one separately labeled
`prediction_ensemble` row per selected random-feature model; its metric names
begin with `test_ensemble_`. Ensemble metrics are not copied into the canonical
primary metric columns. The ensemble row binds both the ordered seed hash and
the ordered frozen-member hash. `run_summary.json` records primary, per-seed,
and ensemble row counts. `dataset_reference.json` stores the full dataset
binding proof, while both it and `run_summary.json` expose the binding kind,
binding status, target-reference validation status, and proof hash.

For every `direct_fourier_decoder` row, `validation_trials.csv` and
`test_metrics.csv` publish:

```text
decoder_policy
decoder_observation_count
decoder_requested_q
decoder_observable_q
decoder_retained_q
decoder_requested_max_mode
decoder_observable_max_mode
decoder_zero_filled_mode_count
decoder_zero_filled_coefficient_count
decoder_zero_fill_applied
```

The same diagnostic is bound into the direct readout's inner selection record,
frozen model, and frozen evaluation plan. Learned readout rows leave the union
CSV columns empty. Completed-run verification recomputes the diagnostic from
`J` and `q`, rejects false diagnostics on learned readouts, and cross-checks
the validation, selection, frozen, and test copies. `run_summary.json` records
the selected direct-decoder diagnostic count, zero-fill count, and any-zero-
fill flag.

Study configuration through `pol-study-v6` is accepted, and pure plans are
`pol-study-plan-v3`. Newly written study-run identity, manifest, and summary
schemas are respectively `v13`, `v14`, and `v14`, with explicit
`pol-study-result-row-v3` validation/test rows. Older supported completed runs
remain readable as selection sources. Selection remains `v8`; the frozen-plan
and frozen-model schemas are `v9`, prediction captures use
`pol-prediction-capture-v1`, and study dataset-reference remains at `v3`. Heat
multiplier coefficient rows use
`pol-heat-multiplier-coefficient-v2`, and case/readout summaries use
`pol-heat-multiplier-summary-v1`. Validation
configuration is `pol-validation-v6` for reaction-diffusion (existing v5
heat/Burgers specs remain parseable for migration); Phase 2-05B advances
validation identity/certificate to `v12`, adds
`pol-field-quadrature-check-v1`, retains
`pol-matched-model1-pipeline-check-v1`, and retains nested
`pol-burgers-cross-solver-spec-v1` /
`pol-burgers-cross-solver-check-v2` semantics. The foundation contract is
`v8`, and the generic primary target-reference contract remains `v4`.
Dataset configuration remains `pol-dataset-v3`, binding proofs are `v7`,
and dataset identity/metadata/archive/resolved-spec families remain `v5`.
The foundation master binding remains at `v3`, initial-condition archives at
`v4`, and feature-state identity/archive/metadata at `v2`. Earlier validation
artifacts are rejected rather than silently treated as carrying the
Phase 2-05B semantics. Long-form primary and self-convergence rows use
`pol-reference-convergence-row-v3`; the primary CSV is
`pol-reference-convergence-csv-v3`. Cross-run configurations use
`pol-report-v1`; identities, manifests, summaries, phase-map tables, and
baseline tables use their corresponding `pol-*-v1` report schemas. Digital
FNO configurations use `pol-digital-baseline-v1`; their identity, selection,
frozen-checkpoint, frozen-plan, summary, and manifest families use
corresponding `pol-digital-*-v1` schemas. Package version `0.2.23` is recorded
by numerical-environment schema
`pol-numerical-environment-v2`.

The observation/output budget and input/simulation-resolution study bind both
their Burgers and reaction-diffusion feature systems and evolution times to
the corresponding validation-selected representative conditions in the
completed parameter/time landscape. The binding verifies the exact completed
source run and its selection/frozen artifacts and places path-independent
source provenance in downstream identity, freeze artifacts, and result rows.
Read-only inspection is available through:

```bash
python3 -m pol selection inspect studies/surrogate_parameter_time_landscape_smoke.json
python3 -m pol selection verify studies/observation_output_budget_smoke.json
python3 -m pol selection verify studies/input_simulation_resolution_smoke.json
```

These commands do not run a study or build a missing dependency. A pure plan
reports `selection_dependencies.status="missing"` until the expected source
run exists.

## Included profiles

Smoke profiles are intentionally small and are used for integration checks:

- `heat_readout_calibration_smoke.json`
- `surrogate_parameter_time_coordinate_search_smoke.json`
- `surrogate_parameter_time_landscape_smoke.json`
- `observation_output_budget_smoke.json`
- `input_simulation_resolution_smoke.json`
- `digital_baselines/fno1d_smoke.json`

Both Burgers and heat dataset profiles use `validated_reference`. Burgers
binds to the Burgers convergence certificate; heat binds only to its
heat-specific analytic/spatial certificate and the exact
`spectral_exact` condition. Burgers smoke enables the small cross-solver
diagnostic. Burgers main declares it disabled and has not been executed or
cross-solver calibrated.

The two phase diagrams are deliberately separate. The observation/output
budget varies `J × q` while holding `n_tar` and `n_sur` fixed. Its checked-in
scope is exactly the Burgers and reaction-diffusion feature variants, and it
produces separate validation maps for direct, affine, and random-feature
readouts. Its predeclared per-cell test values evaluate the budget axes and
do not select a cell. The completed summary distinguishes global-axis
combinations from variant-expanded planned/evaluated/skipped case counts. The
`input_simulation_resolution` map varies `n_tar × n_sur` while holding `J`
and `q` fixed. It uses the same two Phase 3-selected feature variants and the
direct, affine, and random-feature readouts, but has its own study and run
identity.

The surrogate-parameter coordinate search and complete Cartesian landscape
are also separate studies. The former alternates the `nu` and readout-time
axes as an efficient search aid. The latter evaluates every predeclared
`nu × time` cell for heat, Burgers, and reaction-diffusion features, records
the validation-selected representative feature condition, and renders
validation-only `metric_map` figures with the selected cell marked.

The corresponding non-smoke profiles retain the high-resolution research
settings. They can require substantial CPU time, storage, and memory and are
not run by the test suite. The main FNO plan declares two architecture
candidates, five selection seeds, ten evaluation seeds, 100 epochs/model, and
a 64,000 optimizer-step upper bound. It has not been executed, and no main
landscape or digital-baseline result is claimed here.

## Tests

```bash
pytest -q
```

The suite covers Fourier and Nyquist behavior; split-step RFFT-width
ambiguity, explicit-`nx` validation, odd/even nonlinear and short-trajectory
references, exact even-grid preservation, and ETDRK4 parity; GRF
physical-domain covariance scaling on odd and even grids; unit-domain and
P0-04 CPU deterministic regressions; finite-input information isolation;
deterministic split ID/hash preservation; calibration/test separation before
validation compute; calibration certificate/binding provenance and tamper
rejection;
strict CPU-only configuration; CPU tensor invariants at artifact boundaries;
execution-policy tamper detection; dimension independence; fixed and learned
readouts; artifact tamper detection; fixed-decoder observable bandwidth and
exact zero-fill regression; learned `J -> q` readouts with `q > J`;
decoder-diagnostic binding/tamper detection; analytic periodic norms,
reference-grid quadrature convergence, field/data metric separation, and
quadrature certificate tamper detection; foundation validation; dataset
splitting; unified scalar/sweep planning; selection freezing; test-order
enforcement; plot regeneration; CLI behavior; and the absence of
publication-number namespaces from the core package.
Focused digital-baseline tests additionally cover strict CPU configuration,
finite-input-only FNO execution, missing-source preflight, validation-only
selection, freeze/read-back before test access, checkpoint tamper rejection,
independent training-seed statistics, separate prediction ensembles,
deterministic smoke metrics, shared metric implementation, and absence of a
physical-study dependency on the digital adapter.

For a complete local smoke sequence:

```bash
./scripts/run_smoke.sh
```

Additional architectural details are in
[`docs/architecture.md`](docs/architecture.md), and the mapping from the former
experiment-specific layout is documented in
[`docs/migration.md`](docs/migration.md).
