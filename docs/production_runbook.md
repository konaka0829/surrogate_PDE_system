# Production runbook

## Scope and hard gate

This runbook fixes the main-profile order without claiming that any main
result has been executed. Maintenance and automated tests must never run a
main profile.

There is no all-in-one main command. Each production invocation names exactly
one stage and requires the literal environment confirmation:

```bash
POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage STAGE
```

Without `POL_CONFIRM_MAIN=YES`, every execution stage exits before invoking
`pol`. The read-only audit is the only exception:

```bash
scripts/run_main.sh --stage audit
```

## Environment and installation

Use a dedicated, recorded Python environment on the production host. From the
repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -c 'import pol, torch; print(pol.__version__); print(torch.__version__)'
python -m compileall -q pol tests
pytest -q
./scripts/check.sh
```

The current package version is `0.2.30`. Save the Git commit, dirty status,
Python/package versions, host/CPU information, and the audit JSON with the
production log. Do not change code, specifications, dependencies, or the
Python environment between stages; those changes alter content identities.

The validated workflow is CPU-only. Main validation specifications use
`float64`, and checked-in study execution uses one PyTorch thread and batch
size 64. Do not change device, dtype, thread count, or batch configuration
inside a resumed identity. CUDA availability in the PyTorch build is metadata,
not authorization to execute these workflows on CUDA.

## Storage and free space

Artifacts live under `artifacts/`; study, digital-baseline, and cross-run
results live under `outputs/studies/`, `outputs/digital_baselines/`, and
`outputs/reports/`. Storage paths are excluded from
scientific identity, while all published file bytes are manifest-verified.

Before every stage:

```bash
df -h .
du -sh artifacts outputs 2>/dev/null || true
```

Both main datasets use `n_ref=4096` and 1,400 samples. This is a configuration
fact, not a complete byte estimate: validation candidates, master fields,
feature-state caches, frozen members, staging directories, figures, and
diagnostics add substantial storage. Transactional replacement can
temporarily retain the verified final directory, a full staging directory,
and a backup. Operationally, keep at least three times the largest measured
single-run directory free, in addition to expected dataset/cache growth.
Measure on the actual filesystem; do not infer production capacity from smoke
directory size alone.

## Read-only plan audit

Run and archive:

```bash
python3 scripts/plan_main.py > production_plan_audit.json
```

`pol-production-plan-audit-v3` strict-parses all three main validations, both
datasets, nine studies, the independent FNO baseline, and the cross-run
report. It reports parse/plan status, the main marker, case/candidate/model
upper bounds, inner ridge/map/lift/SVD operation counts, shape budgets,
dependency state, and expected output families. It does not build a validation
or dataset, execute a study or
digital training run, resolve a missing dependency by running it, or write a
repository artifact.

At this gate, missing downstream dependencies are expected because main has
not been run. A parse or plan failure is not expected and must be corrected
before production. Re-run the audit after every completed stage; dependency
states should progress from `missing` to verified/completed in the order
below.

## Exact current workload counts

These counts come from strict parsing and pure case expansion, not from an
executed main result.

| Study | Cases | Candidate trials | RF maps | Train/validation lifts | RF selection ridge fits | Lazy evaluation fits | Legacy eager total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `heat_readout_calibration` | 12 | 12 | 1,080 | 2,160 | 6,480 | 120 | 6,600 |
| `surrogate_parameter_time_coordinate_search` | 2 | 120 | 10,800 | 21,600 | 64,800 | 20 | 66,000 |
| `surrogate_parameter_time_landscape` | 3 | 75 | 6,750 | 13,500 | 40,500 | 30 | 41,250 |

The landscape's 75 candidates are the complete three-variant 5-by-5
parameter/time grid. Its legacy eager total is
`40,500 selection fits + 750 evaluation-member fits = 41,250`; the lazy
lifecycle instead performs `40,500 + 30 = 40,530` fits. The three rows above
are resolved without an upstream selection source. Until the landscape is
completed and verified, the other six studies report
`unresolved_selection_dependency` and `counts=null`; the audit deliberately
does not invent workload values from placeholder feature conditions.

The machine-readable formulas are:

- candidate feature-state solves are bounded by declared candidate trials;
- an affine readout contributes one fit per declared zeta and one SVD for each
  zero zeta;
- random-feature structures are
  `widths × weight_scales × bias_scales`;
- unique random maps are
  `candidates × structures × selection_seeds`;
- train/validation lifts count two tensor lifts per unique map;
- selection ridge fits are
  `candidates × structures × selection_seeds × zetas`;
- lazy evaluation fits are evaluation seeds for the one selected candidate of
  each case/random-feature readout;
- the legacy eager comparison is evaluation seeds for every candidate;
- zero-zeta evaluation SVDs are reported as an upper bound because the
  validation-selected zeta is unknown at pure-plan time.

Maximum lifted dimension, target dimension, canonical training count,
convergence solves/comparisons, and declared noise-coordinate evaluations are
stored alongside these counts. They are operation/shape budgets, not elapsed
time estimates.

Before any main execution stage, a human must archive and sign off the
`workload` block from `production_plan_audit.json`, including available CPU
memory, wall-time allocation, and storage capacity. Absence of that review is
a stop condition. Re-run and review the audit whenever a dependency becomes
resolved; do not copy numerical counts from an earlier unresolved plan.

Each validation has 1,400 samples and three reference-resolution candidates.
Burgers and reaction-diffusion each have three numerical-condition candidates;
analytic heat has one exact numerical condition. The report declares four
source runs and four reporters: three phase diagrams and one 12-row baseline
table.

The digital FNO plan is separate from those studies:

| Candidates | Selection seeds | Evaluation seeds | Epochs/model | Models trained | Optimizer-step upper bound | Fairness rows |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5 | 10 | 100 | 20 | 64,000 | 12 physical + 1 digital |

This is a declared CPU training budget, not runtime or accuracy evidence.
Architecture candidates are selected with validation-only statistics. The ten
evaluation seeds are independent trained models; their prediction average is
a separate ensemble.

## Timing evidence and assumptions

On the maintenance workstation, a forced
`random_feature_seed_statistics_smoke` run completed in 6.12 seconds wall
time on 2026-07-30. That benchmark uses smoke sample counts, resolutions, and
only three evaluation seeds; it is evidence that the execution path works,
not a production runtime estimate. Main costs do not scale by one known
factor because PDE resolution, time stepping, cases, cache reuse, diagnostics,
and 32-seed evaluation change differently. Record per-stage wall time and peak
resident/storage usage on the production host. Do not schedule later stages
from an unqualified smoke-to-main extrapolation.

## Stage order and commands

Run one command block at a time. Capture the JSON output; it contains the exact
content-addressed path.

### 1. Validation

```bash
POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage validation
```

This publishes the Burgers foundation/reference certificate, the analytic
heat certificate, and the reaction-diffusion certificate. Verify each printed
path:

```bash
python -m pol verify ARTIFACT_PATH
```

Stop if any certificate status is not `pass`.

### 2. Datasets

```bash
POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage datasets
```

The heat dataset binds only to the heat certificate; the Burgers dataset binds
to the Burgers certificate. Verify both printed paths. Confirm
`binding_status=pass` and
`target_reference_validation_status=validated`.

### 3. Independent heat calibration

```bash
POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage heat-calibration
```

This stage uses the heat dataset and is independent of the parameter/time
selection chain.

### 4. Parameter/time search aid

```bash
POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage parameter-time-search
```

This coordinate search is an efficient validation search aid. It is not the
complete phase diagram and must not replace the next stage.

### 5. Complete parameter/time landscape

```bash
POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage parameter-time-landscape
```

The script runs the landscape and then performs read-only selection
inspection. Verify the printed study-run path before continuing.

### 6. Mandatory manual validation-selection checkpoint

Do not start a downstream stage until this checkpoint is signed off.

```bash
python -m pol selection inspect studies/surrogate_parameter_time_landscape.json \
  > parameter_time_selection.json
```

For each heat, Burgers, and reaction-diffusion representative:

- confirm `selection_metric` begins with `validation_`;
- record case, variant, representative readout/candidate, metric value, full
  feature system, and evolution time;
- confirm source selection/frozen hashes are present;
- verify the completed landscape path again;
- confirm the record states that test tables were not used for condition
  selection.

This is a review of a validation-selected frozen condition. Do not replace it
after viewing any test metric.

### 7. Physical baseline and digital FNO

First execute and verify the physical comparison source:

```bash
POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage baseline-comparison
```

Only after that exact source verifies, run the independent digital adapter:

```bash
POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage digital-baseline
python -m pol digital-baseline verify DIGITAL_RUN_PATH
```

The digital command must fail if the physical source is absent or invalid; it
must not start the physical study implicitly. Review
`selection_record.json`, `frozen_evaluation_plan.json`,
`training_compute.json`, and `fairness_comparison.csv`. Confirm that the
primary row uses independent evaluation-seed metrics, the prediction ensemble
is separate, energy is `not_measured`, and wall-clock/energy comparison is
disallowed without a common protocol.

### 8. Other downstream studies

Each stage first verifies its completed-selection binding and then uses the
ordinary unified study runner:

```bash
POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage readout-stability
POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage learning-curve
POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage random-feature-seeds
POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage observation-output
POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage input-simulation
```

Run these as separate invocations, inspect the output, and verify the printed
path after each one. Re-run `scripts/plan_main.py` between stages when
dependency state or capacity is in doubt.

### 9. Cross-run report

The report requires verified completed landscape, baseline, observation/output,
and input/simulation runs:

```bash
POL_CONFIRM_MAIN=YES scripts/run_main.sh --stage cross-run-report
python -m pol report verify REPORT_PATH
```

Report generation is read-only with respect to every source and cannot start a
missing study. It writes phase-map machine tables and figures plus the
unrounded baseline CSV and formatted tables.

## Resume, reuse, force, and plots

Default execution is the resume mechanism. Re-running the same stage computes
the same expected identity, verifies an existing completed output, and reuses
it. A missing final directory causes that one stage to execute. A missing
upstream dependency causes downstream selection/report commands to fail; it
does not implicitly run the upstream stage.

An ordinary interruption should be followed by:

1. inspect the printed/error path and sibling `.staging-*`/`.backup-*`
   directories without deleting anything;
2. verify the final directory if present;
3. re-run the same single stage without `--force`.

Do not use `--force` merely to resume. It intentionally recomputes and
transactionally replaces the same identity and may need space for old,
staging, and backup copies. If a scientifically justified recomputation is
required, run the exact direct command with `--force` only after backing up
and verifying the current final directory.

Figures for one existing study may be regenerated without inference:

```bash
python -m pol run STUDY_SPEC.json --plots-only
```

This requires the exact verified completed run and preserves numerical tables
and frozen tensors. Re-running `python -m pol report REPORT_SPEC.json` reuses
the exact verified cross-run report; `--force` rerenders transactionally from
verified sources.

## Verification, backup, and failure recovery

Every validation, dataset, study, digital-baseline, and report command verifies
before successful return. Independently repeat:

```bash
python -m pol verify EXACT_VALIDATION_DATASET_OR_STUDY_PATH
python -m pol digital-baseline verify EXACT_DIGITAL_RUN_PATH
python -m pol report verify EXACT_REPORT_PATH
```

For a material completed stage, copy the exact hash directory to a different
filesystem without changing its contents, then run the same verifier on the
copy. Record the source path, destination, manifest SHA-256, and verification
result. Use explicit paths; never use a broad artifact/output root as a
destructive or replacement target.

A failed transaction must not be treated as a completed run. Preserve error
logs and any deliberately archived failure artifact. If a valid final
directory still verifies, keep it and diagnose the new attempt separately.
If only a `.backup-*` directory remains after host/process failure, do not
rename or delete it until its manifest verifies and the corresponding final
path is confirmed absent. Restore or replace directories only under an
operator-reviewed, logged recovery step.

Tamper, missing/extra files, source-identity disagreement, or report/source
byte mismatch is a hard stop. Do not refresh a manifest to bless unexplained
bytes.

## What test data must never select

Test data and test tables must never select or alter:

- target/surrogate system family, numerical method, parameters, or readout
  time;
- surrogate-resolution convergence reruns;
- ridge coefficient, random-feature width/scales, or selection seeds;
- `n_tar`, `n_sur`, `J`, `q`, learning-curve size, or phase-diagram cell;
- evaluation seed membership or which seed represents Model 3;
- representative prediction sample/member coordinates;
- FNO architecture, training/checkpoint seed membership, or checkpoint epoch;
- baseline/report rows, metrics, or display coordinates;
- whether a failed/missing stage is rerun.

Test results evaluate already declared/frozen coordinates. Model 3 primary
results aggregate independent per-seed metrics with mean, standard deviation,
and the declared Student-t interval. The prediction ensemble remains a
separately labeled model/table and must never replace a primary column.

## Current readiness statement

All main files are strict-parse/plan inputs only. The audit currently reports
downstream, digital, and report sources as missing until their main
prerequisites are executed. Smoke validation, datasets, all mandatory
studies, the optional FNO adapter, prediction/report paths, and the four-source
smoke report pass. No main profile was executed during preparation of this
runbook.
