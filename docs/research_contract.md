# Research contract

## Purpose of the surrogate dynamics

The surrogate PDE is a feature generator, not an information-recovery
mechanism. It cannot create information that is absent from the finite input.
Its scientific purpose is to form a representation that is easier to read out
under the finite observation budget and the configured readout class.

Every result must therefore be interpreted as the performance of the complete
finite-information pipeline, not as recovery of discarded reference modes.

## Dimensions and ownership

| Symbol | Owner | Meaning |
|---|---|---|
| `n_ref` | validation-bound dataset specification and artifact | reference grid used to generate and evaluate the reference field |
| `n_tar` | finite-input specification | finite data grid exposed at the learning interface |
| `n_sur` | feature-generator specification | internal grid of the surrogate evolution or static feature state |
| `J` | observation specification | number of observed feature values |
| `q` | output specification | number of real Fourier coefficients predicted by the readout |

These quantities are independent and must not be inferred from one another.
In particular, there is no general `n_tar <= J` assumption. The generic
representability conditions required by the current interfaces are:

- `J <= n_sur`;
- `q <= n_tar`;
- `q` is odd.

An individual component may have an additional, explicitly documented
limitation. Such a limitation must not be promoted to a general dimension
constraint.

## Fixed Fourier decoder bandwidth

The current real Fourier coefficient order is
`[a_0, a_1, b_1, a_2, b_2, ...]`, and requested `q` is a positive odd
integer. For the Model 1 fixed decoder applied to `J` equispaced point
observations, the canonical diagnostic is

```text
observable_q = J if J is odd else J - 1
retained_q = min(requested_q, observable_q)
requested_max_mode = (requested_q - 1) // 2
observable_max_mode = (observable_q - 1) // 2
zero_filled_mode_count =
    max(0, requested_max_mode - observable_max_mode)
zero_filled_coefficient_count = requested_q - retained_q
zero_fill_applied = requested_q > observable_q
```

The even-`J` Nyquist coefficient is not treated as a directly observable
cosine/sine pair in this odd-`q` basis. The decoder analyzes the observable
prefix exactly as before and inserts exact zeros for the remaining requested
coefficients. A request with `q > observable_q`, including `q > J`, is valid
and must not be converted into an interface error.

This bandwidth limitation belongs only to `direct_fourier_decoder`. Affine
and random-feature readouts learn a map from `J` features to `q` outputs and
must not receive a general `q <= J` constraint. It also provides no basis for
an `n_tar <= J` constraint.

## Periodic domain and Gaussian random fields

An endpoint-free grid with `nx` points represents `[0, L)` with spacing
`d = L / nx`. The same finite, strictly positive `L` must be used by the GRF
sampler, PDE evolution, spectral resampling, Fourier analysis/synthesis,
archive metadata, and validation/dataset binding.

The GRF sampler requires `domain_length` explicitly. Its integer Fourier mode
`m` uses physical angular wavenumber
\(k_m = 2\pi m/L\), and the configured covariance eigenvalue remains
\(\lambda_m = \sigma^2(k_m^2+\tau^2)^{-\gamma}\). The deterministic constant
mode, real-field conjugate symmetry, even-grid Nyquist treatment, random draw
order, and mode-parity factor `(-1)^m` retain their established meanings. In
particular, the parity factor represents a physical shift by `L/2` and does
not change when `L` changes.

For `L=1`, the P0-03 correction must preserve the pre-correction deterministic
sampler output. For `L != 1`, mode-wise amplitudes under identical random draws
must scale by \(\sqrt{\lambda_m(L_2)/\lambda_m(L_1)}\).

## Finite-information boundary

The input to every surrogate system must be constructed only from the public
`n_tar` field:

```text
reference input at n_ref
  -> spectral restriction to n_tar
  -> encoding to n_sur
  -> surrogate feature generation
```

The encoding and surrogate computation must never inspect the original
`n_ref` tensor or recover high-wavenumber modes discarded by the `n_tar`
interface. Two reference inputs that induce the same finite `n_tar` input must
induce the same surrogate initial state.

The dataset `n_ref` target is reserved for reference-field evaluation and
quadrature. Training targets are the configured `q` coefficients derived from
the finite `n_tar` target. A target-reference convergence claim is valid only
when the dataset carries a passing `validated_reference` proof; the same grid
role in a `foundation_only` dataset does not imply such a claim.

## Matched Model 1 pipeline consistency

Every current validation artifact carries a profile-independent synthetic
foundation check for the fixed Model 1 pipeline. Each positive case executes

```text
synthetic reference field
  -> spectral restriction to n_tar
  -> build_feature_initial_state(finite field, n_sur)
  -> registered surrogate evolution
  -> J equispaced observations
  -> fixed Fourier decoder to q coefficients
```

The expected coefficients come from a separate registered target evolution
whose input is the same finite `n_tar` field, followed by
`real_fourier_analysis(..., q)`. Target and surrogate evolution outputs must
be distinct tensors produced by distinct calls; equality of their conditions
does not authorize reusing one solve. The synthetic reference tensor is never
passed to the feature-state constructor.

Exact-recovery cases select `q <= observable_q(J)` locally. This is a
diagnostic case condition only and does not add a general `q <= J` constraint.
Requests above the fixed decoder's observable band remain valid and retain the
separate structural zero-fill characterization.

The check covers exact heat flow on odd/even equal grids, heat flow with
different finite and surrogate resolutions, split-step Burgers on odd/even
grids, a small ETDRK4 Burgers case, and reaction-diffusion. A paired-reference
heat case adds a discarded high mode and verifies equality of the finite
field, encoded feature input, target coefficients, and prediction within the
algebraic tolerance. A deliberately mismatched heat readout time is a negative
control; detecting a difference is its passing outcome.

Pipeline consistency is assessed in coefficient space and between the two
`q`-projected fields. The separately reported representation floor compares
the target's `q` projection with its full finite evolved field. Nonlinear
evolution can therefore have a nonzero representation floor while the
pipeline check passes exactly.

## Validation binding and provenance

Content-addressed provenance proves which bytes and upstream identities were
used. It does not, by itself, prove that a requested target-reference
condition was numerically convergence validated. Every `pol-dataset-v3`
configuration must therefore select exactly one binding:

- `validated_reference` binds the dataset target to the certificate's target
  system kind, invariant PDE parameters, evolution time, dtype, and domain by
  canonical exact equality. The dataset `reference_nx` must be an actual
  candidate at or after the selected reference candidate. Its canonical
  numerical condition must likewise be an exact member of the validated
  condition suffix. For the exact spectral heat flow this condition is only
  `{"solver":"spectral_exact"}`; no `dt` or temporal-refinement candidate is
  manufactured. A Burgers condition is the canonical tuple
  `solver`, `requested_outer_dt`, `requested_fine_dt`,
  `outer_step_count`, `effective_substep`, `substeps_per_outer`, and
  `dealias`. Dataset `dt`/`fine_dt` values are canonicalized into that complete
  tuple before exact suffix membership is tested. A reaction-diffusion
  condition is exactly `solver`, `dt`, and `nonlinear_filter`; `nu`, `alpha`,
  and `beta` remain invariant PDE parameters and are checked separately.
- `foundation_only` binds only to the passing general foundation checks,
  sample/split/seed and initial-condition contract, domain, dtype, and master
  initial-condition archive identity and capacity. It requires a nonempty
  reason and fixes `target_reference_validation_status` to `not_claimed`.

Candidate order and the selected indices are part of the certificate. The
allowed relation is exact membership in the recorded suffix, not an inequality
rule: a larger unlisted resolution or a smaller unlisted time step is not
accepted. A foundation-only proof cannot be upgraded, aliased, or used as a
fallback for `validated_reference`.

For Burgers, candidate order has the following formal meaning. Reference
resolutions are unique and strictly increasing. Time candidates use one
canonical solver family and one dealias policy, and every requested outer step
must divide the common final time within the solver alignment tolerance.
For split-step,

```text
substeps_per_outer = ceil(requested_outer_dt / requested_fine_dt)
effective_substep = requested_outer_dt / substeps_per_outer
```

An adjacent coarse-to-fine pair must have nonincreasing requested outer step
and nonincreasing effective substep, with at least one strict decrease.
Different requested `fine_dt` values that produce the same actual condition
are therefore not a refinement. For ETDRK4, `requested_fine_dt` is null,
`effective_substep=requested_outer_dt`, `substeps_per_outer=1`, and adjacent
requested steps are strictly decreasing. Solver aliases are canonicalized
before comparison; switching canonical families or dealias policy inside a
sequence is invalid. These rules validate convergence within one solver
family only and make no split-step-versus-ETDRK4 accuracy claim.

For reaction-diffusion, every candidate uses
`semi_implicit_spectral_euler`, every `dt` aligns with the common final time,
and `dt` is unique and strictly decreasing. The entire sequence uses one
fixed `nonlinear_filter`. In particular, `none` and `two_thirds` are distinct
method conditions and switching between them is never a refinement relation.
The reference evolution's canonical `solver`/`dt`/`nonlinear_filter`
condition must exactly equal the finest candidate.

Spatial comparisons evolve spectral restrictions of one finite master field
with the finest canonical time condition. Temporal comparisons use the finest
reference resolution. The joint comparison independently evolves the
selected `(n_ref, time condition)` and compares it with the finest
`(n_ref, time condition)`; spatial and temporal pass statuses do not imply a
joint pass. Each comparison records its explicit common periodic grid, three
finite relative-L2 metrics, complete coarse/fine canonical conditions, and a
row hash. The selected candidate is the first candidate whose entire adjacent
pair suffix passes, with the finest pair required to pass. Certificate loading
reconstructs that selection and both allowed exact suffixes from the ordered
candidates and hashed rows.

An enabled Burgers cross-solver diagnostic is supporting evidence, not a
target-reference selection rule. It independently validates one split-step
candidate sequence and one ETDRK4 candidate sequence on the finest configured
reference grid, reusing the same actual-step refinement proof and the same
non-test calibration initial tensor for both. Each family's finest adjacent
pair must pass before the two finest solutions are compared. The field-space
relative discrepancy is samplewise

```text
2 * ||u_split - u_etd||_L2
--------------------------------
||u_split||_L2 + ||u_etd||_L2
```

and the low-mode discrepancy uses the same two-norm-over-sum definition for
the two real-Fourier coefficient vectors. Mean/max absolute L2, mean/max
symmetric relative L2, and mean symmetric low-mode relative L2 are stored.
Neither solver is labeled as ground truth. The diagnostic block records both
canonical candidate lists, refinement proofs, long-form self-convergence rows
and hashes, runtime step metadata, finest conditions, common grid, sample IDs,
metric definition, tolerances, and discrepancy evidence hash.

The cross-solver block is separate from the primary target-reference contract.
In particular, an ETDRK4 condition in supporting evidence is never inserted
into a split-step dataset's `allowed_refinement_relation`; dataset binding
continues to use only exact members of the primary selected solver-family
suffix.

Target-reference validation is a strict semantic union within the single
validation execution path. The common foundation and calibration/test
isolation checks are shared. `burgers_convergence` owns its time-refinement
candidates. `heat_analytic` separately establishes the Fourier multiplier
\(\exp[-\nu(2\pi m/L)^2t]\) on constant, sine, cosine, multimode, odd/even,
non-unit-domain, and float32/float64 cases, then validates spatial truncation
over the configured reference-resolution candidates. Heat temporal status is
`analytic_exact`, while its spatial convergence status and selected
reference-resolution suffix are recorded independently.
`reaction_diffusion_convergence` independently characterizes the production
semi-implicit spectral Euler scheme using zero and nonzero constant fields,
the scalar recurrence
`c_next=c+dt*alpha*c-dt*beta*c^3`, applicable equilibria
`+-sqrt(alpha/beta)`, and the `beta=0` one-step Fourier multiplier
`(1+dt*alpha)/(1+dt*nu*(2*pi*m/L)^2)`. Expected values are constructed by
scalar/tensor algebra rather than by a second call to the production spatial
solver. It then uses the same generic spatial, temporal, coarsest-stable-
suffix, and independently evaluated joint convergence path as Burgers.

Every validation solve is checked for finite output immediately. A NaN/Inf
state is an instability failure, not a reason to relax tolerances. Its
diagnostic is transactionally published as an exact-byte-verified validation
failure artifact.

The binding proof is constructed before any dataset target evolution. The
canonical proof and its hash are part of the dataset identity and are copied
to the resolved specification, metadata, tensor archive, loaded dataset, and
downstream study identity/reference/summary. Verifiers reject disagreement
between these copies.

## Split and freeze protocol

The data splits have disjoint responsibilities:

- **train** fits readout parameters;
- **validation** selects readout hyperparameters, surrogate systems, surrogate
  parameters, and any other candidate;
- **test** is used only once the complete choice and evaluation procedure have
  been frozen.

Numerical-foundation checks and target-reference convergence are part of this
isolation contract. Their explicitly configured calibration sample IDs must be
classified by the same deterministic CPU `torch.randperm` split primitive,
with the same counts and seed, that dataset construction uses. Calibration IDs
may belong to train or validation, but never test. Any test overlap is rejected
before initial-condition generation, PDE solves, or artifact publication; IDs
must not be chosen or replaced in response to solver results.

The validation certificate records the calibration IDs, their train/validation
membership, zero test-overlap count, split policy/version, and the canonical
split hash. Certificate loading reconstructs that payload. Dataset binding
recomputes the same split contract and hash before target evolution, and
dataset loading verifies that the stored split tensors are exactly those
produced by the shared primitive.

Before requesting a test feature state or computing any test metric, the
runner must:

1. complete study-level candidate and readout-hyperparameter selection using
   train/validation data;
2. materialize evaluation-seed members only for each selected
   random-feature case/readout, again using train/validation data, without
   feeding their audit metrics back into selection;
3. complete required convergence checks;
4. write and hash the selection record;
5. write and hash the frozen model archive and frozen evaluation plan;
6. read all frozen-selection artifacts back and verify their identities and
   exact bytes.

Test data must not affect convergence checks, tie breaking, candidate
selection, rerun decisions, or the contents of the frozen plan.

For a selected direct decoder, the validation-time bandwidth diagnostic must
be stored in the inner selection record, frozen model, and frozen evaluation
plan. After frozen-model read-back, it must be recomputed from the frozen
trial's `J` and `q` and compared with every stored copy before test evaluation
starts. The canonical diagnostic must be repeated in the Model 1 validation
and test rows; learned-readout rows must not carry fabricated values in those
fields.

## Error spaces

Field-space error and data-space error are different scientific quantities and
must be stored separately.

- For endpoint-free values `v_j` on `n` uniform nodes of `[0,L)`, the
  implemented periodic trapezoidal norm is
  `sqrt((L/n) * sum_j v_j^2)`. The real Fourier synthesis basis is
  `L2`-orthonormal, so a resolved trigonometric polynomial with coefficient
  vector `a` satisfies `||v||_L2^2=sum_i a_i^2`.
- Unqualified `field_*` metrics compare a `q`-coefficient reconstruction with
  the dataset target on `n_ref`, using that grid for periodic field quadrature.
  The separate dataset binding says whether convergence validation of that
  target-reference condition is claimed.
- `data_field_*` metrics compare the same prediction with the finite target on
  `n_tar`.

Neither metric may silently replace the other. Representation-floor metrics
must retain the same distinction.

Samplewise relative field error divides the absolute periodic norm by the
target periodic norm clamped below by the target dtype's machine epsilon.
Thus a zero target and zero error produces zero, while a nonzero prediction
against a zero target produces `absolute_error / eps`; this established
epsilon-clamp policy is not an exact scale-invariant relative error at zero.

Every current validation artifact carries a separate, profile-independent
field-quadrature foundation check. It holds one continuous target, one
coefficient prediction, and one finite `n_tar=16` target fixed while changing
only the reference quadrature grid over the strictly increasing candidates
`[8,15,16,31,32]`. The target's maximum mode is seven. Grid 8 intentionally
under-resolves the squared error and representation-floor integrands; grids
15 and above resolve them. Continuous target, error, and representation-floor
norms are calculated directly by Fourier orthogonality rather than by
declaring the finest numerical grid to be truth.

The selected reference quadrature is the coarsest candidate whose complete
suffix agrees with both analytic absolute and relative norms, with the finest
candidate pair required to agree. The exact allowed suffix, row order, row
hashes, tolerances, field-wrapper agreement, fixed-data invariance, and
reference/data representation-floor behavior are certificate-bound.

## Random-feature model realizations

Each random-feature seed for Model 3 is an independent model realization. The
primary result must aggregate **per-seed metrics**, reporting at least their
mean, standard deviation, and a stated confidence interval. The interval
method and the number of seeds must be recorded.

The inner structural search reuses a seeded random map and its lifted
train/validation tensors across ridge candidates. Candidate evaluation stores
a selection recipe, not a set of evaluation-seed models. Evaluation members
are fitted only after study-level selection and only for the selected
case/readout. The pure workload plan reports selection maps, lifts, ridge
fits, selected-only evaluation fits, and the legacy eager comparison
separately; unresolved upstream selection conditions produce null counts
rather than invented estimates.

At least two distinct evaluation seeds are required. For seed metrics
\(m_1,\ldots,m_S\), the implemented primary summary is the arithmetic mean,
the Bessel-corrected sample standard deviation
\(\sqrt{\sum_s(m_s-\bar m)^2/(S-1)}\), and the two-sided 95% Student-t interval
\(\bar m \mathbin{\pm} t_{0.975,S-1}s/\sqrt{S}\). Interval endpoints are not
clamped.

The same per-seed table also records the median and 0.25/0.75 quantiles using
linear interpolation. These are descriptive summaries of the realized seed
distribution, not uncertainty intervals for the mean. Each evaluation
realization binds a content hash of its seed, random-map tensors, activation,
and scales, plus a content hash of that map and its fitted affine readout.
Evaluation-seed validation metrics may be recorded after the structural
hyperparameters have been selected, but they are marked as not used for
selection and cannot change width, scales, bias, or ridge coefficient.

Averaging predictions over seeds defines an ensemble model. Ensemble metrics
may be reported as a separate result, but they must not be labeled or used as
the primary independent-realization result. Its row binds the ordered member
count, seed hash, and complete frozen-member hash.

The `random_feature_seed_statistics` study uses the same trial/freeze/test
path as other studies. It does not introduce a Model-3-specific runner.
Per-seed random-map hashes join test metrics to learned-map norms and
readout-design covariance conditioning in `readout_stability_models.csv`.
Seed scatter has no connecting line; box and empirical-CDF views also read
only verified completed result tables. The checked-in main profile declares
32 evaluation seeds but has not been executed.

## Prediction capture and read-only reporting

Representative prediction samples and random-feature members are scientific
coordinates, not post-hoc display choices. A prediction-capture specification
therefore lists test sample IDs, readout IDs, and explicit evaluation seeds
before selection and test access. Runtime preflight checks test membership
before selection begins. Best, worst, typical, median-performing, or otherwise
test-selected samples/members are not supported capture kinds. A prediction
ensemble is captured only under its separate `prediction_ensemble` semantics.

The capture is built after selection/frozen-plan/archive persistence and
read-back, from the `FrozenPredictions` objects already produced by the
canonical test pass. It does not invoke a second solver or readout inference.
For predeclared samples it stores the finite `n_tar` input, target fields on
`n_tar` and `n_ref`, target and predicted `q` coefficients, synthesized
predictions on both grids, and coefficient errors. Every entry binds the
dataset/split, selection record, frozen plan/archive, feature condition, model
key, readout semantics, and, for a member, its explicit seed and member hash.

For spectra, storage uses full-test per-coefficient aggregates rather than all
full prediction fields. Mode zero is the DC coefficient. Each positive mode
combines its cosine and sine squared errors and target energies. The sample
aggregate is the arithmetic mean over the unchanged canonical test split;
relative energy error divides those two aggregates with a dtype-machine-
epsilon denominator clamp. Mode indices and physical angular wavenumbers
`2*pi*m/L` are both stored. Completed-run verification checks content hashes,
tensor reconstruction/error identities, spectrum aggregation, frozen
coordinates, and agreement with canonical deterministic/member/ensemble
coefficient-MSE rows.

Publication is two-stage. The numerical run is transactionally published and
verified with no figures. Optional reporting then copies that verified run
into a separate transaction, reads CSV/tensor artifacts, writes figures, and
verifies the updated exact-byte manifest. Automatic reporting and
`--plots-only` call the same primitive. Reporter failure cannot replace or
remove the verified numerical run, and reporter code must never call feature
solvers, readout fitting, or frozen-model inference.

## Cross-run read-only reports

A cross-run report is not a study and does not enter unified study execution.
Its strict `pol-report-v1` declaration names at least two source study
specifications. Each source is resolved to its exact content-addressed
completed run, including any completed-selection dependencies, and
`verify_study_run` succeeds before a report transaction or renderer starts.
The report path cannot build validation/dataset artifacts, solve a feature
system, fit a readout, perform test inference, execute an upstream study, or
modify source bytes.

Every phase-diagram declaration fixes the source, split, metric, variant,
readout, two semantic axes, complete axis values, physical labels, and axis
scales. Validation and test rows are never pooled. A numerical NaN/Inf is an
error; it is not encoded as a missing cell. A missing verified row and a
predeclared invalid cell with a recorded reason remain distinct machine-table
statuses. A validation-selected cell is marked only when the declaration
explicitly requests it, and source run/condition hashes accompany the cell
table.

A baseline-table declaration fixes every variant/readout row before reading
test results. Reference-field and finite-`n_tar` data-field metrics and their
representation floors occupy separate columns. Random-feature primary values
must be the independent-member seed mean and retain the Bessel-corrected
standard deviation, Student-t confidence interval, method, level, and seed
count. Prediction-ensemble rows are not eligible for primary columns.
Unrounded machine-readable CSV is written before optional Markdown/LaTeX
formatting.

The report identity includes the semantic report declaration, exact source
run/scientific/manifest and freeze hashes, and reporting software
environment. It excludes storage locations. Reports are staged and
transactionally published with `resolved_report_spec.json`,
`source_references.json`, machine tables, figures/formatted tables,
`report_summary.json`, and an exact-byte `pol-report-manifest-v1`. Failed
rendering preserves any prior verified report.

## Digital neural-operator baseline adapter

The 1D Fourier neural operator is a digital baseline adapter, not a physical
feature readout and not a `StudyRunner` variant. `pol.digital_baselines` owns
its optimization, validation checkpoint selection, frozen checkpoints,
training logs, test evaluation, and transactional publication. The physical
study package does not import this adapter.

The adapter reuses one exact validated `ReferenceDataset`, its canonical
train/validation/test split, the finite-interface constructor, and the same
Fourier coefficient/data-field/reference-field metrics used by physical
studies. Its public interface is

```text
finite input field on n_tar, with no absolute coordinate by default
    -> FNO1d spectral/local layers on n_tar
    -> finite output field on n_tar
    -> q real L2-orthonormal Fourier coefficients
```

The FNO never receives the dataset's `n_ref` input field. The reference target
is used only after prediction for the configured reference-field quadrature;
the finite `n_tar` data-field metric and both representation floors remain
separate. The current strict interface requires odd `q`, `q <= n_tar`, and an
FNO mode count no larger than the finite `n_tar` RFFT width. It introduces no
`n_tar <= J` or `q <= J` condition and does not own `n_sur` or `J`.

Input and coefficient standard-score statistics are fitted from the canonical
train split only. Candidate architecture and epoch checkpoints are selected
with `validation_field_relative_l2_mean`. Configured selection seeds compare
architectures; a disjoint configured evaluation-seed set is trained only
after the architecture choice. Each evaluation seed is an independent neural
network realization with its own validation-selected epoch checkpoint.

The periodic Burgers baseline uses `coordinate_channel=none`: spectral
convolutions and pointwise layers then preserve circular-shift equivariance,
matching a stationary input law and translation-equivariant target operator.
The optional `periodic_sin_cos` policy deliberately appends
`sin(2*pi*x/L), cos(2*pi*x/L)` on the endpoint-free physical grid
`x_j=L*j/n_tar`; it supplies absolute periodic position and therefore
generally changes translation equivariance. The removed `unit_periodic` ramp
is not periodic and is rejected with an explicit migration error.

Before requesting the finite test view or computing a test metric, the adapter
writes and hashes its validation-only selection record, writes the frozen
evaluation-seed checkpoints and evaluation plan, hashes their exact bytes and
tensor contents, and reads all three back. The event log and completed-run
verifier enforce this order. The primary test row aggregates per-seed metrics
using the same Bessel-corrected standard deviation and two-sided 95%
Student-t interval as Model 3. Prediction averaging is a separately labeled
ensemble table.

Every run first verifies the exact source manifest bytes, dataset binding,
validation rows, selection record, frozen plan, and frozen models without
parsing physical test values or executing the source. Only after the digital
selection record and frozen evaluation boundary have been written, hashed,
and read back may the adapter request either digital test tensors or parse the
physical source test table. Before writing the fairness table, it verifies the
full physical source, requires its manifest and test-table hashes to remain
unchanged, and records these preflight and post-freeze hashes in the source
reference and ordered event log. The
predeclared fairness table requires the same dataset, split, `n_tar`, and `q`;
records input/output dimension, model parameter count, validation selection
metric, separate field/data metrics and floors, seed statistics, and the
distinct inference paths. Digital training wall/process time is recorded, but
energy is not measured and physical/digital wall-clock or energy comparison is
explicitly disallowed without a common measurement protocol.

`pol-digital-baseline-v3` configurations and
`pol-digital-baseline-run-manifest-v4` outputs are content addressed with
storage paths excluded from identity, transactionally published, and
exact-byte plus semantic/tensor-hash verified. The checked-in smoke profile is
an execution-path check only. The checked-in main training budget has not been
executed and establishes no FNO performance claim.

Fairness parameter counts use
`pol-real-scalar-parameter-count-v1` and always describe one independent model
realization. Direct decoding stores no model parameters. Affine readouts count
their frozen `W,b` tensors as trainable. Random-feature readouts count frozen
`A,c` as fixed random parameters and `W,b` as trainable, including the
skip-connected input columns; the primary per-seed result is not multiplied by
the number of seeds. A separate field reports storage across all frozen
realizations. FNO counts are reconstructed from each frozen state dictionary
and cross-checked against the training outcome; complex entries, if ever
introduced, count as two real scalars. These model-capacity counts exclude the
fixed physical dynamics and do not estimate analog or other hardware
components, wall-clock cost, or energy.

## Readout stability under feature noise

The `readout_stability_noise` diagnostic runs only after the selection record
and frozen evaluation plan have been written, hashed, and read back. Every
diagnostic row binds the selection hash, frozen-plan hash, and frozen archive
model key. Noise level is relative to the global RMS of the clean observed
feature matrix, and the saved coordinates include the resulting RMS, draw
seed, repeat index, and sample shape. Common random-number seeds are reused
across frozen evaluation-seed members at a given level and repeat.

For deterministic readouts, metrics are retained per noise repeat and
summarized with a Bessel-corrected standard deviation and Student-t 95%
interval. For random-feature readouts, repeats are first summarized within
each independently frozen evaluation seed; the primary stability result then
aggregates those per-seed means across seeds. Prediction averaging is stored
only in separately labeled ensemble tables. Fixed direct decoders report
readout norms as not applicable. Learned affine and random-feature members
record `W` Frobenius/operator norms, bias norm, selected ridge coefficient,
and centered covariance singular values, numerical rank, cutoff, raw
condition, and retained-rank condition. A rank-deficient raw condition remains
infinite rather than being replaced by a finite surrogate.

## Nested-prefix learning curves

A learning-curve training condition is
`nested_train_prefix(n_train)` under
`canonical_train_order_prefix_v1`. It selects exactly the first `n_train`
IDs of the existing canonical train order. It neither creates a new dataset
artifact nor changes validation or test membership. Direct configuration of
arbitrary subset IDs is not part of this kind. Sizes must be positive,
increasing, unique, and no larger than the canonical train count.

Every subset record stores its ordered IDs, ID hash, parent-train hash,
validation hash, count, policy/version, and content hash. The record is copied
into the validation result, selection record, frozen plan, frozen model
archive, and test result and is verified across those boundaries. All
configured sizes complete validation-only readout selection and archive
read-back before the first test feature request. Test error is therefore an
evaluation coordinate, never a train-size selection signal.

Feature generation remains independent of training-subset size. For a fixed
dataset and feature condition, the selection feature request uses the full
canonical train IDs followed by the fixed validation IDs; readout fitting
then slices the requested prefix. This makes cache reuse across sizes exact
without erasing the training-subset identity of the fitted model. The direct
decoder must consequently have identical test metrics across sizes and is
not described as trained.

## Analytic heat-readout calibration

The `heat_readout_calibration` study compares under-diffusive, matched, and
more-diffusive heat features at the target readout time while retaining the
same finite `n_tar` interface for direct, affine-ridge, and random-feature
readouts. The checked-in heat dataset is bound to a heat-specific validated
reference. Observation-noise and stability perturbations are not part of this
analytic calibration question and are not emitted by this study.

For real-Fourier coefficient mode `m`, the stored target and surrogate heat
multipliers use the physical wavenumber `2*pi*m/L`. On a mode that remains
identifiable at the equispaced observation interface and whose surrogate
multiplier is above the configured numerical floor, the ideal linear readout
multiplier is evaluated in log space as

```text
exp[-(nu_target*T_target - nu_surrogate*T_surrogate)*(2*pi*m/L)^2].
```

The coefficient convention is explicitly `DC, cos(1), sin(1), ...`. For even
`J`, the observation-grid Nyquist cosine is not treated as an identifiable
sine/cosine pair, and the corresponding sine column is zero on that grid.
Modes above the paired observable band are marked aliased rather than adding
a `q <= J` constraint. A surrogate multiplier at or below the configured
floor, and an unrepresentable ideal ratio, produce explicit status fields and
empty ratio/error cells rather than forced division.

The coefficient table records physical conditions, both heat multipliers,
the ideal multiplier, effective linear diagonal, diagonal errors,
off-diagonal contribution, identification status, and amplification. The
case/readout summary records identifiable mode/coefficient counts, diagonal
RMSE and maximum error, off-diagonal Frobenius norm, maximum ideal
amplification, and selected ridge parameter. Direct and affine readouts have
well-defined effective linear maps. A nonlinear random-feature readout is
stored as explicitly not applicable; it is not assigned a fictitious single
linear multiplier.

More-diffusive features have
`nu_surrogate*T_surrogate > nu_target*T_target`, so their ideal inverse toward
a less-diffusive target amplifies high frequencies. This condition and its
plain-language interpretation are stored in every applicable coefficient row
and summary row. The comparison reporter reads only verified completed-run
tables. It cannot initiate dataset construction, feature solves, or study
execution.

## Surrogate-parameter/readout-time landscape

`surrogate_parameter_time_coordinate_search` and
`surrogate_parameter_time_landscape` answer different questions through the
same unified study executor. The coordinate study alternates the two axes as
an efficient validation-search aid. It is not a two-dimensional phase map.
The landscape study uses the convergence-validated Burgers target dataset and
evaluates the complete, predeclared Cartesian product of surrogate diffusion
parameter and readout time independently for heat, Burgers, and
reaction-diffusion feature generators.

Heat feature evolution is spectral exact. Burgers and reaction-diffusion
feature specifications retain the complete checked solver condition used by
the corresponding validated numerical setup. Parameters that are not swept,
including Burgers advection/dealias settings and reaction-diffusion
`alpha`, `beta`, solver, time step, and nonlinear filter, remain explicit in
the resolved trial and every validation row. This reuse of a checked numerical
condition does not turn a Burgers-target study into a separate validation
certificate for every surrogate parameter value.

Every grid cell is an experimental condition declared before validation or
test access. Cartesian axes require unique values and paths. Config order is
the tie order. The selection artifact records the planned cell count, complete
ordered evaluated-candidate list, per-cell axis values and status, and any
skipped/invalid cell with an explicit reason. Coordinate stages carry
`search_kind=coordinate` and cannot be consumed by the generic phase-map
reporter. Duplicate cells are rejected rather than averaged.

Each readout retains its own validation optimum. Separately, the candidate
selected for `selection.representative_readout` is stored as the
representative feature condition for downstream studies. That record contains
the finite-input specification, full feature condition, output interface,
selection metric/value, and config-order candidate ID. It is copied into the
frozen evaluation plan and checked against the frozen trial after exact
write/hash/read-back, all before requesting any test feature state. Test
metrics do not participate in either selection.

`metric_map` is the publication-independent two-dimensional reporter for the
parameter landscape and the existing `J x q` and `n_tar x n_sur` studies. Its
axes are explicitly declared, numerically ordered, and may contain missing
cells represented as missing values. A validation-selected cell can be
marked. The reporter is validation-only and operates solely on a verified
completed result when invoked through plots-only mode. Observation noise and
stability perturbations remain a Phase 6 responsibility and are absent from
both Phase 3 parameter/time studies.

## Observation/output budget

`observation_output_budget` evaluates the complete declared Cartesian product
of observation count `J` and odd real-Fourier output dimension `q`, while
holding the dataset, split, finite-input `n_tar`, surrogate resolution
`n_sur`, readout candidate sets, and all non-imported feature conditions
fixed. It is distinct from `input_simulation_resolution`; changing either
budget axis must not alter `n_tar` or `n_sur`.

The checked-in smoke and main profiles intentionally contain only the current
Burgers and reaction-diffusion feature variants. Each imports its complete
feature system and evolution time from the matching validation-selected
representative condition in the verified
`surrogate_parameter_time_landscape` profile. Adding heat or another variant
is a separate declared expansion of computational scope, not an implicit
default.

Every valid cell evaluates `direct_fourier_decoder`, `affine_ridge`, and
`random_feature_ridge`. The primary maps use
`validation_field_relative_l2_mean` and are generated separately for every
variant/readout pair. A predeclared cell's test result is a budget-axis
evaluation only; it is not used to select a budget cell, feature system,
ridge parameter, or random-feature setting. Readout hyperparameters remain
selected using train/validation data within that cell before the common
freeze/read-back boundary.

Cells with `q > J` are valid. The direct decoder records its observable
bandwidth, retained prefix, and exact zero-filled coefficient/mode counts.
That structural limitation is not applied to either learned readout. Invalid
cells and unevaluated missing cells remain distinct in map data: invalid
cells carry their recorded reason, while a missing cell states that no
verified validation row exists. A non-finite metric is rejected and is never
eligible as a best value.

Every validation and test result row records `J`, `q`, `n_tar`, `n_sur`, the
feature-system condition and hash, completed-selection source hashes, readout
kind, selected learned-readout settings, reference- and data-space
representation floors, and its split-specific metrics. Direct rows alone
carry the fixed-decoder bandwidth fields. Random-feature primary test rows
remain independent-seed summaries with Student-t intervals; prediction
ensembles remain separately labeled rows and tables. The completed summary
separately records declared global-axis combinations, planned
variant-by-combination cases, evaluated cases, and skipped cases.

## Input/simulation resolution

`input_simulation_resolution` evaluates the complete declared Cartesian
product of finite-input resolution `n_tar` and surrogate simulation resolution
`n_sur`, while holding observation count `J`, odd output dimension `q`, the
dataset, split, readout candidate sets, and all non-imported feature conditions
fixed. It has its own JSON, scientific identity, result tables, and completed
run; it is not an alias or execution mode of `observation_output_budget`.

Every cell follows the same finite-information path: the validated `n_ref`
reference input is spectrally restricted to that cell's `n_tar`, and only
that finite field is encoded at the cell's `n_sur` before surrogate evolution,
fixed-`J` observation, and fixed-`q` output. Increasing `n_sur` therefore
cannot restore a reference mode discarded at `n_tar`. No ordering constraint
is imposed between `n_tar` and `n_sur`; valid cells require only
`q <= n_tar` and `J <= n_sur`.

The checked-in smoke and main profiles contain exactly the current Burgers and
reaction-diffusion variants. Both import the complete system and evolution
time from the matching validation-selected representative condition in the
verified Phase 3 landscape. Every cell evaluates the direct, affine, and
random-feature readouts. The six primary maps are validation maps, one for
each variant/readout pair. Per-cell test values evaluate predeclared
resolution axes only and do not select a resolution, system, ridge, or
random-feature setting.

The smoke grid is `n_tar in {9, 16}` by `n_sur in {9, 16}`, with fixed
`J=8` and `q=9`. It exercises odd and even resolutions and all three
relationships `n_tar < n_sur`, `n_tar = n_sur`, and `n_tar > n_sur`. The main
candidate is the explicit `4 x 4` grid
`{64, 128, 256, 512} x {64, 128, 256, 512}` with fixed `J=64`, `q=33`,
two variants, and three readouts: 32 variant-expanded cells and 96 readout
evaluations before inner hyperparameter/seed work. This describes configured
cost only; no main resolution result has been executed or verified here.

Each result row stores both reference-field metrics and `n_tar` data-field
metrics, with their representation floors kept separate, as well as all five
dimensions, feature-system/source hashes, readout kind, and selected readout
settings. Feature-state cache identity includes the dataset artifact, ordered
sample IDs, `n_tar`, `n_sur`, and the complete selected feature dynamics.
Invalid skipped cells retain a reason; unevaluated map cells remain missing.
A non-finite numerical result aborts publication and is not converted to
either a missing cell or a best value.

## Completed-study selection binding

A downstream study may bind a variant's feature evolution to a representative
condition selected by a verified completed study. The declaration names a
source study specification, source variant, representative readout, and a
nonempty allowlisted set of semantic imports. The current allowlist is exactly
`feature.evolution.system` and `feature.evolution.time`; it cannot overwrite
finite-input, feature-resolution, observation, output, readout, dataset, or
split specifications. An imported path cannot also be a downstream search or
global-sweep axis.

Resolution is read-only. It computes the expected content-addressed source run
from the source scientific specification and its existing verified dataset,
requires that exact completed run, and calls the completed-run verifier. It
then cross-checks the source selection record, frozen evaluation plan, frozen
model archive, representative case/readout/candidate, validation metric, and
frozen trial. The selected condition must be explicitly validation-selected,
must carry no test binding, and is extracted only from those frozen selection
artifacts. Source test tables and test metrics are not inputs to condition
resolution.

The downstream scientific identity and resolved study contain no source
storage path. They contain the source run hash, source study scientific
identity hash, selection-record hash, frozen-plan hash, frozen-model archive
hash, source dataset/split identities, case/variant/readout/candidate IDs,
import paths and resolved values, and the selection metric/value. The same
provenance is cross-bound into the downstream selection record, frozen plan,
and frozen model archive before downstream test access.

Source and downstream profiles and dataset/split identities must agree.
Self-dependencies, dependency cycles, missing or tampered source runs,
unresolved legacy schemas, and source/downstream profile or data mismatches
are rejected before downstream feature or test work. A pure plan reports a
missing dependency without manufacturing a selected value. The read-only
`pol selection inspect` and `pol selection verify` commands inspect source
selections and downstream bindings without starting validation, dataset
construction, or study execution.

## Execution and reporting

- Scientific JSON and CLI override parsing reject `NaN`, `Infinity`, and
  `-Infinity` before model construction. Strict models reject non-finite
  floats, including values nested in sweep axes, overrides, source imports,
  optimizer settings, tolerances, and reporter coordinates. Unknown keys
  remain forbidden.
- The validated first-paper workflow is CPU-only. Its public validation schema
  accepts only `samples.device="cpu"` (the default) and rejects `"cuda"`,
  `"auto"`, and unknown values before numerical work begins.
- The CPU-only scope covers validation and reference convergence, master
  initial-condition publication, dataset target batches and serialized/loaded
  tensors, feature-state publication/load, readout fitting inputs, frozen
  models, test evaluation, diagnostics, and completed study runs.
- Every artifact family in that path records
  `execution_device_policy="cpu_only"` and `compute_device="cpu"` in its
  content-addressed provenance. A PyTorch build's `torch_cuda_version` is
  environment metadata only and is not evidence that CUDA computation
  occurred.
- A non-CPU tensor reaching an official workflow boundary is a contract
  violation and must be rejected with the boundary identified. Existing
  publication-time `detach().cpu()` operations define serialized CPU bytes;
  they must not conceal a non-CPU upstream solve.
- Low-level numerical kernels may preserve an input tensor's device. That
  algebraic generality is outside the validated artifact guarantee and does
  not constitute end-to-end GPU support.
- Main profiles and production-resolution studies must not be run by automated
  tests or during Codex maintenance work. Tests and maintenance validation use
  tiny or smoke profiles only.
- Figure generation consumes a verified, completed result. A request to render
  or regenerate a figure must not implicitly start a numerical experiment.
- Project-owned directory components such as study/report/digital-baseline
  names and profiles use the explicit ASCII policy
  `[A-Za-z0-9][A-Za-z0-9_-]*`. They are nonempty basenames, never `.` or
  `..`, never dot-prefixed, and contain no separator, surrounding whitespace,
  or control character. User-selected storage roots remain ordinary paths and
  are not constrained by this component policy.
- Study and cross-run reporter filenames use the same policy and are declared
  without an extension. Reporter formats are nonempty and unique. A configured
  study reporter completes only after producing exactly one file for every
  declared format; zero, duplicate, missing, or unexpected outputs fail the
  reporting transaction. No current reporter has an allow-empty policy.
- A completed study summary records the configured reporter count, exact
  expected and generated figure counts, expected filenames, and completion
  policy. The completed-run verifier binds those values to the resolved study,
  manifest file tree, and exact artifact bytes.
- Publication labels `E0` through `E7` and Figure numbers may appear only in a
  documentation-level correspondence table. They must not name core packages,
  modules, runners, dispatch branches, schemas, or artifact kinds.
- Implementation dispatch uses semantic kinds such as `heat`,
  `equispaced_points`, and `affine_ridge`.

## Change control

Changes to solvers, metrics, information boundaries, split usage, selection,
freezing, or artifact identity are scientific changes. They require focused
tests and an explicit statement of their numerical meaning. Refactors must
preserve numerical meaning unless a later phase explicitly authorizes a
scientific correction.
