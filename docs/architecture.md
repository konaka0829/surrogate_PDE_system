# Architecture

## 1. Dependency direction

The repository uses a one-way dependency graph:

```text
math + numerics
      ↓
systems
      ↓
validation ──→ validated initial-condition certificate
      ↓
data       ──→ content-addressed target dataset
  ├──→ learning + study ──→ plotting/reporting
  └──→ digital_baselines ──→ independent neural-operator run
```

`validation` and `study` are siblings in purpose: validation establishes that
numerical and algebraic assumptions are trustworthy; a study performs
operator-learning selection and evaluation. Dataset construction does not
import a numbered validation experiment, and the prediction path does not
contain validation-specific branches.

The digital baseline consumes validated data and shared learning metrics but
is not a `StudyRunner` readout. `pol.study` does not import
`pol.digital_baselines`; the adapter may read an already verified physical
study only through the read-only completed-run boundary.

Validation keeps one runner and one CLI path. `ValidationSpec.target_reference`
is a strict semantic union: common foundation and calibration isolation are
run once, while `burgers_convergence`, `heat_analytic`, and
`reaction_diffusion_convergence` own only their PDE-specific reference checks.
`pol.validation.model1_consistency` owns the separate profile-independent
finite-input-to-fixed-decoder pipeline suite and has no filesystem or study
dependency. `pol.validation.quadrature` similarly owns the pure analytic
periodic-`L2` and reference-grid quadrature suite. It holds the continuous
prediction, target, and finite `n_tar` data fixed while varying only `n_ref`;
the validation runner serializes its result but does not reimplement its
selection rule. `pol.validation.conditions` canonicalizes system solver
conditions and validates the system-agnostic target-reference contract.
`pol.validation.binding` consumes that contract before dataset target
evolution; it does not infer a safe condition from solver aliases, larger
unlisted grids, or smaller unlisted time steps.

## 2. Ownership of dimensions

| Symbol | Owner | Meaning |
|---|---|---|
| `n_ref` | dataset artifact | numerical reference-grid size |
| `n_tar` | `FiniteInputSpec` | finite target input/output grid |
| `n_sur` | `FeatureGeneratorSpec` | surrogate internal discretization |
| `J` | `PointObservationSpec` | observed feature dimension |
| `q` | `FourierOutputSpec` | target real-Fourier dimension |

The only generic constraints imposed by the current interfaces are
`J <= n_sur` and `q <= n_tar`, together with odd `q`. In particular, `n_tar`
and `J` are independent.

## 3. Information boundary

For every sample, the feature state is constructed as:

```text
reference initial field
  -> spectral restriction to n_tar
  -> spectral encoding from that finite field to n_sur
  -> configured feature transform (surrogate evolution or static baseline)
```

The encoding operation never receives the original reference field. Two
reference fields that differ only in modes discarded at `n_tar` therefore
produce identical feature-generator inputs. Both validation and tests exercise
this property. `pde_dynamics` evolves that state with a registered system;
`static_input` exposes the same finite-information boundary without evolution
as a baseline, using the same cache, observation, readout, and evaluation path.

The target side retains both representations derived from the same sample ID:

```text
validation-bound dataset target at n_ref
  ├─> finite target data at n_tar
  └─> reference field used only for quadrature/evaluation
```

Readouts are fitted to the `q` coefficients extracted from finite target data.
The principal `field_*` metric reconstructs those coefficients on `n_ref` and
compares with the dataset reference target. A separate `data_field_*` metric is
reported on `n_tar`. The dataset's explicit binding proof separately records
whether this target-reference condition is validated or merely uses a checked
foundation. Consequently, a resolution sweep does not redefine the metric
merely by changing its finite target grid.

## 4. Split ownership and calibration isolation

`pol.data.splits` owns the deterministic train/validation/test partition. It
uses a CPU `torch.Generator` and the established `torch.randperm` order, checks
pairwise disjoint full coverage, and constructs the canonical split payload
and hash. Dataset construction and validation calibration both call this
primitive; neither carries a private permutation algorithm.

Validation calibration IDs remain explicit configuration. Before generating
initial conditions or running a PDE, validation classifies them against the
shared split and rejects any test member. The calibration membership, zero
overlap, split policy/version, counts, seed, and split hash are bound into the
validation identity/foundation certificate and copied into the dataset binding
proof. Certificate, proof, and dataset loading independently reconstruct and
verify these values.

## 5. Unified execution

The study execution path has a one-way internal dependency graph:

```text
config + learning + data
          ↓
study.evaluation ← study.readouts
          ↓             ↓
             study.trial
                  ↓
              study.search

study.cases ───────────────┐
study.trial + study.search ├─> study.runner
study.protocol ────────────┤
study.results ─────────────┤
study.verification ────────┘
```

`pol.study.evaluation` owns metric wrappers, representation-floor evaluation,
independent-seed statistics, immutable evaluation results, and pure row
construction. `pol.study.readouts` owns validation-time readout fitting,
readout-local hyperparameter selection, selected random-feature recipe
materialization, frozen payload construction, and frozen prediction. During
random-feature selection it reuses each seeded random map and its lifted
train/validation tensors across ridge candidates. Evaluation-seed members are
not fitted for rejected candidates. Neither module owns dataset splits,
feature-state solves, filesystem transactions, or test-access timing.

`pol.study.trial.TrialEngine` coordinates one validated trial: it obtains the
finite `n_tar` view, requests cached `n_sur` feature states, observes `J`
features, separates train and validation tensors, delegates readout work, and
requests test features only through its test-evaluation entry point.

`pol.study.cases` owns filesystem-free scalar/sweep expansion and planning.
`pol.study.protocol` owns selection records, frozen archives and plans, their
cross-hashes, and the persisted read-back boundary that authorizes test
evaluation. `pol.study.results` owns stable table fields, result/summary
serialization, manifest construction, and reporter inputs.
`pol.study.verification` is the read-only completed-run verifier: it checks
exact bytes, cross-artifact bindings, seed summaries, decoder diagnostics, and
event ordering. These modules do not import `pol.study.runner`.

`pol.study.runner` is the public orchestration façade. It prepares the dataset,
expands cases, invokes trial/search and convergence work, completes the
selection/freeze/read-back protocol, then starts test evaluation and
transactionally publishes results. The sequencing remains visible in one
execution path; planning, verification, and plots-only reuse do not create
alternate experiment runners.

`StudyRunner` expands:

```text
base_trial × global_axes × variants × variant_search_candidates
```

into validated trial specifications. A scalar run has one variant, no global
axis, and a static search. No alternate scalar or matrix runner exists.

Each selection candidate is evaluated on train/validation data. Readout-local
hyperparameters—ridge values and random-feature settings—are also selected on
validation data. A selected random-feature candidate initially carries a
recipe rather than evaluation-seed members. Once study-level selection is
complete, the runner materializes those members from train/validation data
only for the selected case/readout, runs convergence checks, and writes:

1. `selection_record.json`;
2. `frozen_models.pt`;
3. `frozen_evaluation_plan.json`.

It hashes and reads the latter two files back. Only then does it request test
feature states or compute test metrics. `events.json` makes this ordering
auditable, including the selected-only evaluation-member materialization
event. The selection record states that evaluation-member validation metrics
were not used for selection.

## 6. Content addressing

Scientific identity dictionaries exclude storage locations. SHA-256 hashes of
canonical JSON determine artifact and study identities. Artifacts contain a
manifest with every expected regular file, byte size, and SHA-256 digest.

Publication uses a sibling staging directory. The staging tree is validated
before an atomic directory replacement. If publication fails, a previously
valid destination is restored.

The digital adapter follows the same transaction rule but owns separate
strict configuration, identity, selection, checkpoint, frozen-plan, summary,
and manifest schemas. Its selection/checkpoint/plan records are read back
before its sole test finite-view boundary.

## 7. Extension points

### New evolution system

1. Add a strict discriminated system spec in `pol/config/models.py`.
2. Implement the solver under `pol/systems/`.
3. Register dispatch in `pol/systems/registry.py`.
4. Add numerical and integration tests.

The study runner requires no system-name branch.

### New readout

1. Add a discriminated readout spec.
2. Implement fitting, serialization, and frozen prediction in
   `pol/study/readouts.py`.
3. Define serializable frozen parameters.
4. Add selection and read-back tests.

### New experimental question

Prefer a new JSON study combining existing components. Add core Python only
when the question needs a genuinely new system, observation, readout,
diagnostic, or reporter. Do not create a new runner named after a figure or an
experiment number.

For dimensional studies, keep the scientific questions separated:

- `J × q` varies observation and output budgets at fixed `n_tar, n_sur`;
- `n_tar × n_sur` varies finite-data and surrogate discretization at fixed
  `J, q`.

### New digital neural-operator family

Add it under `pol.digital_baselines` only when its optimization/checkpoint
lifecycle differs materially from fixed physical features. Reuse the
validated dataset, canonical split, finite-input constructor, and shared
metrics. Do not register it as a physical readout and do not add a dependency
from `pol.study` back to the digital adapter.
