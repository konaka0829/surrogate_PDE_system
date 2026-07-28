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
      ↓
learning + study
      ↓
plotting and publication-level study specifications
```

`validation` and `study` are siblings in purpose: validation establishes that
numerical and algebraic assumptions are trustworthy; a study performs
operator-learning selection and evaluation. Dataset construction does not
import a numbered validation experiment, and the prediction path does not
contain validation-specific branches.

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
validated target at n_ref
  ├─> finite target data at n_tar
  └─> reference field used only for quadrature/evaluation
```

Readouts are fitted to the `q` coefficients extracted from finite target data.
The principal `field_*` metric reconstructs those coefficients on `n_ref` and
compares with the validated reference target. A separate `data_field_*` metric
is reported on `n_tar`. Consequently, a resolution sweep does not redefine the
metric merely by changing its finite target grid.

## 4. Unified execution

`StudyRunner` expands:

```text
base_trial × global_axes × variants × variant_search_candidates
```

into validated trial specifications. A scalar run has one variant, no global
axis, and a static search. No alternate scalar or matrix runner exists.

Each selection candidate is evaluated on train/validation data. Readout-local
hyperparameters—ridge values and random-feature settings—are also selected on
validation data. After convergence checks, the runner writes:

1. `selection_record.json`;
2. `frozen_models.pt`;
3. `frozen_evaluation_plan.json`.

It hashes and reads the latter two files back. Only then does it request test
feature states or compute test metrics. `events.json` makes this ordering
auditable.

## 5. Content addressing

Scientific identity dictionaries exclude storage locations. SHA-256 hashes of
canonical JSON determine artifact and study identities. Artifacts contain a
manifest with every expected regular file, byte size, and SHA-256 digest.

Publication uses a sibling staging directory. The staging tree is validated
before an atomic directory replacement. If publication fails, a previously
valid destination is restored.

## 6. Extension points

### New evolution system

1. Add a strict discriminated system spec in `pol/config/models.py`.
2. Implement the solver under `pol/systems/`.
3. Register dispatch in `pol/systems/registry.py`.
4. Add numerical and integration tests.

The study runner requires no system-name branch.

### New readout

1. Add a discriminated readout spec.
2. Implement fitting and frozen prediction in `pol/study/trial.py` or a focused
   module if the implementation is large.
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
