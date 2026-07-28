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

## Validation binding and provenance

Content-addressed provenance proves which bytes and upstream identities were
used. It does not, by itself, prove that a requested target-reference
condition was numerically convergence validated. Every `pol-dataset-v2`
configuration must therefore select exactly one binding:

- `validated_reference` binds the dataset target to the certificate's target
  system kind, invariant PDE parameters, evolution time, dtype, and domain by
  canonical exact equality. The dataset `reference_nx` must be an actual
  candidate at or after the selected reference candidate. Its
  solver/`dt`/`fine_dt`/dealias dictionary must likewise be an actual candidate
  at or after the selected time candidate.
- `foundation_only` binds only to the passing general foundation checks,
  sample/split/seed and initial-condition contract, domain, dtype, and master
  initial-condition archive identity and capacity. It requires a nonempty
  reason and fixes `target_reference_validation_status` to `not_claimed`.

Candidate order and the selected indices are part of the certificate. The
allowed relation is exact membership in the recorded suffix, not an inequality
rule: a larger unlisted resolution or a smaller unlisted time step is not
accepted. A foundation-only proof cannot be upgraded, aliased, or used as a
fallback for `validated_reference`.

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

Before requesting a test feature state or computing any test metric, the
runner must:

1. write the selection record;
2. hash the selection record;
3. write the frozen model archive and frozen evaluation plan;
4. hash those files;
5. read them back and verify their identities and exact bytes.

Test data must not affect convergence checks, tie breaking, candidate
selection, rerun decisions, or the contents of the frozen plan.

## Error spaces

Field-space error and data-space error are different scientific quantities and
must be stored separately.

- Unqualified `field_*` metrics compare a `q`-coefficient reconstruction with
  the dataset target on `n_ref`, using that grid for periodic field quadrature.
  The separate dataset binding says whether convergence validation of that
  target-reference condition is claimed.
- `data_field_*` metrics compare the same prediction with the finite target on
  `n_tar`.

Neither metric may silently replace the other. Representation-floor metrics
must retain the same distinction.

## Random-feature model realizations

Each random-feature seed for Model 3 is an independent model realization. The
primary result must aggregate **per-seed metrics**, reporting at least their
mean, standard deviation, and a stated confidence interval. The interval
method and the number of seeds must be recorded.

At least two distinct evaluation seeds are required. For seed metrics
\(m_1,\ldots,m_S\), the implemented primary summary is the arithmetic mean,
the Bessel-corrected sample standard deviation
\(\sqrt{\sum_s(m_s-\bar m)^2/(S-1)}\), and the two-sided 95% Student-t interval
\(\bar m \mathbin{\pm} t_{0.975,S-1}s/\sqrt{S}\). Interval endpoints are not
clamped.

Averaging predictions over seeds defines an ensemble model. Ensemble metrics
may be reported as a separate result, but they must not be labeled or used as
the primary independent-realization result.

## Execution and reporting

- Main profiles and production-resolution studies must not be run by automated
  tests or during Codex maintenance work. Tests and maintenance validation use
  tiny or smoke profiles only.
- Figure generation consumes a verified, completed result. A request to render
  or regenerate a figure must not implicitly start a numerical experiment.
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
