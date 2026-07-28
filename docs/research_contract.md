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
