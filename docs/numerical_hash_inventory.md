# Numerical hash inventory

This inventory classifies fixed hashes in `tests/` separately from runtime
artifact verification. It applies to package `0.2.30`.

## Removed backend-numerical goldens

| Former test payload | Classification | Portable replacement |
|---|---|---|
| GRF `values` and `fourier` tensor SHA-256 | backend-dependent FFT bytes | seeded coefficient reference, odd/even and float32/float64 `irfft`/`rfft` comparisons, explicit DC/domain/Nyquist checks |
| complete random-feature validation/model/test stable-hash snapshot | backend-dependent SVD/solve payload | canonical candidate identity plus direct structure, shape, seed, zero-data metric, tensor-content, independent-seed, ensemble, and same-runtime eager/lazy assertions |

These byte constants are intentionally absent. Updating them to bytes from a
new workstation would recreate the portability defect.

## Retained hard-coded 64-hex values

| Token | Location and multiplicity | Classification | Reason |
|---|---|---|---|
| `d693c22500c07511a76bfb36f5b8227616c87692c8f5448be32b3538412ffc99` | `tests/test_random_feature_evaluation.py`, once | `mock_fixture_split_hash` | Inert fake dataset metadata; no numerical tensor is hashed or compared. |
| `80e69dd30b1caa4acae41729789c90449c8749292a6ceb85680949656dd503e1` | `tests/test_validation_data.py`, three times | `scientific_split_hash` | Exact regression for the declared `cpu_torch_randperm` policy, seed, ordered train/validation/test IDs, and sample-ID binding. |

The split constant is deliberately retained because split membership is a
scientific condition, not a numerical solver result. The current policy is
`cpu_torch_randperm`, version 1. Validation/study identities also contain the
numerical-environment fingerprint, including the Torch version. Therefore an
environment whose permutation policy produces different IDs cannot silently
reuse an existing content identity. The regression is intended to fail and
require an explicit split-policy/version migration in that situation; it is
not a claim that all future Torch releases generate the same permutation.

## Exact hashes that are not fixed numerical goldens

Tests continue to recompute and verify canonical JSON/object identity hashes,
selection and frozen-plan cross-hashes, manifest/file hashes, checkpoint
state hashes, dataset binding hashes, and deliberate tamper values such as
`"0" * 64`. Those checks bind serialized content or detect mutation and must
not be weakened. Their expected value is computed from the current payload
rather than asserted as one backend's FFT/SVD/solve byte sequence.
