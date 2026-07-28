# Repository guidance

## Scientific purpose

The package studies PDE solution-operator approximation with a surrogate
evolution system as a dynamic feature generator. The reusable core must be
organized by scientific responsibility, not by publication, figure, or
experiment number.

## Non-negotiable invariants

- Keep `n_ref`, `n_tar`, `n_sur`, `J`, and `q` independent and owned by their
  corresponding dataset, finite-input, feature-generator, observation, and
  output specifications.
- Never impose a general `n_tar <= J` constraint.
- Enforce the relevant current constraints `J <= n_sur`, odd `q`, and
  `q <= n_tar`.
- Build every surrogate input from the finite `n_tar` field. Core computations
  must never recover or inspect reference modes discarded by that interface.
- Use the convergence-validated `n_ref` target only for reference-field
  quadrature/evaluation. Report finite `n_tar` data-space error separately.
- Keep `J × q` and `n_tar × n_sur` phase diagrams as distinct studies.
- Select systems and readout hyperparameters using training/validation data
  only.
- Write, hash, and read back the selection record and frozen evaluation plan
  before requesting test feature states or computing test metrics.
- Treat random-feature seeds as independent model realizations: primary results
  aggregate per-seed metrics with mean, standard deviation, and a stated
  confidence interval. A seed-prediction average is a separately labeled
  ensemble model.
- Reject unknown scientific configuration keys.
- Use content-addressed identities that exclude storage locations.
- Publish artifacts and study runs transactionally and verify exact bytes.
- Never run a main profile during automated tests or Codex maintenance work.
- Figure generation must read a verified completed result and must not
  implicitly start a numerical experiment.

## Architecture discipline

- Do not add publication- or experiment-number packages under `pol/`.
- A scalar calculation is a one-cell study; do not create separate scalar and
  sweep runners.
- Add a new study JSON before adding Python. Python is justified only for a new
  system, observation, readout, diagnostic, reporter, or reusable primitive.
- The core package must not import `studies/`.
- Do not restore removed compatibility commands or duplicate execution paths.
- Display labels may describe publication models, but implementation dispatch
  must use semantic kinds such as `affine_ridge`.
- Keep E0--E7 and Figure numbers in documentation correspondence tables only;
  never use them for core names or dispatch.

## Change discipline

- Preserve numerical meaning when moving code.
- Add or update tests for every scientific or artifact-contract change.
- Keep main profiles out of automated tests; use tiny or smoke profiles.
- Run `python -m compileall -q pol` and `pytest -q` before committing.
- Run the smoke script after changes to cross-module execution behavior.
