from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pol.artifacts import verify_artifact
from pol.config.loader import (
    load_dataset_spec,
    load_study_spec,
    load_study_with_overrides,
    load_validation_spec,
)
from pol.data.dataset import ensure_dataset
from pol.study.runner import plan_study, run_study, verify_study_run
from pol.validation.runner import ensure_validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pol",
        description="Validated surrogate-dynamics operator-learning workflows",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate", help="validate numerical foundations and publish a certificate"
    )
    validate.add_argument("spec", help="path to a pol-validation-v1 JSON spec")
    validate.add_argument("--force", action="store_true")

    data = commands.add_parser("data", help="reference-dataset operations")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    build = data_commands.add_parser(
        "build", help="build or reuse a validated reference dataset"
    )
    build.add_argument("spec", help="path to a pol-dataset-v1 JSON spec")
    build.add_argument("--force", action="store_true")

    run = commands.add_parser(
        "run", help="run a study; a scalar run is a one-cell study"
    )
    run.add_argument("spec", help="path to a pol-study-v1 JSON spec")
    modes = run.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--force", action="store_true")
    modes.add_argument("--plots-only", action="store_true")
    run.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="DOTTED.PATH=JSON",
        help="override an existing study field; may be repeated",
    )

    verify = commands.add_parser(
        "verify", help="verify an artifact directory or completed study run"
    )
    verify.add_argument("path")
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    repo_root = Path.cwd().resolve()
    try:
        if args.command == "validate":
            spec = load_validation_spec(args.spec, repo_root=repo_root)
            outcome = ensure_validation(spec, force=args.force)
            _print(
                {
                    "status": "pass",
                    "artifact_id": outcome.reference.artifact_id,
                    "path": str(outcome.reference.path),
                    "certificate": outcome.certificate,
                }
            )
            return 0
        if args.command == "data":
            spec = load_dataset_spec(args.spec, repo_root=repo_root)
            dataset = ensure_dataset(spec, repo_root=repo_root, force=args.force)
            _print(
                {
                    "status": "pass",
                    "artifact_id": dataset.artifact_id,
                    "path": str(dataset.path),
                    "reference_nx": dataset.reference_nx,
                    "total_samples": dataset.total_samples,
                    "split_hash": dataset.split_hash,
                }
            )
            return 0
        if args.command == "run":
            spec = (
                load_study_with_overrides(
                    args.spec, repo_root=repo_root, overrides=args.overrides
                )
                if args.overrides
                else load_study_spec(args.spec, repo_root=repo_root)
            )
            if args.plan:
                _print(plan_study(spec))
                return 0
            result = run_study(
                spec,
                repo_root=repo_root,
                force=args.force,
                plots_only=args.plots_only,
            )
            _print(
                {
                    "status": result.summary.get("status", "pass"),
                    "path": str(result.path),
                    "reused": result.reused,
                    "summary": result.summary,
                }
            )
            return 0
        if args.command == "verify":
            path = Path(args.path).resolve()
            manifest_path = path / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError(f"no manifest.json found in {path}")
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            schema = raw.get("schema_version")
            if schema == "pol-artifact-manifest-v1":
                manifest = verify_artifact(path)
                kind = "artifact"
            elif schema == "pol-study-run-manifest-v1":
                manifest = verify_study_run(path)
                kind = "study_run"
            else:
                raise ValueError(f"unsupported manifest schema: {schema}")
            _print({"status": "pass", "kind": kind, "path": str(path), "manifest": manifest})
            return 0
        raise AssertionError("unreachable command")
    except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"pol: error: {exc}", file=sys.stderr)
        return 2
