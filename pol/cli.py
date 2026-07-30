from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pol.artifacts import verify_artifact
from pol.config.loader import (
    load_dataset_spec,
    load_digital_baseline_spec,
    load_report_spec,
    load_study_spec,
    load_study_with_overrides,
    load_validation_spec,
)
from pol.data.dataset import ensure_dataset
from pol.digital_baselines.protocol import plan_digital_baseline
from pol.digital_baselines.runner import (
    run_digital_baseline,
    verify_digital_baseline_run,
)
from pol.reporting.runner import run_report, verify_report
from pol.study.runner import plan_study, run_study, verify_study_run
from pol.study.selection_source import (
    inspect_completed_study_selection,
    verify_downstream_selection,
)
from pol.validation.runner import ensure_validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pol",
        description="Validated surrogate-dynamics operator-learning workflows",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate",
        help=(
            "validate numerical foundations on CPU and publish a CPU-only "
            "certificate"
        ),
    )
    validate.add_argument(
        "spec",
        help="path to a CPU-only pol-validation-v3 JSON spec",
    )
    validate.add_argument("--force", action="store_true")

    data = commands.add_parser("data", help="reference-dataset operations")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    build = data_commands.add_parser(
        "build",
        help=(
            "build or reuse an explicitly validation-bound CPU reference dataset"
        ),
    )
    build.add_argument(
        "spec",
        help=(
            "path to a pol-dataset-v2 JSON spec with an explicit validation binding"
        ),
    )
    build.add_argument("--force", action="store_true")

    run = commands.add_parser(
        "run",
        help="run a CPU-only study; a scalar run is a one-cell study",
    )
    run.add_argument("spec", help="path to a pol-study-v3 JSON spec")
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

    selection = commands.add_parser(
        "selection",
        help="read-only completed-study selection dependency operations",
    )
    selection_commands = selection.add_subparsers(
        dest="selection_command",
        required=True,
    )
    inspect = selection_commands.add_parser(
        "inspect",
        help="inspect a verified completed study's representative selections",
    )
    inspect.add_argument("spec", help="path to a source pol-study-v3 spec")
    selection_verify = selection_commands.add_parser(
        "verify",
        help="verify downstream selection bindings without running a study",
    )
    selection_verify.add_argument(
        "spec",
        help="path to a downstream pol-study-v3 spec",
    )

    report = commands.add_parser(
        "report",
        help="build or verify a read-only cross-run report artifact",
    )
    report.add_argument(
        "target",
        help="path to a pol-report-v1 spec, or the literal 'verify'",
    )
    report.add_argument(
        "path",
        nargs="?",
        help="report artifact directory when target is 'verify'",
    )
    report.add_argument("--force", action="store_true")

    digital = commands.add_parser(
        "digital-baseline",
        help="run, plan, or verify a digital neural-operator baseline",
    )
    digital.add_argument(
        "target",
        help="path to a pol-digital-baseline-v3 spec, or the literal 'verify'",
    )
    digital.add_argument(
        "path",
        nargs="?",
        help="digital baseline run directory when target is 'verify'",
    )
    digital_modes = digital.add_mutually_exclusive_group()
    digital_modes.add_argument("--plan", action="store_true")
    digital_modes.add_argument("--force", action="store_true")
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
                    "binding_kind": dataset.binding_kind,
                    "binding_status": dataset.binding_status,
                    "target_reference_validation_status": (
                        dataset.target_reference_validation_status
                    ),
                    "binding_proof_hash": dataset.binding_proof_hash,
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
                _print(plan_study(spec, repo_root=repo_root))
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
            elif schema in {
                "pol-study-run-manifest-v8",
                "pol-study-run-manifest-v9",
                "pol-study-run-manifest-v10",
                "pol-study-run-manifest-v11",
                "pol-study-run-manifest-v12",
                "pol-study-run-manifest-v13",
                "pol-study-run-manifest-v14",
                "pol-study-run-manifest-v15",
                "pol-study-run-manifest-v16",
            }:
                manifest = verify_study_run(path)
                kind = "study_run"
            elif schema == "pol-report-manifest-v1":
                manifest = verify_report(path)
                kind = "report"
            elif schema == "pol-digital-baseline-run-manifest-v4":
                manifest = verify_digital_baseline_run(path)
                kind = "digital_baseline_run"
            else:
                raise ValueError(f"unsupported manifest schema: {schema}")
            _print({"status": "pass", "kind": kind, "path": str(path), "manifest": manifest})
            return 0
        if args.command == "selection":
            spec = load_study_spec(args.spec, repo_root=repo_root)
            if args.selection_command == "inspect":
                _print(
                    inspect_completed_study_selection(
                        spec,
                        repo_root=repo_root,
                    )
                )
                return 0
            if args.selection_command == "verify":
                _print(
                    verify_downstream_selection(
                        spec,
                        repo_root=repo_root,
                    )
                )
                return 0
            raise AssertionError("unreachable selection command")
        if args.command == "report":
            if args.target == "verify":
                if args.path is None:
                    raise ValueError("pol report verify requires a report directory")
                if args.force:
                    raise ValueError("--force is not valid for report verification")
                path = Path(args.path).resolve()
                manifest = verify_report(path)
                _print(
                    {
                        "status": "pass",
                        "kind": "report",
                        "path": str(path),
                        "manifest": manifest,
                    }
                )
                return 0
            if args.path is not None:
                raise ValueError(
                    "unexpected report argument; use pol report SPEC or "
                    "pol report verify REPORT_DIR"
                )
            spec = load_report_spec(args.target, repo_root=repo_root)
            result = run_report(
                spec,
                repo_root=repo_root,
                force=args.force,
            )
            _print(
                {
                    "status": result.summary["status"],
                    "kind": "report",
                    "path": str(result.path),
                    "report_id": result.report_id,
                    "reused": result.reused,
                    "summary": result.summary,
                }
            )
            return 0
        if args.command == "digital-baseline":
            if args.target == "verify":
                if args.path is None:
                    raise ValueError(
                        "pol digital-baseline verify requires a run directory"
                    )
                if args.plan or args.force:
                    raise ValueError(
                        "--plan/--force are not valid for digital baseline verification"
                    )
                path = Path(args.path).resolve()
                manifest = verify_digital_baseline_run(path)
                _print(
                    {
                        "status": "pass",
                        "kind": "digital_baseline_run",
                        "path": str(path),
                        "manifest": manifest,
                    }
                )
                return 0
            if args.path is not None:
                raise ValueError(
                    "unexpected digital baseline argument; use "
                    "pol digital-baseline SPEC or "
                    "pol digital-baseline verify RUN_DIR"
                )
            spec = load_digital_baseline_spec(
                args.target,
                repo_root=repo_root,
            )
            if args.plan:
                dataset_spec = load_dataset_spec(
                    spec.dataset_spec,
                    repo_root=repo_root,
                )
                validation_spec = load_validation_spec(
                    dataset_spec.validation_spec,
                    repo_root=repo_root,
                )
                _print(
                    plan_digital_baseline(
                        spec,
                        n_train=int(validation_spec.samples.n_train),
                    )
                )
                return 0
            result = run_digital_baseline(
                spec,
                repo_root=repo_root,
                force=args.force,
            )
            _print(
                {
                    "status": result.summary["status"],
                    "kind": "digital_baseline_run",
                    "path": str(result.path),
                    "run_id": result.run_id,
                    "reused": result.reused,
                    "summary": result.summary,
                }
            )
            return 0
        raise AssertionError("unreachable command")
    except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"pol: error: {exc}", file=sys.stderr)
        return 2
