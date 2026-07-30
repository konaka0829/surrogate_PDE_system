"""Read-only, cross-run reporting from verified completed studies."""

from .runner import ReportResult, run_report, verify_report

__all__ = ["ReportResult", "run_report", "verify_report"]
