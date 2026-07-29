"""Unified trial and study execution."""

from importlib import import_module
from typing import Any

__all__ = ["run_study", "plan_study", "StudyRunResult"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    runner = import_module(".runner", __name__)
    value = getattr(runner, name)
    globals()[name] = value
    return value
