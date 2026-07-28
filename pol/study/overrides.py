from __future__ import annotations

import copy
from typing import Any, Mapping

from pydantic import ValidationError

from pol.config.models import TrialSpec


def set_dotted(root: dict[str, Any], path: str, value: Any) -> None:
    if not path or path.startswith(".") or path.endswith("."):
        raise ValueError(f"invalid dotted path: {path!r}")
    parts = path.split(".")
    current: Any = root
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"override path does not exist: {path}")
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise ValueError(f"override path does not exist: {path}")
    current[parts[-1]] = copy.deepcopy(value)


def apply_trial_overrides(base: TrialSpec, overrides: Mapping[str, Any]) -> TrialSpec:
    payload = base.model_dump(mode="python")
    for path, value in overrides.items():
        set_dotted(payload, path, value)
    try:
        return TrialSpec.model_validate(payload)
    except ValidationError as exc:
        location = "$"
        error = exc.errors(include_url=False)[0]
        for part in error.get("loc", ()):
            location += f"[{part}]" if isinstance(part, int) else f".{part}"
        raise ValueError(f"invalid trial after overrides at {location}: {error['msg']}") from exc

