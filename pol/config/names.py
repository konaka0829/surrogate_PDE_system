from __future__ import annotations

from pathlib import Path
import re


_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def validate_safe_path_component(value: str, *, field: str) -> str:
    """Validate one project-owned output path component."""

    if not value or not value.strip():
        raise ValueError(f"{field} must be a nonempty safe basename")
    if value != value.strip():
        raise ValueError(f"{field} must not have leading or trailing whitespace")
    if (
        Path(value).name != value
        or value in {".", ".."}
        or value.startswith(".")
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{field} must be a nonempty safe basename")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field} must not contain control characters")
    if _SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(
            f"{field} must use only ASCII letters, digits, '_' and '-', "
            "and must start with a letter or digit"
        )
    return value


def validate_extension_free_filename(value: str, *, field: str) -> str:
    """Validate one project-owned extension-free output filename."""

    return validate_safe_path_component(value, field=field)


__all__ = [
    "validate_extension_free_filename",
    "validate_safe_path_component",
]
