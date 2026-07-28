"""Stable software/backend fingerprints for reproducible computation identities."""
from __future__ import annotations

import platform
import sys
from typing import Any

import numpy as np
import scipy
import torch

from pol import __version__
from pol.runtime.device import execution_device_policy


def numerical_environment_fingerprint() -> dict[str, Any]:
    """Return backend facts that can change deterministic numerical bytes."""
    return {
        "schema_version": "pol-numerical-environment-v2",
        **execution_device_policy(),
        "pol_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__),
    }


__all__ = ["numerical_environment_fingerprint"]
