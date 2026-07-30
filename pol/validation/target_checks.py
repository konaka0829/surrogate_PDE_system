from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pol.config.models import ValidationSpec
from pol.data.initial_conditions import InitialConditionArchive
from .burgers_reference import (
    run_burgers_reference_checks,
    validate_burgers_reference_checks,
)
from .heat_reference import (
    heat_contract_components,
    run_heat_reference_checks,
    validate_heat_reference_checks,
)
from .reaction_diffusion_reference import (
    run_reaction_diffusion_reference_checks,
    validate_reaction_diffusion_reference_checks,
)
from .reference_convergence import time_refined_contract_components


@dataclass(frozen=True)
class TargetCheckHandler:
    run: Callable[
        [ValidationSpec, InitialConditionArchive],
        tuple[dict[str, Any], list[dict[str, Any]]],
    ]
    validate: Callable[[ValidationSpec, dict[str, Any]], None]
    contract: Callable[
        [ValidationSpec, dict[str, Any]],
        dict[str, Any],
    ]


_TARGET_CHECK_HANDLERS = {
    "burgers_convergence": TargetCheckHandler(
        run=run_burgers_reference_checks,
        validate=validate_burgers_reference_checks,
        contract=time_refined_contract_components,
    ),
    "heat_analytic": TargetCheckHandler(
        run=run_heat_reference_checks,
        validate=validate_heat_reference_checks,
        contract=heat_contract_components,
    ),
    "reaction_diffusion_convergence": TargetCheckHandler(
        run=run_reaction_diffusion_reference_checks,
        validate=validate_reaction_diffusion_reference_checks,
        contract=time_refined_contract_components,
    ),
}


def target_check_handler(spec: ValidationSpec) -> TargetCheckHandler:
    kind = spec.target_reference.kind
    try:
        return _TARGET_CHECK_HANDLERS[kind]
    except KeyError as exc:
        raise TypeError(
            f"unsupported target-reference validation: {kind}"
        ) from exc


def run_target_checks(
    spec: ValidationSpec,
    archive: InitialConditionArchive,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    handler = target_check_handler(spec)
    return handler.run(spec, archive)


def validate_target_checks(
    spec: ValidationSpec,
    checks: dict[str, Any],
) -> None:
    target_check_handler(spec).validate(spec, checks)


def target_contract_components(
    spec: ValidationSpec,
    convergence: dict[str, Any],
) -> dict[str, Any]:
    return target_check_handler(spec).contract(spec, convergence)
