"""Scientific and algebraic validation independent of prediction studies."""

__all__ = [
    "DatasetBindingError",
    "ensure_validation",
    "evaluate_dataset_binding",
    "field_quadrature_foundation_summary",
    "load_validation_certificate",
    "matched_model1_cases",
    "run_field_quadrature_check",
    "run_matched_model1_pipeline_check",
    "run_model1_consistency_case",
    "run_validation",
    "verify_binding_proof",
]


def __getattr__(name: str):
    if name in {
        "DatasetBindingError",
        "evaluate_dataset_binding",
        "verify_binding_proof",
    }:
        from . import binding

        value = getattr(binding, name)
    elif name in {
        "ensure_validation",
        "load_validation_certificate",
        "run_validation",
    }:
        from . import runner

        value = getattr(runner, name)
    elif name in {
        "matched_model1_cases",
        "run_matched_model1_pipeline_check",
        "run_model1_consistency_case",
    }:
        from . import model1_consistency

        value = getattr(model1_consistency, name)
    elif name in {
        "field_quadrature_foundation_summary",
        "run_field_quadrature_check",
    }:
        from . import quadrature

        value = getattr(quadrature, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
