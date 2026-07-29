"""Validation-bound reference datasets and finite-resolution views."""

__all__ = [
    "ReferenceDataset",
    "ensure_dataset",
    "load_dataset",
    "FiniteDataView",
    "derive_finite_view",
    "build_feature_initial_state",
    "InitialConditionArchive",
    "generate_grf_archive",
]


def __getattr__(name: str):
    if name in {"ReferenceDataset", "ensure_dataset", "load_dataset"}:
        from . import dataset

        value = getattr(dataset, name)
    elif name in {
        "FiniteDataView",
        "build_feature_initial_state",
        "derive_finite_view",
    }:
        from . import finite

        value = getattr(finite, name)
    elif name in {"InitialConditionArchive", "generate_grf_archive"}:
        from . import initial_conditions

        value = getattr(initial_conditions, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
