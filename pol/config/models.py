from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    JsonValue,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DomainSpec(StrictModel):
    length: PositiveFloat = 1.0


class GRFSpec(StrictModel):
    kind: Literal["periodic_grf"] = "periodic_grf"
    gamma: PositiveFloat = 2.0
    tau: float = 5.0
    sigma: float = 25.0
    mean: float = 0.0

    @model_validator(mode="after")
    def _finite_nonnegative(self) -> "GRFSpec":
        if self.tau < 0 or self.sigma < 0:
            raise ValueError("tau and sigma must be nonnegative")
        return self


class SampleSpec(StrictModel):
    total_samples: PositiveInt
    n_train: PositiveInt
    n_validation: int = Field(ge=0)
    n_test: int = Field(ge=0)
    seed: int = 0
    dtype: Literal["float32", "float64"] = "float64"
    device: Literal["cpu"] = "cpu"
    initial_condition: GRFSpec = Field(default_factory=GRFSpec)
    preprocessing: Literal["l2_scaling_only"] = "l2_scaling_only"

    @field_validator("device", mode="before")
    @classmethod
    def _cpu_only_device(cls, value: object) -> object:
        if value != "cpu":
            raise ValueError(
                "official scientific workflows are CPU-only; device must be "
                "'cpu' (CUDA, auto selection, and unknown devices are unsupported)"
            )
        return value

    @model_validator(mode="after")
    def _counts(self) -> "SampleSpec":
        if self.total_samples != self.n_train + self.n_validation + self.n_test:
            raise ValueError(
                "total_samples must equal n_train + n_validation + n_test"
            )
        return self


class HeatSystemSpec(StrictModel):
    kind: Literal["heat"] = "heat"
    nu: PositiveFloat


class BurgersSystemSpec(StrictModel):
    kind: Literal["burgers"] = "burgers"
    nu: PositiveFloat
    advection_coefficient: float = 1.0
    solver: Literal[
        "split_step", "semi_implicit", "etdrk4", "fourier_pseudospectral_etdrk4"
    ] = "split_step"
    dt: PositiveFloat
    fine_dt: PositiveFloat | None = None
    dealias: bool = True

    @model_validator(mode="after")
    def _solver_fields(self) -> "BurgersSystemSpec":
        if self.solver in {"split_step", "semi_implicit"} and self.fine_dt is None:
            raise ValueError("split-step Burgers requires fine_dt")
        if (
            self.solver
            in {"etdrk4", "fourier_pseudospectral_etdrk4"}
            and self.fine_dt is not None
        ):
            raise ValueError("ETDRK4 Burgers requires fine_dt=null")
        if self.advection_coefficient != 1.0:
            raise ValueError(
                "the current Burgers kernel supports advection_coefficient=1.0 only"
            )
        return self


class ReactionDiffusionSystemSpec(StrictModel):
    kind: Literal["reaction_diffusion"] = "reaction_diffusion"
    nu: PositiveFloat
    alpha: float = 1.0
    beta: float = 1.0
    solver: Literal["semi_implicit_spectral_euler"] = (
        "semi_implicit_spectral_euler"
    )
    dt: PositiveFloat
    nonlinear_filter: Literal["none", "two_thirds"] = "two_thirds"

    @model_validator(mode="after")
    def _finite_parameters(self) -> "ReactionDiffusionSystemSpec":
        if not all(
            math.isfinite(float(value))
            for value in (self.nu, self.alpha, self.beta, self.dt)
        ):
            raise ValueError(
                "reaction-diffusion parameters and dt must be finite"
            )
        return self


SystemSpec = Annotated[
    Union[HeatSystemSpec, BurgersSystemSpec, ReactionDiffusionSystemSpec],
    Field(discriminator="kind"),
]


class EvolutionSpec(StrictModel):
    system: SystemSpec
    time: float = Field(gt=0)

    @model_validator(mode="after")
    def _alignment(self) -> "EvolutionSpec":
        if not math.isfinite(float(self.time)):
            raise ValueError("evolution time must be finite")
        dt = getattr(self.system, "dt", None)
        if dt is not None:
            steps = round(self.time / dt)
            if abs(steps * dt - self.time) > 1e-10 * max(1.0, abs(self.time)):
                raise ValueError("evolution time must align with the configured dt")
        return self


class BurgersTimeCandidateSpec(StrictModel):
    dt: PositiveFloat
    fine_dt: PositiveFloat | None = None
    solver: Literal[
        "split_step", "semi_implicit", "etdrk4", "fourier_pseudospectral_etdrk4"
    ] = "split_step"
    dealias: bool = True

    @model_validator(mode="after")
    def _fine_dt(self) -> "BurgersTimeCandidateSpec":
        if self.solver in {"split_step", "semi_implicit"} and self.fine_dt is None:
            raise ValueError("split-step candidate requires fine_dt")
        if (
            self.solver
            in {"etdrk4", "fourier_pseudospectral_etdrk4"}
            and self.fine_dt is not None
        ):
            raise ValueError("ETDRK4 candidate requires fine_dt=null")
        return self


class ReactionDiffusionTimeCandidateSpec(StrictModel):
    dt: PositiveFloat
    solver: Literal["semi_implicit_spectral_euler"] = (
        "semi_implicit_spectral_euler"
    )
    nonlinear_filter: Literal["none", "two_thirds"] = "two_thirds"

    @model_validator(mode="after")
    def _finite_dt(self) -> "ReactionDiffusionTimeCandidateSpec":
        if not math.isfinite(float(self.dt)):
            raise ValueError("reaction-diffusion candidate dt must be finite")
        return self


class ReferenceToleranceSpec(StrictModel):
    mean_relative_l2: float = Field(ge=0)
    max_relative_l2: float = Field(ge=0)
    low_mode_relative_l2: float = Field(ge=0)

    @model_validator(mode="after")
    def _finite(self) -> "ReferenceToleranceSpec":
        if not all(
            math.isfinite(value)
            for value in (
                self.mean_relative_l2,
                self.max_relative_l2,
                self.low_mode_relative_l2,
            )
        ):
            raise ValueError("reference tolerances must be finite")
        return self


class BurgersCrossSolverContextSpec(StrictModel):
    system_kind: Literal["burgers"] = "burgers"
    nu: PositiveFloat
    advection_coefficient: float = 1.0
    final_time: PositiveFloat
    domain_length: PositiveFloat
    dtype: Literal["float32", "float64"]
    dealias: bool

    @model_validator(mode="after")
    def _supported_advection(self) -> "BurgersCrossSolverContextSpec":
        if self.advection_coefficient != 1.0:
            raise ValueError(
                "the current Burgers kernel supports "
                "advection_coefficient=1.0 only"
            )
        return self


class BurgersCrossSolverFamilySpec(StrictModel):
    candidates: tuple[BurgersTimeCandidateSpec, ...]


class BurgersCrossSolverFamiliesSpec(StrictModel):
    split_step: BurgersCrossSolverFamilySpec
    etdrk4: BurgersCrossSolverFamilySpec


class DisabledBurgersCrossSolverValidationSpec(StrictModel):
    schema_version: Literal["pol-burgers-cross-solver-spec-v1"] = (
        "pol-burgers-cross-solver-spec-v1"
    )
    enabled: Literal[False] = False


class EnabledBurgersCrossSolverValidationSpec(StrictModel):
    schema_version: Literal["pol-burgers-cross-solver-spec-v1"] = (
        "pol-burgers-cross-solver-spec-v1"
    )
    enabled: Literal[True]
    context: BurgersCrossSolverContextSpec
    solvers: BurgersCrossSolverFamiliesSpec
    tolerances: ReferenceToleranceSpec


BurgersCrossSolverValidationSpec = Annotated[
    Union[
        DisabledBurgersCrossSolverValidationSpec,
        EnabledBurgersCrossSolverValidationSpec,
    ],
    Field(discriminator="enabled"),
]


class AlgebraicToleranceSpec(StrictModel):
    float64_atol: float = Field(default=1e-10, ge=0)
    float64_rtol: float = Field(default=1e-10, ge=0)
    float32_atol: float = Field(default=1e-5, ge=0)
    float32_rtol: float = Field(default=1e-5, ge=0)


class InterfaceDimensionsSpec(StrictModel):
    n_tar: PositiveInt
    n_sur: PositiveInt
    J: PositiveInt
    q: PositiveInt

    @model_validator(mode="after")
    def _representability(self) -> "InterfaceDimensionsSpec":
        if self.J > self.n_sur:
            raise ValueError("J must be <= n_sur")
        if self.q % 2 == 0:
            raise ValueError("q must be odd")
        if self.q > self.n_tar:
            raise ValueError("q must be <= n_tar")
        return self


class ReducedObservationSpec(StrictModel):
    J: PositiveInt
    q: PositiveInt

    @model_validator(mode="after")
    def _q(self) -> "ReducedObservationSpec":
        if self.q % 2 == 0:
            raise ValueError("q must be odd")
        return self


class TargetReferenceValidationBase(StrictModel):
    reference_evolution: EvolutionSpec
    calibration_sample_ids: tuple[int, ...]
    reference_nx_candidates: tuple[PositiveInt, ...]
    q_reference_check: PositiveInt
    reference_tolerances: ReferenceToleranceSpec
    selection_policy: Literal["coarsest_passing_with_finest_pair_required"] = (
        "coarsest_passing_with_finest_pair_required"
    )

    @model_validator(mode="after")
    def _reference_constraints(self) -> "TargetReferenceValidationBase":
        if len(self.reference_nx_candidates) < 2:
            raise ValueError("at least two reference_nx_candidates are required")
        if tuple(sorted(set(self.reference_nx_candidates))) != tuple(
            self.reference_nx_candidates
        ):
            raise ValueError(
                "reference_nx_candidates must be strictly increasing and unique"
            )
        if not self.calibration_sample_ids:
            raise ValueError("calibration_sample_ids must not be empty")
        if len(set(self.calibration_sample_ids)) != len(
            self.calibration_sample_ids
        ):
            raise ValueError("calibration_sample_ids must be unique")
        if self.q_reference_check % 2 == 0:
            raise ValueError("q_reference_check must be odd")
        if self.q_reference_check > min(self.reference_nx_candidates):
            raise ValueError(
                "q_reference_check must be <= the coarsest reference nx"
            )
        return self


class BurgersConvergenceReferenceSpec(TargetReferenceValidationBase):
    kind: Literal["burgers_convergence"] = "burgers_convergence"
    time_candidates: tuple[BurgersTimeCandidateSpec, ...]
    cross_solver_validation: BurgersCrossSolverValidationSpec = Field(
        default_factory=DisabledBurgersCrossSolverValidationSpec
    )

    @model_validator(mode="after")
    def _burgers_constraints(self) -> "BurgersConvergenceReferenceSpec":
        if self.reference_evolution.system.kind != "burgers":
            raise ValueError(
                "burgers_convergence requires a Burgers reference evolution"
            )
        if len(self.time_candidates) < 2:
            raise ValueError("at least two time_candidates are required")
        from pol.validation.conditions import (
            burgers_refinement_proof,
            canonical_numerical_condition,
        )

        evolution_time = float(self.reference_evolution.time)
        proof = burgers_refinement_proof(
            [
                candidate.model_dump(mode="json")
                for candidate in self.time_candidates
            ],
            evolution_time=evolution_time,
        )
        system = self.reference_evolution.system
        reference_time_condition = canonical_numerical_condition(
            "burgers",
            system.model_dump(mode="json"),
            evolution_time=evolution_time,
        )
        if reference_time_condition != proof["ordered_candidates"][-1]:
            raise ValueError(
                "reference_evolution time discretization must equal the finest "
                "canonical time_candidates condition"
            )
        diagnostic = self.cross_solver_validation
        if diagnostic.enabled:
            context = diagnostic.context
            if (
                context.system_kind != system.kind
                or float(context.nu) != float(system.nu)
                or float(context.advection_coefficient)
                != float(system.advection_coefficient)
                or float(context.final_time) != evolution_time
            ):
                raise ValueError(
                    "cross-solver PDE parameters and final time must exactly "
                    "match reference_evolution"
                )
            if context.dealias != reference_time_condition["dealias"]:
                raise ValueError(
                    "cross-solver dealias policy must exactly match the "
                    "primary reference evolution"
                )
            for family_name in ("split_step", "etdrk4"):
                family_spec = getattr(diagnostic.solvers, family_name)
                family_proof = burgers_refinement_proof(
                    [
                        candidate.model_dump(mode="json")
                        for candidate in family_spec.candidates
                    ],
                    evolution_time=evolution_time,
                )
                if family_proof["canonical_solver_family"] != family_name:
                    raise ValueError(
                        "cross-solver candidate sequence is assigned to the "
                        f"wrong solver family: expected {family_name}"
                    )
                if family_proof["dealias"] != context.dealias:
                    raise ValueError(
                        "cross-solver candidate dealias policy must exactly "
                        "match the diagnostic context"
                    )
        return self


class HeatAnalyticReferenceSpec(TargetReferenceValidationBase):
    kind: Literal["heat_analytic"] = "heat_analytic"

    @model_validator(mode="after")
    def _heat_constraints(self) -> "HeatAnalyticReferenceSpec":
        if self.reference_evolution.system.kind != "heat":
            raise ValueError(
                "heat_analytic requires a heat reference evolution"
            )
        return self


class ReactionDiffusionConvergenceReferenceSpec(
    TargetReferenceValidationBase
):
    kind: Literal["reaction_diffusion_convergence"] = (
        "reaction_diffusion_convergence"
    )
    time_candidates: tuple[ReactionDiffusionTimeCandidateSpec, ...]

    @model_validator(mode="after")
    def _reaction_diffusion_constraints(
        self,
    ) -> "ReactionDiffusionConvergenceReferenceSpec":
        if self.reference_evolution.system.kind != "reaction_diffusion":
            raise ValueError(
                "reaction_diffusion_convergence requires a "
                "reaction-diffusion reference evolution"
            )
        if len(self.time_candidates) < 2:
            raise ValueError("at least two time_candidates are required")
        from pol.validation.conditions import (
            canonical_numerical_condition,
            reaction_diffusion_refinement_proof,
        )

        evolution_time = float(self.reference_evolution.time)
        proof = reaction_diffusion_refinement_proof(
            [
                candidate.model_dump(mode="json")
                for candidate in self.time_candidates
            ],
            evolution_time=evolution_time,
        )
        system = self.reference_evolution.system
        reference_condition = canonical_numerical_condition(
            "reaction_diffusion",
            system.model_dump(mode="json"),
            evolution_time=evolution_time,
        )
        if reference_condition != proof["ordered_candidates"][-1]:
            raise ValueError(
                "reference_evolution time discretization must equal the "
                "finest canonical time_candidates condition"
            )
        return self


TargetReferenceValidationSpec = Annotated[
    Union[
        BurgersConvergenceReferenceSpec,
        HeatAnalyticReferenceSpec,
        ReactionDiffusionConvergenceReferenceSpec,
    ],
    Field(discriminator="kind"),
]


class ValidationSpec(StrictModel):
    schema_version: Literal["pol-validation-v5", "pol-validation-v6"] = (
        "pol-validation-v6"
    )
    name: str
    artifact_root: Path = Path("artifacts")
    profile: str = "smoke"
    domain: DomainSpec
    samples: SampleSpec
    target_reference: TargetReferenceValidationSpec
    algebraic_tolerances: AlgebraicToleranceSpec = Field(
        default_factory=AlgebraicToleranceSpec
    )
    full_interface: InterfaceDimensionsSpec
    reduced_observation: ReducedObservationSpec

    @model_validator(mode="after")
    def _validation_constraints(self) -> "ValidationSpec":
        target = self.target_reference
        if (
            isinstance(
                target,
                ReactionDiffusionConvergenceReferenceSpec,
            )
            and self.schema_version != "pol-validation-v6"
        ):
            raise ValueError(
                "reaction_diffusion_convergence requires "
                "schema_version=pol-validation-v6"
            )
        if any(
            i < 0 or i >= self.samples.total_samples
            for i in target.calibration_sample_ids
        ):
            raise ValueError("calibration_sample_ids must lie in the dataset range")
        if self.full_interface.n_tar > max(target.reference_nx_candidates):
            raise ValueError("full_interface.n_tar exceeds the master resolution")
        if self.reduced_observation.J > self.full_interface.n_sur:
            raise ValueError("reduced observation J must be <= full n_sur")
        if self.reduced_observation.q > self.full_interface.n_tar:
            raise ValueError("reduced observation q must be <= full n_tar")
        if (
            isinstance(target, BurgersConvergenceReferenceSpec)
            and target.cross_solver_validation.enabled
        ):
            context = target.cross_solver_validation.context
            if (
                float(context.domain_length) != float(self.domain.length)
                or context.dtype != self.samples.dtype
            ):
                raise ValueError(
                    "cross-solver domain length and dtype must exactly match "
                    "the validation context"
                )
        return self


class ValidatedReferenceBindingSpec(StrictModel):
    kind: Literal["validated_reference"] = "validated_reference"


class FoundationOnlyBindingSpec(StrictModel):
    kind: Literal["foundation_only"] = "foundation_only"
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _nonempty_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("foundation_only reason must not be blank")
        return value


DatasetBindingSpec = Annotated[
    Union[ValidatedReferenceBindingSpec, FoundationOnlyBindingSpec],
    Field(discriminator="kind"),
]


class DatasetSpec(StrictModel):
    schema_version: Literal["pol-dataset-v3"] = "pol-dataset-v3"
    name: str
    artifact_root: Path = Path("artifacts")
    validation_spec: Path
    binding: DatasetBindingSpec
    reference_nx: PositiveInt
    target: EvolutionSpec
    batch_size: PositiveInt = 20


class PointObservationSpec(StrictModel):
    kind: Literal["equispaced_points"] = "equispaced_points"
    J: PositiveInt
    l2_scale: bool = True


class FeatureGeneratorSpec(StrictModel):
    kind: Literal["pde_dynamics", "static_input"] = "pde_dynamics"
    evolution: EvolutionSpec | None = None
    n_sur: PositiveInt
    observation: PointObservationSpec

    @model_validator(mode="after")
    def _observation(self) -> "FeatureGeneratorSpec":
        if self.kind == "pde_dynamics" and self.evolution is None:
            raise ValueError("pde_dynamics feature generation requires evolution")
        if self.kind == "static_input" and self.evolution is not None:
            raise ValueError("static_input feature generation must not define evolution")
        if self.observation.J > self.n_sur:
            raise ValueError("observation.J must be <= feature.n_sur")
        return self


class FiniteInputSpec(StrictModel):
    n_tar: PositiveInt
    resampling: Literal["spectral"] = "spectral"


class FourierOutputSpec(StrictModel):
    kind: Literal["real_fourier"] = "real_fourier"
    q: PositiveInt

    @field_validator("q")
    @classmethod
    def _odd(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("q must be odd")
        return value


class DirectReadoutSpec(StrictModel):
    id: str
    kind: Literal["direct_fourier_decoder"] = "direct_fourier_decoder"
    display_name: str | None = None


class AffineRidgeReadoutSpec(StrictModel):
    id: str
    kind: Literal["affine_ridge"] = "affine_ridge"
    display_name: str | None = None
    zetas: tuple[float, ...]
    tie_tolerance: float = Field(default=1e-12, ge=0)
    tie_break: Literal["largest_zeta", "first"] = "largest_zeta"
    svd_rcond: float | None = None

    @model_validator(mode="after")
    def _zetas(self) -> "AffineRidgeReadoutSpec":
        if not self.zetas or any(value < 0 for value in self.zetas):
            raise ValueError("zetas must be a nonempty nonnegative sequence")
        if len(set(self.zetas)) != len(self.zetas):
            raise ValueError("zetas must be unique")
        return self


class RandomFeatureRidgeReadoutSpec(StrictModel):
    id: str
    kind: Literal["random_feature_ridge"] = "random_feature_ridge"
    display_name: str | None = None
    activation: Literal["tanh", "relu", "identity"] = "tanh"
    widths: tuple[PositiveInt, ...]
    weight_scales: tuple[float, ...]
    bias_scales: tuple[float, ...]
    selection_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    zetas: tuple[float, ...]
    tie_tolerance: float = Field(default=1e-12, ge=0)
    svd_rcond: float | None = None

    @model_validator(mode="after")
    def _candidates(self) -> "RandomFeatureRidgeReadoutSpec":
        sequences = (
            self.widths,
            self.weight_scales,
            self.bias_scales,
            self.selection_seeds,
            self.evaluation_seeds,
            self.zetas,
        )
        if any(not sequence for sequence in sequences):
            raise ValueError("all random-feature candidate sequences must be nonempty")
        if any(value < 0 for value in (*self.weight_scales, *self.bias_scales, *self.zetas)):
            raise ValueError("random-feature scales and zetas must be nonnegative")
        if len(set(self.selection_seeds)) != len(self.selection_seeds):
            raise ValueError("selection_seeds must be unique")
        if len(self.evaluation_seeds) < 2:
            raise ValueError("evaluation_seeds must contain at least two seeds")
        if len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError("evaluation_seeds must be unique")
        if set(self.selection_seeds) & set(self.evaluation_seeds):
            raise ValueError("selection and evaluation seeds must be disjoint")
        return self


ReadoutSpec = Annotated[
    Union[
        DirectReadoutSpec,
        AffineRidgeReadoutSpec,
        RandomFeatureRidgeReadoutSpec,
    ],
    Field(discriminator="kind"),
]


class TrialSpec(StrictModel):
    input: FiniteInputSpec
    feature: FeatureGeneratorSpec
    output: FourierOutputSpec
    readouts: tuple[ReadoutSpec, ...]

    @model_validator(mode="after")
    def _trial(self) -> "TrialSpec":
        if self.output.q > self.input.n_tar:
            raise ValueError("output.q must be <= input.n_tar")
        ids = [readout.id for readout in self.readouts]
        if len(ids) != len(set(ids)):
            raise ValueError("readout ids must be unique")
        if not ids:
            raise ValueError("at least one readout is required")
        return self


class SweepAxisSpec(StrictModel):
    path: str
    values: tuple[JsonValue, ...]

    @model_validator(mode="after")
    def _values(self) -> "SweepAxisSpec":
        if not self.values:
            raise ValueError("sweep axis values must not be empty")
        return self


class StaticSearchSpec(StrictModel):
    kind: Literal["static"] = "static"


class GridSearchSpec(StrictModel):
    kind: Literal["grid"] = "grid"
    axes: tuple[SweepAxisSpec, ...]

    @model_validator(mode="after")
    def _axes(self) -> "GridSearchSpec":
        if not self.axes:
            raise ValueError("grid search requires at least one axis")
        return self


class CoordinateAxisSpec(StrictModel):
    path: str
    values: tuple[JsonValue, ...]
    anchor: JsonValue

    @model_validator(mode="after")
    def _values(self) -> "CoordinateAxisSpec":
        if not self.values:
            raise ValueError("coordinate axis values must not be empty")
        return self


class CoordinateSearchSpec(StrictModel):
    kind: Literal["coordinate"] = "coordinate"
    axes: tuple[CoordinateAxisSpec, CoordinateAxisSpec]
    rounds: int = Field(default=0, ge=0)


SearchSpec = Annotated[
    Union[StaticSearchSpec, GridSearchSpec, CoordinateSearchSpec],
    Field(discriminator="kind"),
]


class VariantSpec(StrictModel):
    id: str
    display_name: str | None = None
    overrides: dict[str, JsonValue] = Field(default_factory=dict)
    search: SearchSpec = Field(default_factory=StaticSearchSpec)


class SelectionSpec(StrictModel):
    metric: Literal[
        "validation_field_relative_l2_mean",
        "validation_coefficient_mse",
    ] = "validation_field_relative_l2_mean"
    tie_tolerance: float = Field(default=1e-12, ge=0)
    tie_break: Literal["first_in_config_order"] = "first_in_config_order"
    representative_readout: str
    freeze_before_test: Literal[True] = True


class ConvergenceToleranceSpec(StrictModel):
    terminal_mean: float = Field(ge=0)
    terminal_max: float = Field(ge=0)
    feature_mean: float = Field(ge=0)
    feature_max: float = Field(ge=0)
    prediction_mean: float = Field(ge=0)
    prediction_max: float = Field(ge=0)


class ConvergenceSpec(StrictModel):
    sample_ids: tuple[int, ...]
    n_sur_candidates: tuple[PositiveInt, ...]
    tolerances: ConvergenceToleranceSpec
    max_auto_reruns: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _candidates(self) -> "ConvergenceSpec":
        if len(self.n_sur_candidates) < 2:
            raise ValueError("convergence requires at least two n_sur candidates")
        if tuple(sorted(set(self.n_sur_candidates))) != tuple(self.n_sur_candidates):
            raise ValueError("n_sur_candidates must be increasing and unique")
        if not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be nonempty and unique")
        return self


class HeatMultiplierDiagnosticSpec(StrictModel):
    kind: Literal["heat_multiplier"] = "heat_multiplier"
    identifiable_variance_floor: float = Field(default=1e-14, ge=0)


class NoiseDiagnosticSpec(StrictModel):
    kind: Literal["noise_robustness"] = "noise_robustness"
    levels: tuple[float, ...]
    repeats: PositiveInt
    seed: int

    @model_validator(mode="after")
    def _levels(self) -> "NoiseDiagnosticSpec":
        if not self.levels or any(value < 0 for value in self.levels):
            raise ValueError("noise levels must be nonempty and nonnegative")
        return self


DiagnosticSpec = Annotated[
    Union[HeatMultiplierDiagnosticSpec, NoiseDiagnosticSpec],
    Field(discriminator="kind"),
]


class MetricCurveReporterSpec(StrictModel):
    kind: Literal["metric_curve"] = "metric_curve"
    filename: str
    x: str
    metric: str
    split: Literal["validation", "test"] = "validation"
    group_by: tuple[str, ...] = ("variant_id", "readout_id")
    xscale: Literal["linear", "log"] = "linear"
    yscale: Literal["linear", "log"] = "log"
    formats: tuple[Literal["png", "pdf"], ...] = ("png",)
    dpi: PositiveInt = 120


class ResolutionMapReporterSpec(StrictModel):
    kind: Literal["resolution_map"] = "resolution_map"
    filename: str
    x: str
    y: str
    metric: str
    split: Literal["validation", "test"] = "validation"
    readout_id: str
    formats: tuple[Literal["png", "pdf"], ...] = ("png",)
    dpi: PositiveInt = 120


class NoiseCurveReporterSpec(StrictModel):
    kind: Literal["noise_curve"] = "noise_curve"
    filename: str
    metric: str = "field_relative_l2_mean"
    formats: tuple[Literal["png", "pdf"], ...] = ("png",)
    dpi: PositiveInt = 120


ReporterSpec = Annotated[
    Union[MetricCurveReporterSpec, ResolutionMapReporterSpec, NoiseCurveReporterSpec],
    Field(discriminator="kind"),
]


class ExecutionSpec(StrictModel):
    torch_threads: PositiveInt | None = 1
    batch_size: PositiveInt = 64
    invalid_trial_policy: Literal["error", "skip"] = "error"
    cache_states: bool = True
    generate_plots: bool = True


class StudySpec(StrictModel):
    schema_version: Literal["pol-study-v1"] = "pol-study-v1"
    name: str
    output_root: Path = Path("outputs/studies")
    artifact_root: Path = Path("artifacts")
    profile: str = "smoke"
    dataset_spec: Path
    base_trial: TrialSpec
    variants: tuple[VariantSpec, ...]
    global_axes: tuple[SweepAxisSpec, ...] = ()
    selection: SelectionSpec
    convergence: ConvergenceSpec | None = None
    diagnostics: tuple[DiagnosticSpec, ...] = ()
    reporters: tuple[ReporterSpec, ...] = ()
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)

    @model_validator(mode="after")
    def _study(self) -> "StudySpec":
        if not self.variants:
            raise ValueError("at least one variant is required")
        variant_ids = [variant.id for variant in self.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("variant ids must be unique")
        readout_ids = {readout.id for readout in self.base_trial.readouts}
        if self.selection.representative_readout not in readout_ids:
            raise ValueError("representative_readout must name a configured readout")
        if self.convergence is not None:
            if any(
                sample_id < 0 for sample_id in self.convergence.sample_ids
            ):
                raise ValueError("convergence sample ids must be nonnegative")
        return self
