from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


DerivativeMethod = Literal["forward_euler", "central_difference", "sindy_default", "node"]
SymbolicMethod = Literal["sindy", "symantic"]


@dataclass
class ExperimentConfig:
    data_path: Path = Path("data/CDE_latest_allvars_Jan2025_cleaned_processed.csv")
    output_dir: Path = Path("outputs")
    batch_id: str = "D-R1-2018"
    train_fraction: float = 0.8
    random_seed: int = 42

    time_col: str = "CumulativeProp / Mlb"
    state_cols: tuple[str, str] = ("R2Salt_Median", "AA_Yield_R2Exit_smoothed")
    input_cols: tuple[str, ...] = (
        "PropyleneFlow / [Mlb/hr]",
        "PctC3H6_by_GC / mol%",
        "O2-C3H6_by_GC / mol",
        "PctSteam_by_GC / mol%",
        "Conversion_R2Exit",
        "AbsTemp",
        "AbsorberOutletPressure / psig",
    )

    normalization: Literal["zscore", "none"] = "zscore"
    differentiation_savgol_window: int = 11
    differentiation_savgol_order: int = 3

    sindy_poly_degree: int = 2
    sindy_threshold_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 5e-2)
    sindy_alpha: float = 1e-3
    sindy_complexity_weight: float = 1e-2
    sindy_include_interaction: bool = True
    sindy_include_bias: bool = True
    symantic_return_point: Literal["utopia", "most_accurate"] = "utopia"

    node_hidden_dim: int = 128
    node_lr: float = 1e-3
    node_weight_decay: float = 1e-5
    node_epochs: int = 300
    node_method: str = "rk4"

    run_derivative_methods: tuple[DerivativeMethod, ...] = (
        "forward_euler",
        "central_difference",
        "sindy_default",
        "node",
    )
    run_symbolic_methods: tuple[SymbolicMethod, ...] = ("sindy", "symantic")

    make_plots: bool = True

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "equations").mkdir(parents=True, exist_ok=True)


@dataclass
class RunRecord:
    derivative_method: DerivativeMethod
    symbolic_method: SymbolicMethod
    state_1_equation: str
    state_2_equation: str
    complexity_1: int
    complexity_2: int
    rmse_state_1: float
    rmse_state_2: float
    rmse_mean: float
    r2_state_1: float
    r2_state_2: float
    r2_mean: float
    metadata: dict[str, str] = field(default_factory=dict)
