"""Example workflow demonstrating the Budyko constrained ML integration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.budyko_ml_workflow import run_budyko_ml_workflow
from src.visualization.budyko_plots import BudykoVisualizer


def _create_synthetic_catchment(catchment_id: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", "2011-12-31", freq="D")
    n = len(dates)

    precip = rng.gamma(shape=2.0, scale=2.0, size=n)
    pet = rng.gamma(shape=2.0, scale=1.6, size=n)
    discharge = np.clip(precip - pet * 0.5 + rng.normal(0, 0.5, size=n), 0, None)

    return pd.DataFrame(
        {
            "date": dates,
            "discharge_mm_day": discharge,
            "precip_mm_day": precip,
            "pet_mm_day": pet,
            "area_km2": np.full(n, 500.0),
        }
    )


def main() -> None:
    catchments = {
        "BASIN_A": _create_synthetic_catchment("BASIN_A", seed=1),
        "BASIN_B": _create_synthetic_catchment("BASIN_B", seed=2),
        "BASIN_C": _create_synthetic_catchment("BASIN_C", seed=3),
    }

    attributes = pd.DataFrame(
        {
            "catchment_id": list(catchments.keys()),
            "TC": [10.5, 12.1, 8.9],
            "SAI": [0.4, 0.6, 0.5],
            "NDVI": [0.45, 0.52, 0.49],
            "HFP": [15, 20, 11],
        }
    )

    workflow = run_budyko_ml_workflow(
        catchments,
        attributes,
        climate_columns=("P_mm_yr", "Ep_mm_yr"),
        perform_cross_validation=False,
    )

    print("Training table columns:", workflow.training_table.columns.tolist())
    print("\nModel training metrics:")
    for target, metrics in workflow.model.training_metrics_.items():
        print(f"  {target}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}")

    print("\nPredicted parameters:")
    print(workflow.predictions[["alpha", "Qbp_mm_yr", "runoff_ratio", "baseflow_ratio"]])

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_data = {
        "Mean": {
            "IA": workflow.predictions["Ep_mm_yr"] / workflow.predictions["P_mm_yr"],
            "IE": 1 - workflow.predictions["runoff_ratio"],
            "omega": workflow.predictions["alpha"].mean(),
            "ia_mean": (workflow.predictions["Ep_mm_yr"] / workflow.predictions["P_mm_yr"]).mean(),
            "ie_mean": (1 - workflow.predictions["runoff_ratio"]).mean(),
            "start_year": 2000,
            "end_year": 2011,
        }
    }
    BudykoVisualizer.plot_catchment_trajectory(ax, plot_data)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
