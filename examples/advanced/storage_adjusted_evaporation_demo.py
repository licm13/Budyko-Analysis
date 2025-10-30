"""Example usage of the storage-adjusted evaporation engine."""

from __future__ import annotations

import numpy as np

from src.budyko.storage_adjustment import StorageAdjustmentEngine

np.random.seed(42)


def run_scenario(name: str, precipitation, runoff, storage=None):
    engine = StorageAdjustmentEngine(allow_negative_evaporation=name == "extreme")
    result = engine.compute(precipitation, runoff, storage_change=storage)
    print(f"Scenario: {name}")
    print(f"  precipitation: {result.precipitation}")
    print(f"  runoff:        {result.runoff}")
    print(f"  storage ΔS:    {result.storage_change}")
    print(f"  actual Ea:     {result.actual_evaporation}\n")


# Minimal deterministic triad.
run_scenario(
    "minimal",
    precipitation=[1.0, 1.1, 0.9],
    runoff=[0.3, 0.4, 0.2],
    storage=[0.05, 0.0, 0.02],
)

# Standard case with mild variability.
run_scenario(
    "standard",
    precipitation=np.linspace(0.8, 1.2, 5),
    runoff=np.linspace(0.2, 0.4, 5),
    storage=np.array([0.02, -0.01, 0.0, 0.01, 0.0]),
)

# Extreme stress-test allowing negative evaporation for diagnostic scenarios.
run_scenario(
    "extreme",
    precipitation=[0.4, 0.35, 0.3],
    runoff=[0.5, 0.45, 0.55],
    storage=[0.0, 0.0, -0.05],
)
