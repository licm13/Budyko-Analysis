"""Utility functions for Budyko-constrained machine learning workflows.

This module centralises the reusable scientific utilities that were
originally bundled inside the stand-alone Cheng-3D-Budyko scripts.  The
functions cover three main areas that are broadly useful across the
library:

* Hydrological diagnostics (baseflow separation, Budyko/BFC curves).
* Evaluation metrics commonly used in hydrological modelling.
* Convenience helpers for loading catchment level tabular data and
  resampling gridded datasets.

All functions are implemented with NumPy friendly signatures so that
callers can work with either arrays or ``pandas``/``xarray`` objects.  No
I/O paths are hard coded, which allows the helpers to be re-used in
unit tests and higher level workflows without modification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

Number = Union[int, float, np.ndarray]


# ---------------------------------------------------------------------------
# Budyko and BFC diagnostic curves
# ---------------------------------------------------------------------------

def budyko_runoff_ratio(
    precipitation: Number,
    potential_et: Number,
    alpha: Number,
) -> Number:
    """Return the Budyko runoff coefficient (``Q/P``) using the Fu-Zhang form.

    Parameters
    ----------
    precipitation:
        Annual or long-term mean precipitation [mm yr⁻¹].
    potential_et:
        Potential evapotranspiration [mm yr⁻¹].
    alpha:
        Budyko parameter that controls curve shape.

    Notes
    -----
    The Fu-Zhang expression is:

    .. math:: \frac{Q}{P} = -\frac{E_p}{P} + \left[1 + \left(\frac{E_p}{P}\right)^\alpha\right]^{1/\alpha}

    Results are clipped to the physically meaningful interval ``[0, 1]``.
    """

    precipitation = np.asarray(precipitation, dtype=float)
    potential_et = np.asarray(potential_et, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        aridity = potential_et / precipitation
        runoff_ratio = -aridity + (1.0 + np.power(aridity, alpha)) ** (1.0 / alpha)

    return np.clip(runoff_ratio, 0.0, 1.0)


def bfc_baseflow_ratio(
    precipitation: Number,
    potential_et: Number,
    alpha: Number,
    potential_baseflow: Number,
) -> Number:
    """Return the Budyko baseflow coefficient (``Q_b/P``).

    The formulation follows Cheng et al. (2021).  Returned values are
    clipped so that ``0 <= Q_b/P <= Q/P``.
    """

    precipitation = np.asarray(precipitation, dtype=float)
    potential_et = np.asarray(potential_et, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    potential_baseflow = np.asarray(potential_baseflow, dtype=float)

    aridity = potential_et / precipitation
    qbp_over_p = potential_baseflow / precipitation

    term_runoff = (1.0 + np.power(aridity, alpha)) ** (1.0 / alpha)
    term_baseflow = (1.0 + np.power(aridity + qbp_over_p, alpha)) ** (1.0 / alpha)
    baseflow_ratio = qbp_over_p + term_runoff - term_baseflow

    runoff_ratio = budyko_runoff_ratio(precipitation, potential_et, alpha)
    return np.clip(baseflow_ratio, 0.0, runoff_ratio)


# ---------------------------------------------------------------------------
# Parameter estimation helpers
# ---------------------------------------------------------------------------

def estimate_alpha(
    precipitation: np.ndarray,
    potential_et: np.ndarray,
    runoff: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """Estimate Budyko ``alpha`` parameters from observed series.

    A Newton iteration is used to invert the Fu-Zhang expression.  The
    solver is vectorised so callers can provide arrays of annual values.
    """

    precipitation = np.asarray(precipitation, dtype=float)
    potential_et = np.asarray(potential_et, dtype=float)
    runoff = np.asarray(runoff, dtype=float)

    runoff_ratio_obs = runoff / precipitation
    aridity = potential_et / precipitation
    alpha = np.full_like(runoff_ratio_obs, 2.0, dtype=float)

    for _ in range(max_iter):
        modeled = budyko_runoff_ratio(precipitation, potential_et, alpha)
        residual = modeled - runoff_ratio_obs

        if np.all(np.abs(residual) < tol):
            break

        delta = 1e-3
        derivative = (
            budyko_runoff_ratio(precipitation, potential_et, alpha + delta) - modeled
        ) / delta

        mask = np.abs(derivative) > 1e-10
        alpha[mask] -= residual[mask] / derivative[mask]
        alpha[~mask] = np.maximum(alpha[~mask], 1.0)
        alpha = np.clip(alpha, 1.0, 10.0)

    return alpha


def estimate_potential_baseflow(
    precipitation: np.ndarray,
    potential_et: np.ndarray,
    baseflow: np.ndarray,
    alpha: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """Estimate potential baseflow (``Q_b,p``) from observed records."""

    precipitation = np.asarray(precipitation, dtype=float)
    potential_et = np.asarray(potential_et, dtype=float)
    baseflow = np.asarray(baseflow, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    observed_ratio = baseflow / precipitation
    qbp = baseflow.astype(float)

    for _ in range(max_iter):
        modeled = bfc_baseflow_ratio(precipitation, potential_et, alpha, qbp)
        residual = modeled - observed_ratio

        if np.all(np.abs(residual) < tol):
            break

        delta = 1.0
        derivative = (
            bfc_baseflow_ratio(
                precipitation,
                potential_et,
                alpha,
                qbp + delta,
            )
            - modeled
        ) / delta

        mask = np.abs(derivative) > 1e-10
        qbp[mask] -= residual[mask] / derivative[mask]
        qbp[~mask] = np.maximum(qbp[~mask], 1.0)
        qbp = np.maximum(qbp, 1.0)

    return qbp


# ---------------------------------------------------------------------------
# Baseflow separation
# ---------------------------------------------------------------------------

def lyne_hollick_filter(
    discharge: Union[np.ndarray, Iterable[float]],
    *,
    alpha: float = 0.925,
    passes: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separate baseflow using the Lyne–Hollick digital filter."""

    flow = np.asarray(discharge, dtype=float)
    quick = flow.copy()

    for p in range(max(passes, 1)):
        if p % 2 == 0:
            for i in range(1, len(flow)):
                quick[i] = alpha * quick[i - 1] + (1 + alpha) / 2 * (
                    flow[i] - flow[i - 1]
                )
                quick[i] = np.clip(quick[i], 0.0, flow[i])
        else:
            for i in range(len(flow) - 2, -1, -1):
                quick[i] = alpha * quick[i + 1] + (1 + alpha) / 2 * (
                    flow[i] - flow[i + 1]
                )
                quick[i] = np.clip(quick[i], 0.0, flow[i])

    baseflow = flow - quick
    return baseflow, quick


# ---------------------------------------------------------------------------
# Model evaluation metrics
# ---------------------------------------------------------------------------

def _remove_nan_pairs(obs: np.ndarray, sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = ~(np.isnan(obs) | np.isnan(sim))
    return obs[mask], sim[mask]


def coefficient_of_determination(obs: np.ndarray, sim: np.ndarray) -> float:
    """Return the coefficient of determination (R²)."""

    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    obs, sim = _remove_nan_pairs(obs, sim)
    if obs.size == 0:
        return float("nan")

    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def root_mean_square_error(obs: np.ndarray, sim: np.ndarray) -> float:
    """Return the root mean square error (RMSE)."""

    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    obs, sim = _remove_nan_pairs(obs, sim)
    if obs.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def nash_sutcliffe_efficiency(obs: np.ndarray, sim: np.ndarray) -> float:
    """Return the Nash–Sutcliffe efficiency."""

    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    obs, sim = _remove_nan_pairs(obs, sim)
    if obs.size == 0:
        return float("nan")

    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - numerator / denominator if denominator > 0 else float("-inf")


def kling_gupta_efficiency(obs: np.ndarray, sim: np.ndarray) -> float:
    """Return the Kling–Gupta efficiency (KGE)."""

    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    obs, sim = _remove_nan_pairs(obs, sim)
    if obs.size == 0:
        return float("nan")

    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs) if np.std(obs) > 0 else 0.0
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else 0.0
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def resample_to_grid(
    data: xr.DataArray,
    *,
    resolution: float = 0.25,
    method: str = "bilinear",
) -> xr.DataArray:
    """Interpolate an ``xarray`` DataArray onto a regular lat/lon grid."""

    lat = np.arange(-90 + resolution / 2, 90, resolution)
    lon = np.arange(-180 + resolution / 2, 180, resolution)
    return data.interp(lat=lat, lon=lon, method=method)


def load_catchment_table(data_dir: Union[str, Path], catchment_id: str) -> pd.DataFrame:
    """Load a catchment CSV file that contains a ``date`` column."""

    data_dir = Path(data_dir)
    csv_path = data_dir / f"{catchment_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Catchment data file not found: {csv_path}")
    return pd.read_csv(csv_path, parse_dates=["date"])


__all__ = [
    "Number",
    "bfc_baseflow_ratio",
    "budyko_runoff_ratio",
    "estimate_alpha",
    "estimate_potential_baseflow",
    "lyne_hollick_filter",
    "coefficient_of_determination",
    "root_mean_square_error",
    "nash_sutcliffe_efficiency",
    "kling_gupta_efficiency",
    "resample_to_grid",
    "load_catchment_table",
]
