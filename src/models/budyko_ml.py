"""Machine learning models constrained by Budyko relationships."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold

from ..utils import (
    bfc_baseflow_ratio,
    budyko_runoff_ratio,
    coefficient_of_determination,
    root_mean_square_error,
)

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore


@dataclass
class CrossValidationResult:
    target: str
    fold: int
    train_r2: float
    train_rmse: float
    test_r2: float
    test_rmse: float


class BudykoConstrainedModel(BaseEstimator, RegressorMixin):
    """Regressor that predicts Budyko parameters with optional constraints.

    Parameters
    ----------
    constraint_weight:
        Strength of the Budyko constraint applied during prediction.  The
        value is clipped to ``[0, 1]`` and blends the unconstrained
        prediction with a physically valid alternative (``0`` disables the
        constraint, ``1`` enforces it fully).
    target_columns:
        Names of the targets to model.  The default assumes the processed
        dataset provides ``alpha`` and ``Qbp_mm_yr`` columns.
    base_model:
        Either ``"xgboost"`` (default when available) or ``"gbr"`` for the
        scikit-learn GradientBoostingRegressor.
    model_params:
        Optional estimator parameters overriding the defaults inspired by
        the Cheng-3D-Budyko replication scripts.
    n_splits:
        Default number of folds for :meth:`cross_validate`.
    random_state:
        Seed controlling any stochastic behaviour.
    """

    def __init__(
        self,
        *,
        constraint_weight: float = 0.5,
        target_columns: Sequence[str] = ("alpha", "Qbp_mm_yr"),
        base_model: Optional[str] = None,
        model_params: Optional[Dict[str, float]] = None,
        n_splits: int = 5,
        random_state: int = 42,
        alpha_bounds: Sequence[float] = (1.0, 10.0),
    ) -> None:
        self.constraint_weight = constraint_weight
        self.target_columns = tuple(target_columns)
        self.base_model = base_model
        self.model_params = model_params or {}
        self.n_splits = n_splits
        self.random_state = random_state
        self.alpha_bounds = alpha_bounds

    # ------------------------------------------------------------------
    # scikit-learn API
    # ------------------------------------------------------------------
    def fit(self, X: Mapping, y: Mapping) -> "BudykoConstrainedModel":
        X_df = self._to_dataframe(X)
        y_df = self._to_dataframe(y, expected_columns=self.target_columns)

        self.feature_names_in_ = list(X_df.columns)
        self.models_: Dict[str, object] = {}
        self.training_metrics_: Dict[str, Dict[str, float]] = {}

        for target in self.target_columns:
            estimator = self._make_estimator()
            estimator.fit(X_df, y_df[target])
            self.models_[target] = estimator

            preds = estimator.predict(X_df)
            self.training_metrics_[target] = {
                "r2": coefficient_of_determination(y_df[target].to_numpy(), preds),
                "rmse": root_mean_square_error(y_df[target].to_numpy(), preds),
            }

        return self

    def predict(
        self,
        X: Mapping,
        *,
        climate_inputs: Optional[Mapping[str, Iterable[float]]] = None,
    ) -> pd.DataFrame:
        if not hasattr(self, "models_"):
            raise RuntimeError("Model must be fitted before calling predict().")

        X_df = self._to_dataframe(X)
        alpha = self.models_[self.target_columns[0]].predict(X_df)
        qbp = self.models_[self.target_columns[1]].predict(X_df)

        result = pd.DataFrame(
            {
                self.target_columns[0]: alpha,
                self.target_columns[1]: qbp,
            },
            index=X_df.index,
        )

        climate_df = None
        if climate_inputs is not None:
            climate_df = self._to_dataframe(
                climate_inputs,
                expected_columns=("P_mm_yr", "Ep_mm_yr"),
            )

        constrained = self._apply_constraints(result, climate_df)
        if climate_df is not None:
            constrained = pd.concat([constrained, climate_df], axis=1)
            constrained["runoff_ratio"] = budyko_runoff_ratio(
                constrained["P_mm_yr"].to_numpy(),
                constrained["Ep_mm_yr"].to_numpy(),
                constrained[self.target_columns[0]].to_numpy(),
            )
            constrained["Q_mm_yr"] = constrained["runoff_ratio"] * constrained["P_mm_yr"]
            constrained["baseflow_ratio"] = bfc_baseflow_ratio(
                constrained["P_mm_yr"].to_numpy(),
                constrained["Ep_mm_yr"].to_numpy(),
                constrained[self.target_columns[0]].to_numpy(),
                constrained[self.target_columns[1]].to_numpy(),
            )
            constrained["Qb_mm_yr"] = constrained["baseflow_ratio"] * constrained["P_mm_yr"]
            constrained["Qq_mm_yr"] = constrained["Q_mm_yr"] - constrained["Qb_mm_yr"]

        return constrained

    # ------------------------------------------------------------------
    # Cross-validation helper
    # ------------------------------------------------------------------
    def cross_validate(
        self,
        X: Mapping,
        y: Mapping,
        *,
        climate_inputs: Optional[Mapping[str, Iterable[float]]] = None,
        n_splits: Optional[int] = None,
    ) -> pd.DataFrame:
        X_df = self._to_dataframe(X)
        y_df = self._to_dataframe(y, expected_columns=self.target_columns)
        climate_df = None
        if climate_inputs is not None:
            climate_df = self._to_dataframe(
                climate_inputs,
                expected_columns=("P_mm_yr", "Ep_mm_yr"),
            )

        splitter = KFold(n_splits=n_splits or self.n_splits, shuffle=True, random_state=self.random_state)
        records: list[CrossValidationResult] = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X_df), start=1):
            model = clone(self)
            model.fit(X_df.iloc[train_idx], y_df.iloc[train_idx])

            preds = model.predict(
                X_df.iloc[test_idx],
                climate_inputs=None if climate_df is None else climate_df.iloc[test_idx],
            )
            for target in self.target_columns:
                train_pred = model.models_[target].predict(X_df.iloc[train_idx])
                train_metrics = (
                    coefficient_of_determination(
                        y_df[target].iloc[train_idx].to_numpy(),
                        train_pred,
                    ),
                    root_mean_square_error(
                        y_df[target].iloc[train_idx].to_numpy(),
                        train_pred,
                    ),
                )
                test_metrics = (
                    coefficient_of_determination(
                        y_df[target].iloc[test_idx].to_numpy(),
                        preds[target].to_numpy(),
                    ),
                    root_mean_square_error(
                        y_df[target].iloc[test_idx].to_numpy(),
                        preds[target].to_numpy(),
                    ),
                )
                records.append(
                    CrossValidationResult(
                        target=target,
                        fold=fold,
                        train_r2=train_metrics[0],
                        train_rmse=train_metrics[1],
                        test_r2=test_metrics[0],
                        test_rmse=test_metrics[1],
                    )
                )

        return pd.DataFrame([r.__dict__ for r in records])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_estimator(self) -> object:
        model_name = self.base_model
        if model_name is None:
            model_name = "xgboost" if XGBRegressor is not None else "gbr"

        if model_name == "xgboost":
            if XGBRegressor is None:
                raise ImportError(
                    "xgboost is not installed; install it or set base_model='gbr'."
                )
            params = {
                "max_depth": 12,
                "learning_rate": 0.01,
                "subsample": 0.50,
                "colsample_bytree": 0.8,
                "n_estimators": 500,
                "objective": "reg:squarederror",
                "n_jobs": -1,
                "random_state": self.random_state,
            }
            params.update(self.model_params)
            return XGBRegressor(**params)

        if model_name == "gbr":
            from sklearn.ensemble import GradientBoostingRegressor

            params = {
                "max_depth": 3,
                "learning_rate": 0.05,
                "n_estimators": 400,
                "random_state": self.random_state,
            }
            params.update(self.model_params)
            return GradientBoostingRegressor(**params)

        raise ValueError(f"Unsupported base_model '{model_name}'.")

    def _apply_constraints(
        self,
        params_df: pd.DataFrame,
        climate_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        constrained = params_df.copy()
        constrained[self.target_columns[0]] = np.clip(
            constrained[self.target_columns[0]].to_numpy(),
            self.alpha_bounds[0],
            self.alpha_bounds[1],
        )
        constrained[self.target_columns[1]] = np.maximum(
            constrained[self.target_columns[1]].to_numpy(),
            0.0,
        )

        if climate_df is None:
            return constrained

        weight = float(np.clip(self.constraint_weight, 0.0, 1.0))
        if weight == 0.0:
            return constrained

        precip = climate_df["P_mm_yr"].to_numpy(dtype=float)
        pet = climate_df["Ep_mm_yr"].to_numpy(dtype=float)

        runoff_ratio = budyko_runoff_ratio(precip, pet, constrained[self.target_columns[0]].to_numpy())
        runoff_mm = runoff_ratio * precip

        qbp = constrained[self.target_columns[1]].to_numpy()
        qbp_constrained = np.minimum(qbp, runoff_mm)
        qbp_constrained = np.clip(qbp_constrained, 0.0, precip)
        constrained[self.target_columns[1]] = (
            weight * qbp_constrained + (1.0 - weight) * qbp
        )

        return constrained

    @staticmethod
    def _to_dataframe(
        data: Mapping,
        *,
        expected_columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, Mapping):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)

        if "index" in df.columns:
            df = df.set_index("index")

        if expected_columns is not None:
            missing = [col for col in expected_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            df = df[list(expected_columns)]

        return df


__all__ = ["BudykoConstrainedModel", "CrossValidationResult"]
