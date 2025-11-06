"""High level workflow for Budyko-constrained machine learning models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import pandas as pd

from ..data_processing import BudykoMLPreprocessor
from ..models import BudykoConstrainedModel


@dataclass
class BudykoMLWorkflowResult:
    model: BudykoConstrainedModel
    training_table: pd.DataFrame
    predictions: pd.DataFrame
    qc_log: pd.DataFrame
    cv_metrics: Optional[pd.DataFrame] = None


def run_budyko_ml_workflow(
    catchment_series: Mapping[str, pd.DataFrame],
    catchment_attributes: Optional[pd.DataFrame] = None,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    climate_columns: Sequence[str] = ("P_mm_yr", "Ep_mm_yr"),
    preprocessor: Optional[BudykoMLPreprocessor] = None,
    model: Optional[BudykoConstrainedModel] = None,
    perform_cross_validation: bool = True,
) -> BudykoMLWorkflowResult:
    """Execute the full Budyko constrained ML pipeline."""

    preprocessor = preprocessor or BudykoMLPreprocessor()
    model = model or BudykoConstrainedModel()

    training_table = preprocessor.prepare_training_dataset(
        catchment_series,
        catchment_attributes,
    )

    if training_table.empty:
        raise ValueError("No catchments passed quality control; nothing to train.")

    target_columns = list(model.target_columns)

    if feature_columns is None:
        excluded = set(target_columns) | set(climate_columns) | {
            "catchment_id",
            "passed_qc",
            "qc_reason",
            "Q_mm_yr",
            "Qb_mm_yr",
            "Qq_mm_yr",
            "runoff_ratio",
            "baseflow_ratio",
        }
        feature_columns = [
            col
            for col in training_table.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(training_table[col])
        ]
        if not feature_columns:
            raise ValueError(
                "Unable to infer feature columns automatically; supply feature_columns explicitly."
            )

    X = training_table[feature_columns]
    y = training_table[target_columns]

    climate_df = None
    if all(col in training_table.columns for col in climate_columns):
        climate_df = training_table[list(climate_columns)]

    model.fit(X, y)
    predictions = model.predict(X, climate_inputs=climate_df)

    qc_log = pd.DataFrame([qc.__dict__ for qc in preprocessor.quality_control_log])

    cv_metrics = None
    if perform_cross_validation:
        cv_metrics = model.cross_validate(
            X,
            y,
            climate_inputs=climate_df,
            n_splits=model.n_splits,
        )

    return BudykoMLWorkflowResult(
        model=model,
        training_table=training_table,
        predictions=predictions,
        qc_log=qc_log,
        cv_metrics=cv_metrics,
    )


__all__ = ["BudykoMLWorkflowResult", "run_budyko_ml_workflow"]
