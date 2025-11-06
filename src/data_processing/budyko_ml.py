"""Data preparation utilities for Budyko-constrained machine learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from ..utils import lyne_hollick_filter


@dataclass
class BudykoMLColumnMap:
    """Column names expected in the raw catchment time-series table."""

    discharge: str = "discharge_mm_day"
    precipitation: str = "precip_mm_day"
    potential_et: str = "pet_mm_day"
    baseflow: Optional[str] = None
    area: Optional[str] = "area_km2"


@dataclass
class BudykoMLPreprocessorConfig:
    """Configuration controlling quality control thresholds."""

    min_record_length_years: int = 10
    max_missing_rate: float = 0.2
    min_catchment_area: float = 50.0
    max_catchment_area: float = 5000.0
    water_balance_threshold: float = 0.1
    lyne_hollick_alpha: float = 0.925
    lyne_hollick_passes: int = 3


@dataclass
class QCResult:
    catchment_id: str
    passed: bool
    reason: str


class BudykoMLPreprocessor:
    """Prepare machine-learning ready Budyko training tables."""

    def __init__(
        self,
        config: Optional[BudykoMLPreprocessorConfig] = None,
        column_map: Optional[BudykoMLColumnMap] = None,
    ) -> None:
        self.config = config or BudykoMLPreprocessorConfig()
        self.column_map = column_map or BudykoMLColumnMap()
        self.quality_control_log: list[QCResult] = []

    # ------------------------------------------------------------------
    # Public high level API
    # ------------------------------------------------------------------
    def prepare_training_dataset(
        self,
        catchment_series: Mapping[str, pd.DataFrame],
        catchment_attributes: Optional[pd.DataFrame] = None,
        *,
        keep_qc_failures: bool = False,
    ) -> pd.DataFrame:
        """Aggregate raw catchment data into a machine learning table.

        Parameters
        ----------
        catchment_series:
            Mapping between ``catchment_id`` and daily time-series dataframes.
        catchment_attributes:
            Optional dataframe containing static descriptors per catchment.
        keep_qc_failures:
            Whether to include rows that fail quality control (tagged with
            ``passed_qc``) instead of discarding them entirely.
        """

        summaries: list[dict] = []
        self.quality_control_log.clear()

        for catchment_id, df in catchment_series.items():
            result = self._quality_control(df, catchment_id)
            self.quality_control_log.append(result)

            if not result.passed and not keep_qc_failures:
                continue

            summary = self._summarise_catchment(df, catchment_id)
            if summary is None:
                continue

            summary["catchment_id"] = catchment_id
            summary["passed_qc"] = result.passed
            summary["qc_reason"] = result.reason
            summaries.append(summary)

        if not summaries:
            return pd.DataFrame(columns=["catchment_id"])

        dataset = pd.DataFrame(summaries)

        if catchment_attributes is not None and not catchment_attributes.empty:
            dataset = dataset.merge(
                catchment_attributes,
                on="catchment_id",
                how="left",
            )

        return dataset

    # ------------------------------------------------------------------
    # Quality control
    # ------------------------------------------------------------------
    def _quality_control(self, df: pd.DataFrame, catchment_id: str) -> QCResult:
        required = [
            self.column_map.discharge,
            self.column_map.precipitation,
            self.column_map.potential_et,
        ]

        missing_columns = [col for col in required if col not in df.columns]
        if missing_columns:
            return QCResult(
                catchment_id,
                False,
                f"missing required columns: {missing_columns}",
            )

        df = df.dropna(subset=["date"])
        if df.empty:
            return QCResult(catchment_id, False, "no valid records")

        df = df.sort_values("date")
        years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
        if years < self.config.min_record_length_years:
            return QCResult(
                catchment_id,
                False,
                f"record shorter than {self.config.min_record_length_years} years",
            )

        missing_rate = df[required].isna().mean().mean()
        if missing_rate > self.config.max_missing_rate:
            return QCResult(
                catchment_id,
                False,
                f"missing rate {missing_rate:.1%} exceeds threshold",
            )

        # Seasonal coverage check
        monthly_counts = df.groupby(df["date"].dt.month)[required[0]].count()
        if (monthly_counts == 0).any():
            return QCResult(catchment_id, False, "seasonal gaps detected")

        if self.column_map.area and self.column_map.area in df.columns:
            area = float(df[self.column_map.area].iloc[0])
            if not (
                self.config.min_catchment_area
                <= area
                <= self.config.max_catchment_area
            ):
                return QCResult(catchment_id, False, "catchment area outside bounds")

        return QCResult(catchment_id, True, "passed")

    # ------------------------------------------------------------------
    # Catchment summarisation
    # ------------------------------------------------------------------
    def _summarise_catchment(
        self,
        df: pd.DataFrame,
        catchment_id: str,
    ) -> Optional[Dict[str, float]]:
        df = df.copy()
        df = df.sort_values("date")

        baseflow_column = self.column_map.baseflow
        if baseflow_column and baseflow_column not in df.columns:
            baseflow_column = None  # Force computation below

        if baseflow_column:
            pass  # baseflow column exists, will use it below
        else:
            baseflow, quick = lyne_hollick_filter(
                df[self.column_map.discharge].to_numpy(dtype=float),
                alpha=self.config.lyne_hollick_alpha,
                passes=self.config.lyne_hollick_passes,
            )
            df["baseflow_tmp"] = baseflow
            df["quickflow_tmp"] = quick
            baseflow_column = "baseflow_tmp"
        if baseflow_column != "baseflow_tmp":
            df["quickflow_tmp"] = (
                df[self.column_map.discharge].to_numpy(dtype=float) -
                df[baseflow_column].to_numpy(dtype=float)
            )

        resampled = (
            df.set_index("date")
            [[self.column_map.precipitation,
              self.column_map.potential_et,
              self.column_map.discharge,
              baseflow_column,
              "quickflow_tmp"]]
            .resample("A")
            .sum(min_count=200)
        )

        resampled = resampled.dropna()
        if resampled.empty:
            return None

        mean_values = resampled.mean()

        runoff_mm_yr = float(mean_values[self.column_map.discharge])
        baseflow_mm_yr = float(mean_values[baseflow_column])
        quickflow_mm_yr = float(mean_values["quickflow_tmp"])
        precip_mm_yr = float(mean_values[self.column_map.precipitation])
        pet_mm_yr = float(mean_values[self.column_map.potential_et])

        if precip_mm_yr <= 0 or runoff_mm_yr < 0:
            return None

        from ..budyko import (
            cheng_baseflow_ratio,
            fu_zhang_runoff_ratio,
            invert_cheng_qbp,
            invert_fu_zhang_alpha,
        )

        alpha_series = invert_fu_zhang_alpha(
            resampled[self.column_map.precipitation].to_numpy(),
            resampled[self.column_map.potential_et].to_numpy(),
            resampled[self.column_map.discharge].to_numpy(),
        )
        alpha = float(np.mean(alpha_series))

        qbp_values = invert_cheng_qbp(
            resampled[self.column_map.precipitation].to_numpy(),
            resampled[self.column_map.potential_et].to_numpy(),
            resampled[baseflow_column].to_numpy(),
            alpha_series,
        )
        qbp_mm_yr = float(np.mean(qbp_values))

        runoff_ratio = fu_zhang_runoff_ratio(
            precip_mm_yr,
            pet_mm_yr,
            alpha,
        )
        baseflow_ratio = cheng_baseflow_ratio(
            precip_mm_yr,
            pet_mm_yr,
            alpha,
            qbp_mm_yr,
        )

        return {
            "P_mm_yr": precip_mm_yr,
            "Ep_mm_yr": pet_mm_yr,
            "Q_mm_yr": runoff_mm_yr,
            "Qb_mm_yr": baseflow_mm_yr,
            "Qq_mm_yr": quickflow_mm_yr,
            "alpha": float(alpha),
            "Qbp_mm_yr": qbp_mm_yr,
            "runoff_ratio": float(runoff_ratio),
            "baseflow_ratio": float(baseflow_ratio),
        }


__all__ = [
    "BudykoMLPreprocessor",
    "BudykoMLPreprocessorConfig",
    "BudykoMLColumnMap",
    "QCResult",
]
