# src/budyko/water_balance.py
"""
水量平衡计算模块 (Budyko分析的基石)

水量平衡：P - Q = EA + ΔS
长时间尺度假设 ΔS≈0 ⇒ EA≈P-Q

Budyko指数：
- 干旱指数 IA = PET / P
- 蒸发指数 IE = EA / P = (P - Q) / P
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import warnings

import numpy as np
import pandas as pd


@dataclass
class WaterBalanceResults:
    """水量平衡计算结果数据类"""
    # 输入数据
    precipitation: np.ndarray                 # P
    runoff: np.ndarray                        # Q
    pet: np.ndarray                           # PET
    storage_change: Optional[np.ndarray]      # ΔS (可为None)

    # 核心结果
    actual_evaporation: np.ndarray            # EA
    aridity_index: np.ndarray                 # IA = PET/P
    evaporation_index: np.ndarray             # IE = EA/P
    runoff_coefficient: np.ndarray            # RC = Q/P

    # 质量标志（非默认，需放在默认字段前）
    data_quality_flags: np.ndarray

    # 可选/扩展
    closure_error: Optional[np.ndarray] = None
    storage_change_index: Optional[np.ndarray] = None
    actual_evaporation_extended: Optional[np.ndarray] = None
    evaporation_index_extended: Optional[np.ndarray] = None


class WaterBalanceCalculator:
    """
    水量平衡计算器
    - 基于径流Q计算EA
    - 计算IA、IE、RC
    - 可选：考虑ΔS和TWS
    - 基础数据质量控制
    """

    def __init__(self,
                 allow_negative_ea: bool = False,
                 min_precipitation: float = 100.0,
                 max_runoff_ratio: float = 0.95,
                 water_year_start_month: int = 1) -> None:
        self.allow_negative_ea = allow_negative_ea
        self.min_precipitation = float(min_precipitation)
        self.max_runoff_ratio = float(max_runoff_ratio)
        self.water_year_start_month = int(water_year_start_month)

    def calculate_budyko_indices(self,
                                 P: np.ndarray,
                                 Q: np.ndarray,
                                 PET: np.ndarray,
                                 delta_S: Optional[np.ndarray] = None,
                                 TWS: Optional[np.ndarray] = None) -> WaterBalanceResults:
        """
        计算水量平衡与Budyko指数
        """
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float)
        PET = np.asarray(PET, dtype=float)

        if not (len(P) == len(Q) == len(PET)):
            raise ValueError("P, Q, PET 长度必须一致")

        include_storage = delta_S is not None
        if include_storage:
            delta_S = np.asarray(delta_S, dtype=float)
            if len(delta_S) != len(P):
                raise ValueError("delta_S 长度必须与 P 相同")
            EA = P - Q - delta_S
        else:
            EA = P - Q

        if not self.allow_negative_ea:
            EA = np.maximum(EA, 0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            IA = np.where(P > 0, PET / P, np.nan)
            IE = np.where(P > 0, EA / P, np.nan)
            RC = np.where(P > 0, Q / P, np.nan)

        # 简单的SCI示例（标准化TWS），如未提供则为None
        SCI = None
        if TWS is not None:
            TWS = np.asarray(TWS, dtype=float)
            if len(TWS) != len(P):
                raise ValueError("TWS 长度必须与 P 相同")
            mu = np.nanmean(TWS)
            sigma = np.nanstd(TWS)
            if sigma == 0 or np.isnan(sigma):
                SCI = np.full_like(P, np.nan, dtype=float)
            else:
                SCI = (TWS - mu) / sigma

        # 数据质量控制
        quality_flags = self._quality_control(P, Q, PET, EA, IA, IE)

        # 水量平衡闭合误差（若未考虑ΔS则按0处理）
        closure_error = P - Q - EA - (delta_S if include_storage else np.zeros_like(P))

        return WaterBalanceResults(
            precipitation=P,
            runoff=Q,
            pet=PET,
            storage_change=delta_S if include_storage else None,
            actual_evaporation=EA,
            aridity_index=IA,
            evaporation_index=IE,
            runoff_coefficient=RC,
            data_quality_flags=quality_flags,
            closure_error=closure_error,
            storage_change_index=SCI,
            actual_evaporation_extended=None,
            evaporation_index_extended=None
        )

    def _quality_control(self,
                         P: np.ndarray,
                         Q: np.ndarray,
                         PET: np.ndarray,
                         EA: np.ndarray,
                         IA: np.ndarray,
                         IE: np.ndarray) -> np.ndarray:
        """
        简单数据质量控制：
        - P < min_precipitation → flag=2
        - Q/P > max_runoff_ratio → flag>=1
        - 任一比值为NaN → flag>=1
        """
        flags = np.zeros(len(P), dtype=int)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # 1) 过小的降水
            flags[(P < self.min_precipitation) | ~np.isfinite(P)] = np.maximum(
                flags[(P < self.min_precipitation) | ~np.isfinite(P)],
                2
            )

            # 2) 过大的径流系数
            with np.errstate(divide='ignore', invalid='ignore'):
                rc = np.where(P > 0, Q / P, np.nan)
            flags[(rc > self.max_runoff_ratio)] = np.maximum(
                flags[(rc > self.max_runoff_ratio)],
                1
            )

            # 3) 比值的NaN/Inf
            bad_ratio = ~np.isfinite(IA) | ~np.isfinite(IE)
            flags[bad_ratio] = np.maximum(flags[bad_ratio], 1)

            # 4) EA为负（如不允许）
            if not self.allow_negative_ea:
                flags[EA < 0] = np.maximum(flags[EA < 0], 1)

        return flags

    def aggregate_to_periods(self,
                             data: pd.DataFrame,
                             period_length: int = 20) -> pd.DataFrame:
        """
        将年度数据聚合为多年时段
        要求 data 至少包含列: year, P, Q, PET
        生成 [start_year, end_year, P_sum, Q_sum, PET_sum, IA_mean, IE_mean, RC_mean]
        """
        required = {"year", "P", "Q", "PET"}
        if not required.issubset(set(data.columns)):
            missing = required - set(data.columns)
            raise ValueError(f"缺少列: {missing}")

        years = np.asarray(data["year"], dtype=int)
        min_year = int(np.min(years))
        max_year = int(np.max(years))

        periods = []
        for start in range(min_year, max_year - period_length + 2):
            end = start + period_length - 1
            mask = (years >= start) & (years <= end)
            if not np.any(mask):
                continue
            P = data.loc[mask, "P"].to_numpy(float)
            Q = data.loc[mask, "Q"].to_numpy(float)
            PET = data.loc[mask, "PET"].to_numpy(float)

            EA = P - Q
            with np.errstate(divide='ignore', invalid='ignore'):
                IA = np.where(P > 0, PET / P, np.nan)
                IE = np.where(P > 0, EA / P, np.nan)
                RC = np.where(P > 0, Q / P, np.nan)

            periods.append({
                "start_year": start,
                "end_year": end,
                "P_sum": float(np.nansum(P)),
                "Q_sum": float(np.nansum(Q)),
                "PET_sum": float(np.nansum(PET)),
                "IA_mean": float(np.nanmean(IA)),
                "IE_mean": float(np.nanmean(IE)),
                "RC_mean": float(np.nanmean(RC)),
            })

        return pd.DataFrame(periods)

    def prepare_annual_dataframe(self,
                                 dates: pd.DatetimeIndex,
                                 **kwargs: np.ndarray) -> pd.DataFrame:
        """
        将时间序列数据按水文年聚合为年度数据
        示例仅提供最小实现：对 P/Q/PET 做年累积，其它变量做年均值
        """
        df = pd.DataFrame(kwargs, index=dates)
        if "P" not in df.columns or "Q" not in df.columns or "PET" not in df.columns:
            return df  # 非核心字段不强制处理

        # 定义水文年
        if self.water_year_start_month == 1:
            df["water_year"] = df.index.year
        else:
            offset = self.water_year_start_month
            # 若月份 >= 起始月，则属于当年水文年，否则属于上一年水文年
            df["water_year"] = np.where(df.index.month >= offset,
                                        df.index.year,
                                        df.index.year - 1)

        # 聚合：累积型变量(P/Q/PET)求和，其余求均值
        agg = {}
        for col in df.columns:
            if col in ("water_year",):
                continue
            if col in ("P", "Q", "PET"):
                agg[col] = "sum"
            else:
                agg[col] = "mean"

        annual = df.groupby("water_year").agg(agg).reset_index().rename(
            columns={"water_year": "year"}
        )
        return annual


class RunoffAnalyzer:
    """
    径流数据诊断工具
    - 基础统计
    - 径流系数统计（若提供P）
    - 简单数据质量指标
    """

    def diagnose_runoff_data(self,
                             Q: np.ndarray,
                             P: Optional[np.ndarray] = None) -> Dict:
        Q = np.asarray(Q, dtype=float)
        result: Dict = {}

        # 基础统计
        q_valid = Q[np.isfinite(Q)]
        result["basic_stats"] = {
            "mean_runoff": float(np.nanmean(Q)),
            "median_runoff": float(np.nanmedian(Q)),
            "std_runoff": float(np.nanstd(Q)),
            "min_runoff": float(np.nanmin(Q)),
            "max_runoff": float(np.nanmax(Q)),
            "missing_count": int(np.size(Q) - q_valid.size)
        }

        # 数据质量
        pct_missing = 100.0 * (np.size(Q) - q_valid.size) / max(1, np.size(Q))
        pct_nonpositive = 100.0 * np.sum(Q <= 0) / max(1, np.size(Q))
        result["data_quality"] = {
            "pct_missing": float(pct_missing),
            "pct_nonpositive": float(pct_nonpositive)
        }

        # 径流系数（如果提供P）
        if P is not None:
            P = np.asarray(P, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                rc_series = np.where(P > 0, Q / P, np.nan)
            result["runoff_coefficient"] = {
                "series": rc_series,
                "mean": float(np.nanmean(rc_series)),
                "std": float(np.nanstd(rc_series))
            }
        else:
            result["runoff_coefficient"] = {
                "series": None,
                "mean": float("nan"),
                "std": float("nan")
            }

        return result