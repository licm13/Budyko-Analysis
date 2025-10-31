# src/budyko/water_balance.py
"""
水量平衡计算模块

核心原理：
---------
径流数据(Q)是Budyko分析的基石！

水量平衡方程：
P - Q = EA + ΔS

其中：
- P: 降水 (mm)
- Q: 径流 (mm) - 观测数据，核心！
- EA: 实际蒸发 (mm) - 无法直接测量
- ΔS: 储量变化 (mm) - 可用GRACE估算

长时间尺度假设：
ΔS ≈ 0，因此 EA ≈ P - Q

Budyko指数：
- 干旱指数 IA = PET/P （X轴）
- 蒸发指数 IE = EA/P = (P-Q)/P （Y轴）

**没有径流Q，我们无法计算IE，Budyko分析无法进行！**

References
----------
- Budyko (1974). Climate and Life.
- Ibrahim et al. (2023). Water-Energy Balance Framework.
- Wang & Dickinson (2012). A review of global terrestrial evapotranspiration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class WaterBalanceResults:
    """水量平衡计算结果数据类"""
    # 输入数据
    precipitation: np.ndarray
    runoff: np.ndarray
    pet: np.ndarray
    storage_change: Optional[np.ndarray]

    # 计算结果
    actual_evaporation: np.ndarray
    aridity_index: np.ndarray
    evaporation_index: np.ndarray
    runoff_coefficient: np.ndarray

    # 质量控制标志
    data_quality: np.ndarray
    closure_error: Optional[np.ndarray]


class WaterBalanceCalculator:
    """
    水量平衡计算器

    核心功能：
    1. 基于径流Q计算实际蒸发EA
    2. 计算Budyko指数（IA, IE）
    3. 水量平衡闭合检验
    4. 数据质量控制
    """

    def __init__(self,
                 allow_negative_ea: bool = False,
                 min_precipitation: float = 100,
                 max_runoff_ratio: float = 0.95):
        """
        初始化计算器

        Parameters
        ----------
        allow_negative_ea : bool
            是否允许负的EA值（某些情况下Q > P）
        min_precipitation : float
            最小有效降水量 (mm/yr)，低于此值的年份可能不可靠
        max_runoff_ratio : float
            最大径流系数 (Q/P)，超过此值可能有数据问题
        """
        self.allow_negative_ea = allow_negative_ea
        self.min_precipitation = min_precipitation
        self.max_runoff_ratio = max_runoff_ratio

    def calculate_budyko_indices(self,
                                 P: np.ndarray,
                                 Q: np.ndarray,
                                 PET: np.ndarray,
                                 delta_S: Optional[np.ndarray] = None) -> WaterBalanceResults:
        """
        计算水量平衡和Budyko指数

        **径流Q是核心输入，决定实际蒸发EA！**

        Parameters
        ----------
        P : np.ndarray
            降水 (mm)
        Q : np.ndarray
            径流 (mm) - 观测数据，是Budyko分析的基石！
        PET : np.ndarray
            潜在蒸发 (mm)
        delta_S : np.ndarray, optional
            储量变化 (mm)，可用GRACE数据提供
            如提供，则 EA = P - Q - ΔS
            如不提供，则 EA = P - Q （假设ΔS≈0）

        Returns
        -------
        WaterBalanceResults
            包含所有计算结果的对象
        """
        # 确保输入为numpy数组
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float)
        PET = np.asarray(PET, dtype=float)

        # 检查数组长度一致性
        if not (len(P) == len(Q) == len(PET)):
            raise ValueError("P, Q, PET数组长度必须一致")

        # ============ 核心计算：实际蒸发EA（基于径流Q） ============
        if delta_S is not None:
            # 考虑储量变化：EA = P - Q - ΔS
            delta_S = np.asarray(delta_S, dtype=float)
            if len(delta_S) != len(P):
                raise ValueError("delta_S长度必须与P一致")
            EA = P - Q - delta_S
            include_storage = True
        else:
            # 长时间尺度假设：EA = P - Q
            EA = P - Q
            delta_S = None
            include_storage = False

        # 处理负EA值
        if not self.allow_negative_ea:
            negative_mask = EA < 0
            if np.any(negative_mask):
                n_negative = np.sum(negative_mask)
                warnings.warn(
                    f"发现 {n_negative} 个负EA值 (Q > P)，已设为0。"
                    f"这可能表明：1) 数据误差, 2) 地下水补给, 3) 跨流域调水"
                )
                EA[negative_mask] = 0

        # ============ 计算Budyko指数 ============

        # 干旱指数 IA = PET/P （X轴）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            IA = PET / P

        # 蒸发指数 IE = EA/P （Y轴）- 由径流Q决定！
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            IE = EA / P

        # 径流系数 RC = Q/P
        RC = Q / P

        # ============ 数据质量控制 ============
        quality_flags = self._quality_control(P, Q, PET, EA, IA, IE, RC)

        # ============ 水量平衡闭合误差 ============
        if include_storage:
            # 闭合误差：水量平衡方程的残差
            closure_error = P - Q - EA - delta_S
        else:
            closure_error = None

        # 返回结果
        return WaterBalanceResults(
            precipitation=P,
            runoff=Q,
            pet=PET,
            storage_change=delta_S,
            actual_evaporation=EA,
            aridity_index=IA,
            evaporation_index=IE,
            runoff_coefficient=RC,
            data_quality=quality_flags,
            closure_error=closure_error
        )

    def _quality_control(self,
                        P: np.ndarray,
                        Q: np.ndarray,
                        PET: np.ndarray,
                        EA: np.ndarray,
                        IA: np.ndarray,
                        IE: np.ndarray,
                        RC: np.ndarray) -> np.ndarray:
        """
        数据质量控制

        Returns
        -------
        np.ndarray
            质量标志：0=良好, 1=可疑, 2=差
        """
        flags = np.zeros(len(P), dtype=int)

        # 检查1：降水太小
        flags[P < self.min_precipitation] = 2

        # 检查2：径流系数过大（Q > 0.95*P）
        flags[RC > self.max_runoff_ratio] = 2

        # 检查3：蒸发指数超出物理范围
        flags[(IE < 0) | (IE > 1.2)] = 1

        # 检查4：干旱指数异常
        flags[(IA < 0) | (IA > 10)] = 1

        # 检查5：PET < EA（违反物理规律，除非有平流）
        flags[PET < EA * 0.8] = 1

        # 检查6：NaN值
        nan_mask = (np.isnan(P) | np.isnan(Q) | np.isnan(PET) |
                   np.isnan(EA) | np.isnan(IA) | np.isnan(IE))
        flags[nan_mask] = 2

        return flags

    def aggregate_to_periods(self,
                            results: WaterBalanceResults,
                            dates: pd.DatetimeIndex,
                            period_length: int = 20) -> pd.DataFrame:
        """
        将年度数据聚合为多年时段（用于Ibrahim方法）

        Parameters
        ----------
        results : WaterBalanceResults
            水量平衡结果
        dates : pd.DatetimeIndex
            日期索引
        period_length : int
            时段长度（年），默认20年

        Returns
        -------
        pd.DataFrame
            聚合后的时段数据
        """
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'P': results.precipitation,
            'Q': results.runoff,
            'PET': results.pet,
            'EA': results.actual_evaporation,
            'IA': results.aridity_index,
            'IE': results.evaporation_index,
            'quality': results.data_quality
        })

        # 提取年份
        df['year'] = df['date'].dt.year

        # 过滤低质量数据
        df_good = df[df['quality'] <= 1].copy()

        # 创建时段
        min_year = df_good['year'].min()
        max_year = df_good['year'].max()

        periods = []
        period_id = 0

        for start_year in range(min_year, max_year - period_length + 2):
            end_year = start_year + period_length - 1

            # 选择时段数据
            period_data = df_good[
                (df_good['year'] >= start_year) &
                (df_good['year'] <= end_year)
            ]

            if len(period_data) >= period_length * 0.8:  # 至少80%数据
                periods.append({
                    'period_id': period_id,
                    'start_year': start_year,
                    'end_year': end_year,
                    'n_years': len(period_data),
                    'P_mean': period_data['P'].mean(),
                    'Q_mean': period_data['Q'].mean(),
                    'PET_mean': period_data['PET'].mean(),
                    'EA_mean': period_data['EA'].mean(),
                    'IA_mean': period_data['IA'].mean(),
                    'IE_mean': period_data['IE'].mean(),
                    'RC_mean': (period_data['Q'] / period_data['P']).mean()
                })
                period_id += 1

        return pd.DataFrame(periods)

    def calculate_with_grace(self,
                            P: np.ndarray,
                            Q: np.ndarray,
                            PET: np.ndarray,
                            TWS: np.ndarray,
                            time_resolution: str = 'monthly') -> Tuple[WaterBalanceResults, WaterBalanceResults]:
        """
        对比有无GRACE TWS数据的水量平衡

        用于检验储量变化假设的影响

        Parameters
        ----------
        P, Q, PET : np.ndarray
            降水、径流、PET
        TWS : np.ndarray
            陆地水储量 (mm)，来自GRACE
        time_resolution : str
            时间分辨率，'monthly' 或 'annual'

        Returns
        -------
        Tuple[WaterBalanceResults, WaterBalanceResults]
            (无TWS结果, 有TWS结果)
        """
        # 计算储量变化 ΔS
        if time_resolution == 'monthly':
            delta_S = np.diff(TWS, prepend=TWS[0])
        elif time_resolution == 'annual':
            delta_S = np.diff(TWS, prepend=TWS[0])
        else:
            raise ValueError(f"Unknown time_resolution: {time_resolution}")

        # 无TWS：EA = P - Q
        results_no_storage = self.calculate_budyko_indices(P, Q, PET, delta_S=None)

        # 有TWS：EA = P - Q - ΔS
        results_with_storage = self.calculate_budyko_indices(P, Q, PET, delta_S=delta_S)

        return results_no_storage, results_with_storage


class RunoffAnalyzer:
    """
    径流数据分析器

    专门用于分析径流观测数据的质量和特征
    """

    def __init__(self):
        pass

    def diagnose_runoff_data(self,
                            Q: np.ndarray,
                            P: np.ndarray,
                            dates: pd.DatetimeIndex = None) -> Dict:
        """
        诊断径流数据质量

        Parameters
        ----------
        Q : np.ndarray
            径流数据 (mm)
        P : np.ndarray
            降水数据 (mm)
        dates : pd.DatetimeIndex, optional
            日期索引

        Returns
        -------
        Dict
            诊断结果
        """
        Q = np.asarray(Q)
        P = np.asarray(P)

        diagnosis = {}

        # 基本统计
        diagnosis['basic_stats'] = {
            'mean_runoff': np.nanmean(Q),
            'std_runoff': np.nanstd(Q),
            'min_runoff': np.nanmin(Q),
            'max_runoff': np.nanmax(Q),
            'cv_runoff': np.nanstd(Q) / np.nanmean(Q)
        }

        # 径流系数
        RC = Q / P
        diagnosis['runoff_coefficient'] = {
            'mean': np.nanmean(RC),
            'std': np.nanstd(RC),
            'min': np.nanmin(RC),
            'max': np.nanmax(RC)
        }

        # 数据质量
        n_total = len(Q)
        n_missing = np.sum(np.isnan(Q))
        n_negative = np.sum(Q < 0)
        n_zero = np.sum(Q == 0)
        n_extreme = np.sum(Q > P)  # 径流大于降水

        diagnosis['data_quality'] = {
            'n_total': n_total,
            'n_missing': n_missing,
            'pct_missing': 100 * n_missing / n_total,
            'n_negative': n_negative,
            'n_zero': n_zero,
            'n_extreme': n_extreme,
            'pct_extreme': 100 * n_extreme / n_total
        }

        # 时间连续性（如果提供了日期）
        if dates is not None:
            diagnosis['temporal_continuity'] = self._check_temporal_continuity(dates, Q)

        # 季节性
        if dates is not None:
            diagnosis['seasonality'] = self._analyze_seasonality(dates, Q)

        # 警告
        warnings_list = []
        if diagnosis['data_quality']['pct_missing'] > 10:
            warnings_list.append(f"缺失率过高: {diagnosis['data_quality']['pct_missing']:.1f}%")
        if diagnosis['data_quality']['pct_extreme'] > 5:
            warnings_list.append(f"异常值过多: {diagnosis['data_quality']['pct_extreme']:.1f}%")
        if diagnosis['runoff_coefficient']['mean'] > 0.8:
            warnings_list.append(f"径流系数过大: {diagnosis['runoff_coefficient']['mean']:.2f}")

        diagnosis['warnings'] = warnings_list

        return diagnosis

    def _check_temporal_continuity(self,
                                   dates: pd.DatetimeIndex,
                                   Q: np.ndarray) -> Dict:
        """检查时间连续性"""
        # 找出数据缺口
        date_diff = np.diff(dates.to_julian_date())
        expected_diff = np.median(date_diff)

        gaps = np.where(date_diff > expected_diff * 2)[0]

        return {
            'n_gaps': len(gaps),
            'longest_gap_days': int(np.max(date_diff)) if len(date_diff) > 0 else 0,
            'is_continuous': len(gaps) == 0
        }

    def _analyze_seasonality(self,
                            dates: pd.DatetimeIndex,
                            Q: np.ndarray) -> Dict:
        """分析季节性"""
        df = pd.DataFrame({'date': dates, 'Q': Q})
        df['month'] = df['date'].dt.month

        monthly_mean = df.groupby('month')['Q'].mean()

        return {
            'max_month': int(monthly_mean.idxmax()),
            'min_month': int(monthly_mean.idxmin()),
            'seasonality_index': (monthly_mean.max() - monthly_mean.min()) / monthly_mean.mean()
        }


# 使用示例
if __name__ == '__main__':
    # 创建示例数据
    n_years = 30
    np.random.seed(42)

    # 模拟流域数据
    P = 800 + 200 * np.random.randn(n_years)  # 降水 (mm/yr)
    PET = 1000 + 150 * np.random.randn(n_years)  # PET (mm/yr)

    # 模拟径流（关键！）
    # RC ≈ 0.3，加上噪声
    Q = 0.3 * P + 50 * np.random.randn(n_years)
    Q = np.maximum(Q, 0)  # 确保非负

    # 水量平衡计算
    wb_calc = WaterBalanceCalculator()

    results = wb_calc.calculate_budyko_indices(P, Q, PET)

    print("======= 水量平衡计算结果 =======\n")
    print("基本统计:")
    print(f"  平均降水 P: {np.mean(P):.1f} mm/yr")
    print(f"  平均径流 Q: {np.mean(Q):.1f} mm/yr")
    print(f"  平均PET: {np.mean(PET):.1f} mm/yr")
    print(f"  平均实际蒸发 EA: {np.mean(results.actual_evaporation):.1f} mm/yr")

    print("\nBudyko指数:")
    print(f"  干旱指数 IA (PET/P): {np.mean(results.aridity_index):.2f}")
    print(f"  蒸发指数 IE (EA/P): {np.mean(results.evaporation_index):.2f}")
    print(f"  径流系数 RC (Q/P): {np.mean(results.runoff_coefficient):.2f}")

    print("\n数据质量:")
    quality_counts = np.bincount(results.data_quality)
    print(f"  良好: {quality_counts[0] if len(quality_counts) > 0 else 0}")
    print(f"  可疑: {quality_counts[1] if len(quality_counts) > 1 else 0}")
    print(f"  差: {quality_counts[2] if len(quality_counts) > 2 else 0}")

    # 径流诊断
    print("\n======= 径流数据诊断 =======\n")
    analyzer = RunoffAnalyzer()
    diagnosis = analyzer.diagnose_runoff_data(Q, P)

    print("径流系数统计:")
    print(f"  平均: {diagnosis['runoff_coefficient']['mean']:.3f}")
    print(f"  范围: [{diagnosis['runoff_coefficient']['min']:.3f}, "
          f"{diagnosis['runoff_coefficient']['max']:.3f}]")

    print("\n数据质量:")
    print(f"  总样本数: {diagnosis['data_quality']['n_total']}")
    print(f"  缺失数: {diagnosis['data_quality']['n_missing']} "
          f"({diagnosis['data_quality']['pct_missing']:.1f}%)")
    print(f"  异常值: {diagnosis['data_quality']['n_extreme']} "
          f"({diagnosis['data_quality']['pct_extreme']:.1f}%)")

    if diagnosis['warnings']:
        print("\n警告:")
        for warning in diagnosis['warnings']:
            print(f"  - {warning}")
    else:
        print("\n数据质量良好，无警告。")
