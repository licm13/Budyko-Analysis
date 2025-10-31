# src/budyko/water_balance.py
"""
水量平衡计算模块 (Budyko分析的基石)

核心原理 (来自您的研究思路):
-------------------------------
径流数据(Q)是Budyko分析的基石！
水量平衡方程：P - Q = EA + ΔS

长时间尺度假设：
ΔS ≈ 0，因此 EA ≈ P - Q

Budyko指数：
- 干旱指数 IA = PET/P （X轴）
- 蒸发指数 IE = EA/P = (P-Q)/P （Y轴）

**没有径流Q，我们无法计算IE，Budyko分析无法进行！**
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Literal
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

    # 核心计算结果
    actual_evaporation: np.ndarray
    aridity_index: np.ndarray
    evaporation_index: np.ndarray
    
    # 3D框架指数 (He et al. 2023)
    storage_change_index: Optional[np.ndarray] = None
    actual_evaporation_extended: Optional[np.ndarray] = None
    evaporation_index_extended: Optional[np.ndarray] = None

    # 质量控制
    data_quality_flags: np.ndarray # type: ignore
    closure_error: Optional[np.ndarray] = None

class WaterBalanceCalculator:
    """
    水量平衡计算器

    核心功能：
    1. 基于径流Q计算实际蒸发EA
    2. 计算2D Budyko指数（IA, IE）
    3. (可选) 计算3D Budyko指数 (整合ΔS)
    4. 数据质量控制
    """

    def __init__(self,
                 allow_negative_ea: bool = False,
                 min_precipitation: float = 100,
                 max_runoff_ratio: float = 0.95,
                 water_year_start_month: int = 9):
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
        water_year_start_month : int
            水文年起始月份 (1-12)，用于聚合
        """
        self.allow_negative_ea = allow_negative_ea
        self.min_precipitation = min_precipitation
        self.max_runoff_ratio = max_runoff_ratio
        self.water_year_start_month = water_year_start_month

    def calculate_budyko_indices(self,
                                 P: np.ndarray,
                                 Q: np.ndarray,
                                 PET: np.ndarray,
                                 delta_S: Optional[np.ndarray] = None,
                                 TWS: Optional[np.ndarray] = None) -> WaterBalanceResults:
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
            储量变化 (mm)，如 P - Q - EA。
            如果提供，则 EA = P - Q - delta_S
            如果不提供，则 EA = P - Q （假设ΔS≈0）
        TWS : np.ndarray, optional
            (He et al. 2023) 陆地水储量 (mm)，用于计算SCI

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
            delta_S = np.zeros_like(P) # 假设为0
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

        # ============ 计算2D Budyko指数 ============
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # 干旱指数 IA = PET/P （X轴）
            IA = PET / P
            # 蒸发指数 IE = EA/P （Y轴）- 由径流Q决定！
            IE = EA / P

        # ============ (可选) 计算3D Budyko指数 (He et al. 2023) ============
        SCI = None
        EA_ext = None
        IE_ext = None

        if TWS is not None:
            if delta_S is None and not include_storage:
                # 如果用户没提供delta_S，但提供了TWS，我们计算它
                delta_S_from_TWS = np.diff(TWS, prepend=TWS[0])
                delta_S = delta_S_from_TWS
            
            # 储量变化指数
            SCI = delta_S / P
            
            # 扩展的实际蒸发 (EA_ext)
            EA_ext = P - Q - delta_S
            EA_ext = np.maximum(EA_ext, 0)
            
            # 扩展的蒸发指数 (IE_ext)
            IE_ext = EA_ext / P

        # ============ 数据质量控制 ============
        quality_flags = self._quality_control(P, Q, PET, EA, IA, IE)

        # ============ 水量平衡闭合误差 ============
        closure_error = P - Q - EA - (delta_S if include_storage else np.zeros_like(P))

        # 返回结果
        return WaterBalanceResults(
            precipitation=P,
            runoff=Q,
            pet=PET,
            storage_change=delta_S,
            actual_evaporation=EA,
            aridity_index=IA,
            evaporation_index=IE,
            storage_change_index=SCI,
            actual_evaporation_extended=EA_ext,
            evaporation_index_extended=IE_ext,
            data_quality_flags=quality_flags,
            closure_error=closure_error
        )

    def _quality_control(self,
                        P: np.ndarray,
                        Q: np.ndarray,
                        PET: np.ndarray,
                        EA: np.ndarray,
                        IA: np.ndarray,
                        IE: np.ndarray) -> np.ndarray:
        """数据质量控制 (返回标志)"""
        flags = np.zeros(len(P), dtype=int)
        
        # 0=良好, 1=可疑, 2=差
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            RC = Q / P
            # 检查1：降水太小
            flags[P < self.min_precipitation] = 2
            # 检查2：径流系数过大
            flags[RC > self.max_runoff_ratio] = 2
            # 检查3：蒸发指数超出物理范围
            flags[(IE < 0) | (IE > 1.2)] = 1
            # 检查4：干旱指数异常
            flags[(IA < 0) | (IA > 10)] = 1
            # 检查5：PET < EA（违反物理规律，除非有平流）
            flags[PET < EA * 0.9] = 1 # 允许10%误差
            # 检查6：NaN值
            nan_mask = (np.isnan(P) | np.isnan(Q) | np.isnan(PET) |
                       np.isnan(EA) | np.isnan(IA) | np.isnan(IE))
            flags[nan_mask] = 2
            
        return flags

    def aggregate_to_periods(self,
                            data: pd.DataFrame,
                            period_length: int = 20) -> pd.DataFrame:
        """
        将年度数据聚合为多年时段（用于Ibrahim方法）

        Parameters
        ----------
        data : pd.DataFrame
            包含 'year', 'P', 'Q', 'PET' 的年度数据
        period_length : int
            时段长度（年），默认20年

        Returns
        -------
        pd.DataFrame
            聚合后的时段数据
        """
        min_year = data['year'].min()
        max_year = data['year'].max()
        
        periods = []
        
        for start_year in range(min_year, max_year - period_length + 2):
            end_year = start_year + period_length - 1
            
            period_data = data[
                (data['year'] >= start_year) &
                (data['year'] <= end_year)
            ]
            
            # 确保至少有80%的数据
            if len(period_data) >= period_length * 0.8:
                # 水量平衡计算
                P_mean = period_data['P'].mean()
                Q_mean = period_data['Q'].mean()
                PET_mean = period_data['PET'].mean()
                
                # 核心：EA = P - Q
                EA_mean = P_mean - Q_mean
                
                # 计算时段的IA和IE
                IA_mean = PET_mean / P_mean
                IE_mean = EA_mean / P_mean
                
                periods.append({
                    'period_id': f"T_{start_year}",
                    'start_year': start_year,
                    'end_year': end_year,
                    'n_years': len(period_data),
                    'P_mean': P_mean,
                    'Q_mean': Q_mean,
                    'PET_mean': PET_mean,
                    'EA_mean': EA_mean,
                    'IA_mean': IA_mean,
                    'IE_mean': IE_mean,
                })
        
        return pd.DataFrame(periods)

    def prepare_annual_dataframe(self,
                                 dates: pd.DatetimeIndex,
                                 **kwargs: np.ndarray) -> pd.DataFrame:
        """
        将时间序列数据转换为年度DataFrame
        
        Parameters
        ----------
        dates : pd.DatetimeIndex
            日期索引
        **kwargs : np.ndarray
            变量数组 (P, Q, PET, LAI, TWS...)
            
        Returns
        -------
        pd.DataFrame
            按水文年聚合的年度数据
        """
        df = pd.DataFrame(kwargs, index=dates)
        
        # 定义水文年
        df['water_year'] = df.index.year # type: ignore
        if self.water_year_start_month != 1:
            df['water_year'] = df.index.map(
                lambda x: x.year if x.month < self.water_year_start_month else x.year + 1
            )
            
        # 聚合规则
        # 累积型变量
        sum_cols = [col for col in ['P', 'Q', 'PET', 'ET'] if col in df.columns]
        # 均值型变量
        mean_cols = [col for col in ['T', 'LAI', 'CO2', 'TWS', 'RH', 'u2'] if col in df.columns]
        
        agg_dict = {}
        for col in sum_cols:
            agg_dict[col] = 'sum'
        for col in mean_cols:
            agg_dict[col] = 'mean'
            
        annual_df = df.groupby('water_year').agg(agg_dict)
        annual_df = annual_df.rename_axis('year').reset_index()
        
        # 计算年度 TWS 变化
        if 'TWS' in annual_df.columns:
            annual_df['delta_S'] = annual_df['TWS'].diff().fillna(0)
            
        return annual_df