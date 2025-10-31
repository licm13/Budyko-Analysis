# src/analysis/snow_analyzer.py
"""
积雪数据处理器

研究积雪对Budyko关系的影响
(基于 3D_budyko_framework/processor.py 中的 SnowDataProcessor)
"""

import numpy as np
import pandas as pd
from typing import List, Dict

class SnowImpactAnalyzer:
    """
    积雪影响分析器
    """
    
    def __init__(self, T_threshold: float = 0.0, melt_factor: float = 2.5):
        """
        初始化
        
        Parameters:
        -----------
        T_threshold : float
            融雪阈值温度 [°C]
        melt_factor : float
            融雪系数 [mm/°C/day]
        """
        self.T_threshold = T_threshold
        self.melt_factor = melt_factor

    def calculate_snowfall_and_snowmelt(self,
                                        precipitation: np.ndarray,
                                        temperature: np.ndarray,
                                        snow_depth_initial: float = 0.0) -> Dict[str, np.ndarray]:
        """
        计算降雪、融雪和雪水当量(SWE)变化
        
        使用简化的度日模型
        
        Parameters:
        -----------
        precipitation : array
            日降水 [mm]
        temperature : array
            日均温 [°C]
        snow_depth_initial : float
            初始雪深(SWE) [mm]
            
        Returns:
        --------
        dict
            包含 'rainfall', 'snowfall', 'potential_melt', 'actual_melt', 'swe'
        """
        n_days = len(precipitation)
        
        # 输出数组
        rainfall = np.zeros(n_days)
        snowfall = np.zeros(n_days)
        potential_melt = np.zeros(n_days)
        actual_melt = np.zeros(n_days)
        swe = np.zeros(n_days)
        
        # 初始积雪
        current_swe = snow_depth_initial
        
        for i in range(n_days):
            # 1. 区分雨雪
            if temperature[i] <= self.T_threshold:
                snowfall[i] = precipitation[i]
                rainfall[i] = 0.0
            else:
                snowfall[i] = 0.0
                rainfall[i] = precipitation[i]
                
            # 2. 计算潜在融雪
            if temperature[i] > self.T_threshold:
                potential_melt[i] = self.melt_factor * (temperature[i] - self.T_threshold)
            else:
                potential_melt[i] = 0.0
                
            # 3. 更新雪水当量 (SWE)
            # 增加量 = 降雪, 减少量 = 融雪
            current_swe += snowfall[i]
            
            # 4. 计算实际融雪
            # 融雪不能超过当前积雪量
            melt = np.minimum(current_swe, potential_melt[i])
            actual_melt[i] = melt
            
            # 5. 更新最终SWE
            current_swe -= melt
            swe[i] = current_swe
            
        return {
            'rainfall': rainfall,
            'snowfall': snowfall,
            'potential_melt': potential_melt,
            'actual_melt': actual_melt,
            'swe': swe  # 雪水当量 (mm)
        }

    def identify_snow_dominated_basins(self,
                                      data: pd.DataFrame,
                                      snow_ratio_threshold: float = 0.2) -> List:
        """
        识别积雪主导的流域
        
        Parameters:
        -----------
        data : DataFrame
            包含 'basin_id' 和 'snow_ratio' (降雪/总降水) 列
        snow_ratio_threshold : float
            积雪比例阈值
            
        Returns:
        --------
        snow_basins : list
            积雪主导的流域ID列表
        """
        if 'snow_ratio' not in data.columns:
            if 'snowfall' in data.columns and 'P' in data.columns:
                annual_data = data.groupby('basin_id')[['snowfall', 'P']].sum()
                annual_data['snow_ratio'] = annual_data['snowfall'] / annual_data['P']
            else:
                raise ValueError("数据需要包含'snow_ratio'列, 或 'snowfall' 和 'P' 列")
        else:
            annual_data = data.groupby('basin_id')['snow_ratio'].mean()
            
        snow_basins = annual_data[annual_data > snow_ratio_threshold].index.unique().tolist()
        
        print(f"识别出 {len(snow_basins)} 个积雪主导流域 (阈值 > {snow_ratio_threshold*100}%)")
        
        return snow_basins

    def analyze_budyko_with_snow(self,
                                 annual_data: pd.DataFrame,
                                 snow_results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        在Budyko框架中分离融雪和降雨的影响
        
        P_eff = Rainfall + Actual_Melt
        Q = Q_rain + Q_melt
        
        Parameters:
        -----------
        annual_data : pd.DataFrame
            包含 P, Q, PET 的年度数据
        snow_results : dict
            来自 calculate_snowfall_and_snowmelt 的结果
            
        Returns:
        --------
        pd.DataFrame
            包含修正后Budyko指数的DataFrame
        """
        # 聚合日度融雪/降雨到年度
        snow_df = pd.DataFrame(snow_results)
        snow_df['year'] = annual_data['year'] # 假设已对齐
        
        annual_snow = snow_df.groupby('year').sum()
        
        analysis = annual_data.merge(annual_snow, on='year')
        
        # 修正的水量平衡
        analysis['P_liquid'] = analysis['rainfall'] + analysis['actual_melt']
        
        # 修正的干旱指数 (使用液态水供应)
        analysis['IA_liquid'] = analysis['PET'] / analysis['P_liquid']
        
        # 修正的蒸发 (假设Q不变)
        analysis['EA_liquid'] = analysis['P_liquid'] - analysis['Q']
        analysis['IE_liquid'] = analysis['EA_liquid'] / analysis['P_liquid']
        
        return analysis