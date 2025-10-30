"""
Budyko Framework Analysis Module
实现传统和三维Budyko框架分析
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import warnings

class BudykoAnalyzer:
    """
    Budyko框架分析器
    
    实现功能:
    1. 传统二维Budyko关系计算
    2. 三维Budyko框架（整合TWS）
    3. 参数ω的估计
    4. 偏离分析
    """
    
    def __init__(self, time_scale='annual'):
        """
        初始化分析器
        
        Parameters:
        -----------
        time_scale : str
            时间尺度: 'annual', 'seasonal', 'monthly'
        """
        self.time_scale = time_scale
        self.data = None
        self.results = {}
        
    def load_basin_data(self, P: np.ndarray, Q: np.ndarray, 
                       PET: np.ndarray, basin_ids: Optional[np.ndarray] = None,
                       TWS: Optional[np.ndarray] = None):
        """
        加载流域数据
        
        Parameters:
        -----------
        P : array
            降水 [mm]
        Q : array
            径流 [mm]
        PET : array
            潜在蒸散发 [mm]
        basin_ids : array, optional
            流域ID
        TWS : array, optional
            陆地水储量 [mm]
        """
        n_basins = len(P)
        
        if basin_ids is None:
            basin_ids = np.arange(n_basins)
            
        self.data = pd.DataFrame({
            'basin_id': basin_ids,
            'P': P,
            'Q': Q,
            'PET': PET
        })
        
        if TWS is not None:
            self.data['TWS'] = TWS
            
        # 计算基本水文变量
        self._calculate_basic_indices()
        
    def _calculate_basic_indices(self):
        """计算基础Budyko指数"""
        # 实际蒸散发 (水量平衡)
        self.data['ET'] = self.data['P'] - self.data['Q']
        
        # 确保ET非负
        self.data['ET'] = self.data['ET'].clip(lower=0)
        
        # 蒸发指数 (Evaporative Index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.data['EI'] = self.data['ET'] / self.data['P']
            self.data['EI'] = self.data['EI'].clip(upper=1.0)  # 不应超过1
        
        # 干旱指数 (Dryness Index)
        self.data['DI'] = self.data['PET'] / self.data['P']
        
        # 径流系数
        self.data['runoff_ratio'] = self.data['Q'] / self.data['P']
        
    def calculate_3d_indices(self):
        """
        计算三维Budyko指数（整合TWS）
        
        基于论文公式:
        - SCI (Storage Change Index) = ΔS/P
        - SI (Storage Index) = TWS/P
        - Extended ET = P - Q - ΔS
        """
        if 'TWS' not in self.data.columns:
            raise ValueError("需要TWS数据来计算三维指数")
        
        # 计算储量变化（时间差分）
        self.data['dTWS'] = self.data.groupby('basin_id')['TWS'].diff()
        
        # Storage Change Index
        self.data['SCI'] = self.data['dTWS'] / self.data['P']
        
        # Storage Index
        self.data['SI'] = self.data['TWS'] / self.data['P']
        
        # 扩展的实际蒸散发
        self.data['ET_extended'] = self.data['P'] - self.data['Q'] - self.data['dTWS']
        self.data['ET_extended'] = self.data['ET_extended'].clip(lower=0)
        
        # 扩展的蒸发指数
        self.data['EI_extended'] = self.data['ET_extended'] / self.data['P']
        
        # 修正的干旱指数（有效降水）
        self.data['P_effective'] = self.data['P'] - self.data['dTWS']
        self.data['DI_extended'] = self.data['PET'] / self.data['P_effective']
        
    def budyko_curve(self, DI: np.ndarray, omega: float = 2.6) -> np.ndarray:
        """
        Fu (1981) Budyko曲线
        
        EI = 1 + DI - [1 + DI^ω]^(1/ω)
        
        Parameters:
        -----------
        DI : array
            干旱指数
        omega : float
            景观参数，默认2.6 (Creed et al., 2014)
            
        Returns:
        --------
        EI : array
            理论蒸发指数
        """
        return 1 + DI - np.power(1 + np.power(DI, omega), 1/omega)
    
    def estimate_omega(self, DI: np.ndarray, EI: np.ndarray, 
                       method='optimization') -> float:
        """
        估计Budyko参数ω
        
        Parameters:
        -----------
        DI, EI : array
            观测的干旱指数和蒸发指数
        method : str
            'optimization': 最小化RMSE
            'regression': 非线性回归
            
        Returns:
        --------
        omega : float
            最优ω参数
        """
        # 移除异常值
        valid_mask = (DI > 0) & (EI > 0) & (EI <= 1) & np.isfinite(DI) & np.isfinite(EI)
        DI_clean = DI[valid_mask]
        EI_clean = EI[valid_mask]
        
        if len(DI_clean) < 10:
            warnings.warn("有效数据点太少，返回默认值")
            return 2.6
        
        def objective(omega):
            """最小化RMSE"""
            if omega <= 0:
                return 1e10
            EI_pred = self.budyko_curve(DI_clean, omega)
            rmse = np.sqrt(np.mean((EI_pred - EI_clean)**2))
            return rmse
        
        # 优化
        result = minimize(objective, x0=2.6, bounds=[(0.5, 10.0)], 
                         method='L-BFGS-B')
        
        return result.x[0]
    
    def calculate_budyko_deviation(self, omega: Optional[float] = None) -> pd.Series:
        """
        计算Budyko曲线偏离
        
        偏离 = EI_observed - EI_theoretical
        
        Parameters:
        -----------
        omega : float, optional
            如果为None，自动估计
            
        Returns:
        --------
        deviation : Series
            偏离值
        """
        if omega is None:
            omega = self.estimate_omega(
                self.data['DI'].values, 
                self.data['EI'].values
            )
            self.results['omega'] = omega
        
        # 理论蒸发指数
        EI_theoretical = self.budyko_curve(self.data['DI'].values, omega)
        
        # 偏离
        deviation = self.data['EI'] - EI_theoretical
        
        return deviation
    
    def classify_water_energy_limit(self) -> pd.Series:
        """
        分类水分限制和能量限制区域
        
        基于DI阈值:
        - DI < 1: 能量限制 (Energy-limited)
        - DI > 1: 水分限制 (Water-limited)
        
        Returns:
        --------
        classification : Series
            'energy_limited' 或 'water_limited'
        """
        classification = pd.Series('', index=self.data.index)
        classification[self.data['DI'] < 1] = 'energy_limited'
        classification[self.data['DI'] >= 1] = 'water_limited'
        
        return classification
    
    def calculate_si_sci_relationships(self) -> Dict[str, float]:
        """
        计算DI/EI与SI/SCI的线性关系
        
        论文发现:
        - 湿季: DI/EI与SI/SCI正相关
        - 干季: DI/EI与SI/SCI负相关
        
        Returns:
        --------
        results : dict
            相关系数和斜率
        """
        if 'SI' not in self.data.columns:
            raise ValueError("需要先计算三维指数")
        
        results = {}
        
        # DI vs SI
        mask = np.isfinite(self.data['DI']) & np.isfinite(self.data['SI'])
        if mask.sum() > 0:
            corr_di_si = np.corrcoef(
                self.data.loc[mask, 'DI'], 
                self.data.loc[mask, 'SI']
            )[0, 1]
            results['corr_DI_SI'] = corr_di_si
        
        # DI vs SCI
        mask = np.isfinite(self.data['DI']) & np.isfinite(self.data['SCI'])
        if mask.sum() > 0:
            corr_di_sci = np.corrcoef(
                self.data.loc[mask, 'DI'], 
                self.data.loc[mask, 'SCI']
            )[0, 1]
            results['corr_DI_SCI'] = corr_di_sci
        
        # EI vs SI
        mask = np.isfinite(self.data['EI']) & np.isfinite(self.data['SI'])
        if mask.sum() > 0:
            corr_ei_si = np.corrcoef(
                self.data.loc[mask, 'EI'], 
                self.data.loc[mask, 'SI']
            )[0, 1]
            results['corr_EI_SI'] = corr_ei_si
        
        # EI vs SCI
        mask = np.isfinite(self.data['EI']) & np.isfinite(self.data['SCI'])
        if mask.sum() > 0:
            corr_ei_sci = np.corrcoef(
                self.data.loc[mask, 'EI'], 
                self.data.loc[mask, 'SCI']
            )[0, 1]
            results['corr_EI_SCI'] = corr_ei_sci
        
        return results
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        获取汇总统计
        
        Returns:
        --------
        summary : DataFrame
            描述性统计
        """
        cols_to_summarize = ['P', 'Q', 'PET', 'ET', 'DI', 'EI', 'runoff_ratio']
        
        if 'SI' in self.data.columns:
            cols_to_summarize.extend(['SI', 'SCI', 'EI_extended', 'DI_extended'])
        
        summary = self.data[cols_to_summarize].describe()
        
        return summary


class SeasonalBudykoAnalyzer(BudykoAnalyzer):
    """
    季节性Budyko分析
    
    分湿季和干季分别分析
    """
    
    def __init__(self):
        super().__init__(time_scale='seasonal')
        
    def split_seasons(self, dates: pd.DatetimeIndex, 
                     wet_months: list = [6, 7, 8, 9, 10],
                     dry_months: list = [11, 12, 1, 2, 3, 4, 5]):
        """
        划分湿季和干季
        
        Parameters:
        -----------
        dates : DatetimeIndex
            时间索引
        wet_months : list
            湿季月份 (默认6-10月)
        dry_months : list
            干季月份 (默认11-5月)
        """
        self.data['month'] = dates.month
        self.data['season'] = 'unknown'
        
        self.data.loc[self.data['month'].isin(wet_months), 'season'] = 'wet'
        self.data.loc[self.data['month'].isin(dry_months), 'season'] = 'dry'
        
    def analyze_by_season(self) -> Dict[str, pd.DataFrame]:
        """
        按季节分别分析
        
        Returns:
        --------
        results : dict
            {'wet': DataFrame, 'dry': DataFrame}
        """
        if 'season' not in self.data.columns:
            raise ValueError("需要先调用split_seasons划分季节")
        
        results = {}
        
        for season in ['wet', 'dry']:
            season_data = self.data[self.data['season'] == season].copy()
            
            # 估计ω参数
            omega = self.estimate_omega(
                season_data['DI'].values,
                season_data['EI'].values
            )
            
            # 计算偏离
            EI_theoretical = self.budyko_curve(season_data['DI'].values, omega)
            season_data['deviation'] = season_data['EI'] - EI_theoretical
            season_data['omega'] = omega
            
            results[season] = season_data
        
        return results


if __name__ == "__main__":
    # 测试示例
    print("Budyko分析器加载成功")
    print("可用类: BudykoAnalyzer, SeasonalBudykoAnalyzer")