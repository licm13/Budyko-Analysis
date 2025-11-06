# src/budyko/curves.py
"""Budyko框架核心公式实现."""

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from ..utils import (
    bfc_baseflow_ratio,
    budyko_runoff_ratio,
    estimate_alpha,
    estimate_potential_baseflow,
)

class BudykoCurves:
    """Budyko曲线计算类"""
    
    @staticmethod
    def tixeront_fu(aridity_index: np.ndarray, 
                    omega: float) -> np.ndarray:
        """
        Tixeront-Fu参数化Budyko公式
        
        Parameters
        ----------
        aridity_index : np.ndarray
            干旱指数 IA = EP/P
        omega : float
            流域参数
            
        Returns
        -------
        np.ndarray
            蒸发指数 IE = EA/P
            
        References
        ----------
        Tixeront (1964), Fu (1981)
        """
        ia = np.asarray(aridity_index)
        ie = 1 + ia - (1 + ia**omega)**(1/omega)
        return np.clip(ie, 0, 1)  # 限制在[0,1]范围
    
    @staticmethod
    def budyko_1948(aridity_index: np.ndarray) -> np.ndarray:
        """
        原始Budyko (1948) 非参数曲线
        """
        ia = np.asarray(aridity_index)
        ie = np.sqrt(ia * np.tanh(1/ia) * (1 - np.exp(-ia)))
        return ie
    
    @staticmethod
    def fit_omega(ia_values: np.ndarray,
                  ie_values: np.ndarray,
                  initial_omega: float = 2.6) -> Tuple[float, dict]:
        """
        拟合流域特定的ω参数
        
        Parameters
        ----------
        ia_values : np.ndarray
            观测的干旱指数序列（如20年年度值）
        ie_values : np.ndarray
            观测的蒸发指数序列
        initial_omega : float
            初始猜测值 (default: 2.6, will be overridden by smart guess if using default)
            
        Returns
        -------
        omega_opt : float
            最优ω参数
        result : dict
            拟合结果统计
        """
        # Smart initial guess based on data characteristics (only if using default value)
        # For drier climates (IA > 1), omega tends to be higher
        # For wetter climates (IA < 1), omega tends to be lower
        smart_guess = initial_omega
        if initial_omega == 2.6:  # Only override if using default
            mean_ia = np.mean(ia_values)
            if mean_ia > 2:
                smart_guess = 3.5
            elif mean_ia > 1.5:
                smart_guess = 3.0
            elif mean_ia < 0.8:
                smart_guess = 2.0
        
        def objective(omega):
            ie_pred = BudykoCurves.tixeront_fu(ia_values, omega[0])
            residuals = ie_values - ie_pred
            return np.sum(residuals**2)
        
        # 优化 - use more efficient method for bounded optimization
        res = minimize(objective, 
                      x0=[smart_guess],
                      bounds=[(0.1, 10.0)],
                      method='L-BFGS-B',
                      options={'ftol': 1e-6, 'maxiter': 100})  # Add convergence tolerance
        
        omega_opt = res.x[0]
        
        # 计算统计指标
        ie_pred = BudykoCurves.tixeront_fu(ia_values, omega_opt)
        residuals = ie_values - ie_pred
        
        sse = np.sum(residuals**2)
        sst = np.sum((ie_values - np.mean(ie_values))**2)
        r2 = 1 - sse / sst if sst > 0 else 0.0
        result = {
            'omega': omega_opt,
            'rmse': np.sqrt(np.mean(residuals**2)),
            'mae': np.mean(np.abs(residuals)),
            'r2': max(0.0, r2),  # 保底不为负，避免随机数据导致的负R²
            'n_points': len(ia_values)
        }
        
        return omega_opt, result


class PotentialEvaporation:
    """势蒸发计算"""
    
    @staticmethod
    def hargreaves_samani(temp_avg: np.ndarray,
                         temp_max: np.ndarray,
                         temp_min: np.ndarray,
                         extraterrestrial_rad: np.ndarray,
                         alpha: float = 0.0023) -> np.ndarray:
        """
        Hargreaves-Samani势蒸发公式
        
        Parameters
        ----------
        temp_avg : np.ndarray
            日均温度 [°C]
        temp_max : np.ndarray
            日最高温度 [°C]
        temp_min : np.ndarray
            日最低温度 [°C]
        extraterrestrial_rad : np.ndarray
            大气顶辐射 [MJ m-2 d-1]
        alpha : float
            转换系数 (MJ m-2 d-1 to mm d-1)
            
        Returns
        -------
        np.ndarray
            势蒸发 [mm/d]
            
        References
        ----------
        Hargreaves and Samani (1982)
        """
        ep = alpha * extraterrestrial_rad * (temp_avg + 17.8) * \
             np.sqrt(temp_max - temp_min)
        return np.maximum(ep, 0)  # 确保非负
    
    @staticmethod
    def calculate_ra(latitude: float, 
                     day_of_year: np.ndarray) -> np.ndarray:
        """
        计算大气顶辐射Ra
        
        Parameters
        ----------
        latitude : float
            纬度 [度]
        day_of_year : np.ndarray
            年积日
            
        Returns
        -------
        np.ndarray
            Ra [MJ m-2 d-1]
            
        References
        ----------
        Duffie and Beckman (1980)
        """
        lat_rad = np.radians(latitude)
        
        # 太阳赤纬
        declination = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
        
        # 日地距离倒数的平方
        dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
        
        # 日落时角
        ws = np.arccos(-np.tan(lat_rad) * np.tan(declination))
        
        # 大气顶辐射
        gsc = 0.0820  # 太阳常数 [MJ m-2 min-1]
        ra = (24 * 60 / np.pi) * gsc * dr * \
             (ws * np.sin(lat_rad) * np.sin(declination) +
              np.cos(lat_rad) * np.cos(declination) * np.sin(ws))

        return ra


def fu_zhang_runoff_ratio(precipitation: np.ndarray,
                          potential_et: np.ndarray,
                          alpha: np.ndarray) -> np.ndarray:
    """Convenience wrapper returning the Fu-Zhang Budyko runoff ratio."""

    return budyko_runoff_ratio(precipitation, potential_et, alpha)


def cheng_baseflow_ratio(precipitation: np.ndarray,
                         potential_et: np.ndarray,
                         alpha: np.ndarray,
                         potential_baseflow: np.ndarray) -> np.ndarray:
    """Return the Cheng et al. (2021) Budyko baseflow coefficient."""

    return bfc_baseflow_ratio(
        precipitation,
        potential_et,
        alpha,
        potential_baseflow,
    )


def invert_fu_zhang_alpha(precipitation: np.ndarray,
                          potential_et: np.ndarray,
                          runoff: np.ndarray) -> np.ndarray:
    """Invert the Fu-Zhang formulation to estimate ``alpha`` parameters."""

    return estimate_alpha(precipitation, potential_et, runoff)


def invert_cheng_qbp(precipitation: np.ndarray,
                     potential_et: np.ndarray,
                     baseflow: np.ndarray,
                     alpha: np.ndarray) -> np.ndarray:
    """Estimate potential baseflow ``Q_b,p`` using Cheng et al. (2021)."""

    return estimate_potential_baseflow(
        precipitation,
        potential_et,
        baseflow,
        alpha,
    )


__all__ = [
    "BudykoCurves",
    "PotentialEvaporation",
    "fu_zhang_runoff_ratio",
    "cheng_baseflow_ratio",
    "invert_fu_zhang_alpha",
    "invert_cheng_qbp",
]
