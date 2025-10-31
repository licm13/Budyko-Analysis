# src/models/pet/penman_monteith_lai_co2.py
"""
创新PET方法：考虑LAI和CO2的Penman-Monteith模型

核心创新点 (来自您的研究思路):
1. LAI动态调整表面阻抗
2. CO2浓度影响气孔导度（CO2施肥效应）

物理机制：
1. LAI ↑ → 蒸腾面积 ↑ → 表面阻抗 rs ↓ → PET ↑
2. CO2 ↑ → 气孔部分关闭 → 气孔导度 gs ↓ → 表面阻抗 rs ↑ → PET ↓
"""

import numpy as np
from typing import Dict, Optional, Tuple, Literal
from dataclasses import dataclass
import warnings

# 物理常数
RHO_AIR = 1.225  # 空气密度 (kg/m³)
CP_AIR = 1.013e-3 # 空气定压比热 (MJ/kg/K)
LAMBDA_HEAT = 2.45  # 汽化潜热 (MJ/kg)
VON_KARMAN = 0.41 # von Karman 常数

@dataclass
class PETComponents:
    """PET计算组分数据类"""
    pet_total: np.ndarray
    radiation_term: np.ndarray
    aerodynamic_term: np.ndarray
    surface_resistance: np.ndarray
    aerodynamic_resistance: np.ndarray

class PenmanMonteithLAICO2:
    """
    考虑LAI和CO2的Penman-Monteith PET模型
    
    基于FAO-56 PM公式，但用动态的表面阻抗(rs)替换了固定值(70 s/m)
    """

    def __init__(self,
                 elevation: float = 100.0,
                 rs_min_ref: float = 70.0,
                 co2_ref: float = 380.0,
                 k_co2_sensitivity: float = 0.20):
        """
        初始化PET计算器

        Parameters
        ----------
        elevation : float
            海拔高度 (m)
        rs_min_ref : float
            参考最小气孔阻抗 (s/m)
        co2_ref : float
            参考CO2浓度 (ppm)
        k_co2_sensitivity : float
            CO2敏感性参数 (k_co2)
        """
        self.elevation = elevation
        self.rs_min_ref = rs_min_ref
        self.co2_ref = co2_ref
        self.k_co2 = k_co2_sensitivity
        
        # 计算一次性常数
        self.pressure = 101.3 * ((293 - 0.0065 * self.elevation) / 293)**5.26
        self.gamma = (CP_AIR * self.pressure) / (0.622 * LAMBDA_HEAT) # 干湿表常数 (kPa/°C)

    def calculate(self,
                  T_avg: np.ndarray,
                  Rn: np.ndarray,
                  u2: np.ndarray,
                  RH: np.ndarray,
                  LAI: np.ndarray,
                  CO2: np.ndarray,
                  G: Optional[np.ndarray] = None,
                  return_components: bool = False) -> np.ndarray | PETComponents:
        """
        计算PET

        Parameters
        ----------
        T_avg : np.ndarray
            日均气温 (°C)
        Rn : np.ndarray
            净辐射 (MJ/m²/day)
        u2 : np.ndarray
            2m风速 (m/s)
        RH : np.ndarray
            相对湿度 (%)
        LAI : np.ndarray
            叶面积指数 (m²/m²)
        CO2 : np.ndarray
            大气CO2浓度 (ppm)
        G : np.ndarray, optional
            土壤热通量 (MJ/m²/day)，默认为0
        return_components : bool
            是否返回各组分

        Returns
        -------
        np.ndarray or PETComponents
            PET (mm/day)
        """
        if G is None:
            G = np.zeros_like(T_avg)
            
        # 1. 气象变量
        delta = self._vapor_pressure_slope(T_avg)  # (kPa/°C)
        es = self._saturation_vapor_pressure(T_avg) # (kPa)
        ea = es * RH / 100.0                       # (kPa)
        vpd = es - ea                              # (kPa)

        # 2. 核心创新：计算动态表面阻抗 (rs)
        rs = self._calculate_surface_resistance(LAI, CO2)
        
        # 3. 计算空气动力学阻抗 (ra)
        ra = self._calculate_aerodynamic_resistance(u2)

        # 4. Penman-Monteith (FAO-56, Allen et al. 1998)
        # 辐射项 (mm/day)
        rad_num = delta * (Rn - G)
        rad_den = delta + self.gamma * (1 + rs / ra)
        radiation_term = rad_num / rad_den / LAMBDA_HEAT
        
        # 空气动力项 (mm/day)
        aero_num = (RHO_AIR * CP_AIR * vpd / ra)
        aero_den = delta + self.gamma * (1 + rs / ra)
        aerodynamic_term = aero_num / aero_den / LAMBDA_HEAT

        # 总PET (mm/day)
        pet_total = radiation_term + aerodynamic_term
        pet_total = np.maximum(pet_total, 0) # 确保非负
        
        if return_components:
            return PETComponents(
                pet_total=pet_total,
                radiation_term=radiation_term,
                aerodynamic_term=aerodynamic_term,
                surface_resistance=rs,
                aerodynamic_resistance=ra
            )
        else:
            return pet_total

    def _calculate_surface_resistance(self, lai: np.ndarray, co2: np.ndarray) -> np.ndarray:
        """
        根据LAI和CO2计算冠层表面阻抗 (rs)
        """
        # 1. LAI对最小阻抗的影响
        # rs_min 随 LAI 增加而降低 (更多叶片)
        # 使用指数衰减来模拟叶片重叠效应
        lai_safe = np.maximum(lai, 0.1) # 避免除零
        k_lai = 0.5 # 消光系数
        # 假设参考冠层 (LAI=3) 具有 rs_min_ref (70 s/m)
        # rs_min = rs_min_ref * (LAI_ref / LAI_eff)
        # LAI_eff = (1 - exp(-k_lai * LAI)) / k_lai (Goudriaan & van Laar, 1994)
        # 简化：rs_min_canopy = rs_min_leaf / LAI
        # 假设 rs_min_ref 是参考冠层的 rs_min_canopy
        
        # 方法：rs_canopy = rs_leaf / LAI_effective
        # 假设 rs_min_ref(70) 是一个标准值
        # rs 随 LAI 增加而降低，随 CO2 增加而增加
        
        # CO2 效应：气孔导度 gs = 1/rs
        # gs_factor = 1 - k * ln(CO2/CO2_ref) (Medlyn et al. 2001)
        # rs_factor = 1 / gs_factor 
        # 简化：rs_factor ≈ 1 + k * ln(CO2/CO2_ref) (线性近似)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            co2_factor = 1 + self.k_co2 * np.log(co2 / self.co2_ref)
            co2_factor = np.clip(co2_factor, 0.8, 1.5) # 限制范围
            
            # 基础阻抗（仅CO2）
            rs_base = self.rs_min_ref * co2_factor
            
            # LAI 效应：LAI增加，总冠层阻抗降低
            # rs = rs_base / LAI (简化)
            rs = rs_base / np.maximum(lai, 0.1)
        
        # 限制在合理范围
        rs = np.clip(rs, 20, 500)
        return rs

    def _calculate_aerodynamic_resistance(self, u2: np.ndarray, zh: float = 0.12) -> np.ndarray:
        """
        计算空气动力学阻抗 (ra) (s/m)
        (FAO-56 Eq. 4)
        
        Parameters
        ----------
        u2 : np.ndarray
            2m风速 (m/s)
        zh : float
            参考作物高度 (m), 默认0.12m (草地)
        """
        # 零平面位移 (FAO-56 Eq. 3)
        d = (2/3) * zh
        # 动量粗糙度 (FAO-56 Eq. 2)
        zom = 0.123 * zh
        # 热量粗糙度 (FAO-56 Eq. 2)
        zoh = 0.1 * zom
        
        # (FAO-56 Eq. 4)
        num = np.log((2 - d) / zom) * np.log((2 - d) / zoh)
        den = VON_KARMAN**2 * np.maximum(u2, 0.1) # 避免除零
        
        ra = num / den
        return ra

    def _saturation_vapor_pressure(self, T: np.ndarray) -> np.ndarray:
        """饱和水汽压 (kPa)"""
        return 0.6108 * np.exp((17.27 * T) / (T + 237.3))

    def _vapor_pressure_slope(self, T: np.ndarray) -> np.ndarray:
        """饱和水汽压曲线斜率 (kPa/°C)"""
        es = self._saturation_vapor_pressure(T)
        return (4098 * es) / (T + 237.3)**2