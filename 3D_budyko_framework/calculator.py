"""
Potential Evapotranspiration (PET) Calculation Module
实现多种PET计算方法
"""

import numpy as np
from typing import Optional, Dict
import warnings

class PETCalculator:
    """
    潜在蒸散发计算器
    
    实现方法:
    1. Penman-Monteith (标准和改进版)
    2. Priestley-Taylor
    3. Hargreaves
    4. Thornthwaite
    5. 考虑LAI和CO2的改进PM方法 (研究创新点)
    """
    
    def __init__(self, elevation: float = 0):
        """
        初始化
        
        Parameters:
        -----------
        elevation : float
            海拔高度 [m]，用于气压计算
        """
        self.elevation = elevation
        self.atm_pressure = self._calculate_atmospheric_pressure()
        
    def _calculate_atmospheric_pressure(self) -> float:
        """
        根据海拔计算大气压
        
        P = 101.3 * [(293 - 0.0065 * z) / 293]^5.26
        
        Returns:
        --------
        pressure : float
            大气压 [kPa]
        """
        return 101.3 * np.power((293 - 0.0065 * self.elevation) / 293, 5.26)
    
    @staticmethod
    def saturation_vapor_pressure(T: np.ndarray) -> np.ndarray:
        """
        饱和水汽压 (Tetens公式)
        
        es = 0.6108 * exp(17.27 * T / (T + 237.3))
        
        Parameters:
        -----------
        T : array
            气温 [°C]
            
        Returns:
        --------
        es : array
            饱和水汽压 [kPa]
        """
        return 0.6108 * np.exp(17.27 * T / (T + 237.3))
    
    @staticmethod
    def slope_vapor_pressure_curve(T: np.ndarray) -> np.ndarray:
        """
        饱和水汽压曲线斜率
        
        Δ = 4098 * es / (T + 237.3)^2
        
        Parameters:
        -----------
        T : array
            气温 [°C]
            
        Returns:
        --------
        delta : array
            斜率 [kPa/°C]
        """
        es = PETCalculator.saturation_vapor_pressure(T)
        return 4098 * es / np.power(T + 237.3, 2)
    
    @staticmethod
    def psychrometric_constant(P: Optional[float] = None) -> float:
        """
        干湿表常数
        
        γ = 0.665 * 10^-3 * P
        
        Parameters:
        -----------
        P : float, optional
            大气压 [kPa]，默认101.3
            
        Returns:
        --------
        gamma : float
            干湿表常数 [kPa/°C]
        """
        if P is None:
            P = 101.3
        return 0.665 * 1e-3 * P
    
    def penman_monteith_fao56(self, 
                              T: np.ndarray,
                              RH: np.ndarray,
                              u2: np.ndarray,
                              Rn: np.ndarray,
                              G: Optional[np.ndarray] = None,
                              crop_type: str = 'reference') -> np.ndarray:
        """
        FAO-56 Penman-Monteith方法
        
        PET = (0.408 * Δ * (Rn - G) + γ * (900/(T+273)) * u2 * (es - ea)) / 
              (Δ + γ * (1 + 0.34 * u2))
        
        Parameters:
        -----------
        T : array
            平均气温 [°C]
        RH : array
            相对湿度 [%]
        u2 : array
            2m高度风速 [m/s]
        Rn : array
            净辐射 [MJ/m²/day]
        G : array, optional
            土壤热通量 [MJ/m²/day]，默认0
        crop_type : str
            'reference' 或 'grass'
            
        Returns:
        --------
        PET : array
            潜在蒸散发 [mm/day]
        """
        if G is None:
            G = np.zeros_like(T)
        
        # 饱和水汽压
        es = self.saturation_vapor_pressure(T)
        
        # 实际水汽压
        ea = es * RH / 100.0
        
        # 斜率
        delta = self.slope_vapor_pressure_curve(T)
        
        # 干湿表常数
        gamma = self.psychrometric_constant(self.atm_pressure)
        
        # PM方程
        numerator = 0.408 * delta * (Rn - G) + gamma * (900 / (T + 273)) * u2 * (es - ea)
        denominator = delta + gamma * (1 + 0.34 * u2)
        
        PET = numerator / denominator
        
        return np.maximum(PET, 0)  # 确保非负
    
    def penman_monteith_with_lai_co2(self,
                                     T: np.ndarray,
                                     RH: np.ndarray,
                                     u2: np.ndarray,
                                     Rn: np.ndarray,
                                     LAI: np.ndarray,
                                     CO2: Optional[np.ndarray] = None,
                                     G: Optional[np.ndarray] = None) -> np.ndarray:
        """
        考虑LAI和CO2影响的改进Penman-Monteith方法
        
        研究创新点：整合植被叶面积指数和大气CO2浓度
        
        改进点:
        1. 气孔阻抗受LAI调控: rs = rs_min / LAI
        2. CO2影响气孔导度: rs = rs * (1 + β * ln(CO2/CO2_ref))
        3. 冠层阻抗: rc = rs / (LAI * 0.5)
        
        Parameters:
        -----------
        T, RH, u2, Rn : array
            气象变量 (同标准PM)
        LAI : array
            叶面积指数 [m²/m²]
        CO2 : array, optional
            大气CO2浓度 [ppm]，默认400 ppm
        G : array, optional
            土壤热通量
            
        Returns:
        --------
        PET : array
            潜在蒸散发 [mm/day]
        """
        if G is None:
            G = np.zeros_like(T)
        if CO2 is None:
            CO2 = 400 * np.ones_like(T)
        
        # 基本参数
        es = self.saturation_vapor_pressure(T)
        ea = es * RH / 100.0
        delta = self.slope_vapor_pressure_curve(T)
        gamma = self.psychrometric_constant(self.atm_pressure)
        
        # LAI影响：最小气孔阻抗
        rs_min = 70  # s/m，参考值
        LAI_safe = np.maximum(LAI, 0.1)  # 避免除零
        rs = rs_min / LAI_safe
        
        # CO2影响：气孔导度修正
        CO2_ref = 400  # ppm，参考浓度
        beta = 0.15    # CO2敏感性参数（经验值）
        co2_factor = 1 + beta * np.log(CO2 / CO2_ref)
        rs = rs * co2_factor
        
        # 冠层阻抗
        rc = rs / (LAI_safe * 0.5)  # 有效LAI = LAI * 0.5
        
        # 空气动力学阻抗
        ra = 208 / u2  # s/m
        
        # 修改后的PM方程
        numerator = (delta * (Rn - G) + 
                    (1.2 * 1.01 * (es - ea) / ra))  # 1.2 kg/m³为空气密度
        denominator = delta + gamma * (1 + rc / ra)
        
        PET = 0.408 * numerator / denominator  # 转换为mm/day
        
        return np.maximum(PET, 0)
    
    @staticmethod
    def priestley_taylor(T: np.ndarray,
                        Rn: np.ndarray,
                        G: Optional[np.ndarray] = None,
                        alpha: float = 1.26) -> np.ndarray:
        """
        Priestley-Taylor方法
        
        PET = α * Δ/(Δ+γ) * (Rn - G) / λ
        
        Parameters:
        -----------
        T : array
            气温 [°C]
        Rn : array
            净辐射 [MJ/m²/day]
        G : array, optional
            土壤热通量
        alpha : float
            Priestley-Taylor系数，默认1.26
            
        Returns:
        --------
        PET : array
            潜在蒸散发 [mm/day]
        """
        if G is None:
            G = np.zeros_like(T)
        
        delta = PETCalculator.slope_vapor_pressure_curve(T)
        gamma = PETCalculator.psychrometric_constant()
        
        lambda_val = 2.45  # 汽化潜热 [MJ/kg]
        
        PET = alpha * (delta / (delta + gamma)) * (Rn - G) / lambda_val
        
        return np.maximum(PET, 0)
    
    @staticmethod
    def hargreaves(T_mean: np.ndarray,
                   T_max: np.ndarray,
                   T_min: np.ndarray,
                   Ra: np.ndarray) -> np.ndarray:
        """
        Hargreaves方法（简化，仅需温度）
        
        PET = 0.0023 * (T_mean + 17.8) * (T_max - T_min)^0.5 * Ra
        
        Parameters:
        -----------
        T_mean : array
            平均气温 [°C]
        T_max, T_min : array
            最高和最低气温 [°C]
        Ra : array
            大气顶层辐射 [MJ/m²/day]
            
        Returns:
        --------
        PET : array
            潜在蒸散发 [mm/day]
        """
        TD = np.maximum(T_max - T_min, 0)
        
        PET = 0.0023 * (T_mean + 17.8) * np.sqrt(TD) * Ra * 0.408
        
        return np.maximum(PET, 0)
    
    @staticmethod
    def thornthwaite(T: np.ndarray,
                     latitude: float,
                     day_of_year: np.ndarray) -> np.ndarray:
        """
        Thornthwaite方法（仅需温度，适用于数据稀缺地区）
        
        PET = 16 * (L/12) * (N/30) * (10*T/I)^a
        
        Parameters:
        -----------
        T : array
            月平均气温 [°C]
        latitude : float
            纬度 [度]
        day_of_year : array
            日序
            
        Returns:
        --------
        PET : array
            潜在蒸散发 [mm/month]
        """
        # 简化实现
        T_positive = np.maximum(T, 0)
        
        # 热量指数
        I = np.sum(np.power(T_positive / 5, 1.514))
        
        # 指数a
        a = 0.49239 + 0.01792 * I - 7.71e-5 * I**2 + 6.75e-7 * I**3
        
        # 日照时数修正
        L = self._calculate_daylight_hours(latitude, day_of_year)
        
        # PET计算
        PET = 16 * (L / 12) * np.power(10 * T_positive / I, a)
        
        return np.maximum(PET, 0)
    
    @staticmethod
    def _calculate_daylight_hours(latitude: float, day_of_year: np.ndarray) -> np.ndarray:
        """
        计算日照时数
        
        Parameters:
        -----------
        latitude : float
            纬度 [度]
        day_of_year : array
            日序 (1-365)
            
        Returns:
        --------
        N : array
            日照时数 [小时]
        """
        # 太阳赤纬
        delta = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
        
        # 纬度转弧度
        phi = np.deg2rad(latitude)
        
        # 日照时角
        omega_s = np.arccos(-np.tan(phi) * np.tan(delta))
        
        # 日照时数
        N = 24 / np.pi * omega_s
        
        return N
    
    def compare_methods(self, 
                       T: np.ndarray,
                       **kwargs) -> Dict[str, np.ndarray]:
        """
        比较不同PET方法
        
        Parameters:
        -----------
        T : array
            气温数据
        **kwargs : 
            其他方法所需参数
            
        Returns:
        --------
        results : dict
            不同方法的PET结果
        """
        results = {}
        
        # 尝试计算各种方法
        if all(k in kwargs for k in ['RH', 'u2', 'Rn']):
            results['PM_FAO56'] = self.penman_monteith_fao56(
                T, kwargs['RH'], kwargs['u2'], kwargs['Rn']
            )
            
            if 'LAI' in kwargs:
                results['PM_LAI_CO2'] = self.penman_monteith_with_lai_co2(
                    T, kwargs['RH'], kwargs['u2'], kwargs['Rn'], 
                    kwargs['LAI'], kwargs.get('CO2', None)
                )
        
        if 'Rn' in kwargs:
            results['Priestley_Taylor'] = self.priestley_taylor(
                T, kwargs['Rn']
            )
        
        if all(k in kwargs for k in ['T_max', 'T_min', 'Ra']):
            results['Hargreaves'] = self.hargreaves(
                T, kwargs['T_max'], kwargs['T_min'], kwargs['Ra']
            )
        
        return results


class PETUncertaintyAnalyzer:
    """
    PET方法不确定性分析
    
    评估不同PET公式对Budyko关系的影响
    """
    
    def __init__(self):
        self.methods = []
        self.pet_results = {}
        
    def add_pet_method(self, name: str, pet_values: np.ndarray):
        """
        添加PET计算结果
        
        Parameters:
        -----------
        name : str
            方法名称
        pet_values : array
            PET值
        """
        self.methods.append(name)
        self.pet_results[name] = pet_values
    
    def calculate_spread(self) -> Dict[str, np.ndarray]:
        """
        计算PET方法间的差异
        
        Returns:
        --------
        spread : dict
            包含mean, std, cv, range等
        """
        if len(self.methods) < 2:
            raise ValueError("需要至少2种PET方法")
        
        # 转换为数组
        pet_array = np.array([self.pet_results[m] for m in self.methods])
        
        spread = {
            'mean': np.mean(pet_array, axis=0),
            'std': np.std(pet_array, axis=0),
            'min': np.min(pet_array, axis=0),
            'max': np.max(pet_array, axis=0),
            'range': np.ptp(pet_array, axis=0),  # peak-to-peak
        }
        
        # 变异系数
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spread['cv'] = spread['std'] / spread['mean'] * 100
        
        return spread
    
    def rank_methods(self, reference_pet: np.ndarray) -> pd.DataFrame:
        """
        相对于参考PET排序各方法
        
        Parameters:
        -----------
        reference_pet : array
            参考PET（如实测或最佳估计）
            
        Returns:
        --------
        ranking : DataFrame
            方法排序和误差统计
        """
        import pandas as pd
        
        results = []
        
        for method in self.methods:
            pet = self.pet_results[method]
            
            # 计算误差指标
            bias = np.mean(pet - reference_pet)
            rmse = np.sqrt(np.mean((pet - reference_pet)**2))
            mae = np.mean(np.abs(pet - reference_pet))
            
            # 相关系数
            if len(pet) > 1:
                r = np.corrcoef(pet, reference_pet)[0, 1]
            else:
                r = np.nan
            
            results.append({
                'method': method,
                'bias': bias,
                'rmse': rmse,
                'mae': mae,
                'correlation': r
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('rmse')
        
        return df


if __name__ == "__main__":
    print("PET计算模块加载成功")
    print("可用类: PETCalculator, PETUncertaintyAnalyzer")
    print("\n支持的PET方法:")
    print("1. Penman-Monteith (FAO-56)")
    print("2. Penman-Monteith with LAI & CO2 (创新)")
    print("3. Priestley-Taylor")
    print("4. Hargreaves")
    print("5. Thornthwaite")