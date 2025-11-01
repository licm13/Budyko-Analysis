# src/models/pet_models.py
"""
多种势蒸发(PET)估算方法

Jaramillo et al. (2022)强调PET方法选择对结果的重大影响
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from abc import ABC, abstractmethod

class PETModel(ABC):
    """PET模型基类"""
    
    @abstractmethod
    def calculate(self, **kwargs) -> np.ndarray:
        """计算PET"""
        pass
    
    @property
    @abstractmethod
    def required_inputs(self) -> List[str]:
        """所需输入变量"""
        pass


class HargreavesSamani(PETModel):
    """
    Hargreaves-Samani方法（已在前面实现）
    
    优点：仅需温度数据
    缺点：可能高估PET
    """
    
    @property
    def required_inputs(self):
        return ['temp_avg', 'temp_max', 'temp_min', 'latitude', 'day_of_year']
    
    def calculate(self,
                 temp_avg: np.ndarray,
                 temp_max: np.ndarray,
                 temp_min: np.ndarray,
                 latitude: float,
                 day_of_year: np.ndarray,
                 alpha: float = 0.0023) -> np.ndarray:
        """
        EP = α * Ra * (Ta + 17.8) * sqrt(Tmax - Tmin)
        """
        # 计算大气顶辐射
        ra = self._calculate_ra(latitude, day_of_year)
        
        # PET
        pet = alpha * ra * (temp_avg + 17.8) * np.sqrt(temp_max - temp_min)
        return np.maximum(pet, 0)
    
    @staticmethod
    def _calculate_ra(latitude: float, day_of_year: np.ndarray) -> np.ndarray:
        """同前面实现"""
        lat_rad = np.radians(latitude)
        declination = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
        dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
        ws = np.arccos(-np.tan(lat_rad) * np.tan(declination))
        gsc = 0.0820
        ra = (24 * 60 / np.pi) * gsc * dr * \
             (ws * np.sin(lat_rad) * np.sin(declination) + 
              np.cos(lat_rad) * np.cos(declination) * np.sin(ws))
        return ra


class PenmanMonteith(PETModel):
    """
    Penman-Monteith方法（FAO-56标准）
    
    优点：物理基础强，精度高
    缺点：需要更多气象数据
    """
    
    @property
    def required_inputs(self):
        return ['temp_avg', 'temp_max', 'temp_min', 'rh_mean', 
                'wind_speed', 'solar_radiation', 'latitude', 'elevation']
    
    def calculate(self,
                 temp_avg: np.ndarray,
                 temp_max: np.ndarray,
                 temp_min: np.ndarray,
                 rh_mean: np.ndarray,
                 wind_speed: np.ndarray,
                 solar_radiation: np.ndarray,
                 latitude: float,
                 elevation: float,
                 day_of_year: np.ndarray) -> np.ndarray:
        """
        FAO-56 Penman-Monteith公式
        
        ET0 = (0.408Δ(Rn-G) + γ(900/(T+273))u2(es-ea)) / (Δ + γ(1+0.34u2))
        
        References
        ----------
        Allen et al. (1998) FAO Irrigation and Drainage Paper 56
        """
        # 1. 饱和水汽压
        es_tmax = 0.6108 * np.exp(17.27 * temp_max / (temp_max + 237.3))
        es_tmin = 0.6108 * np.exp(17.27 * temp_min / (temp_min + 237.3))
        es = (es_tmax + es_tmin) / 2  # [kPa]
        
        # 2. 实际水汽压
        ea = es * rh_mean / 100  # [kPa]
        
        # 3. 饱和水汽压曲线斜率
        delta = 4098 * (0.6108 * np.exp(17.27 * temp_avg / (temp_avg + 237.3))) / \
                (temp_avg + 237.3)**2  # [kPa/°C]
        
        # 4. 大气压（根据海拔）
        P = 101.3 * ((293 - 0.0065 * elevation) / 293)**5.26  # [kPa]
        
        # 5. 干湿表常数
        gamma = 0.000665 * P  # [kPa/°C]
        
        # 6. 净辐射
        # 简化：假设土壤热通量G≈0（日尺度）
        Rn = solar_radiation * 0.408  # 转换MJ m-2 d-1到mm d-1的等效蒸发
        G = 0
        
        # 7. 风速调整（2m高度）
        u2 = wind_speed  # 假设已经是2m高度
        
        # 8. ET0计算
        numerator = 0.408 * delta * (Rn - G) + \
                   gamma * (900 / (temp_avg + 273)) * u2 * (es - ea)
        denominator = delta + gamma * (1 + 0.34 * u2)
        
        et0 = numerator / denominator
        return np.maximum(et0, 0)


class PriestleyTaylor(PETModel):
    """
    Priestley-Taylor方法
    
    优点：中等数据需求，基于能量平衡
    缺点：需要辐射数据
    """
    
    @property
    def required_inputs(self):
        return ['temp_avg', 'solar_radiation', 'latitude', 'elevation']
    
    def calculate(self,
                 temp_avg: np.ndarray,
                 solar_radiation: np.ndarray,
                 latitude: float,
                 elevation: float,
                 alpha: float = 1.26) -> np.ndarray:
        """
        ET = α * (Δ/(Δ+γ)) * (Rn - G) / λ
        
        Parameters
        ----------
        alpha : float
            Priestley-Taylor系数，通常1.26
        """
        # 饱和水汽压曲线斜率
        delta = 4098 * (0.6108 * np.exp(17.27 * temp_avg / (temp_avg + 237.3))) / \
                (temp_avg + 237.3)**2
        
        # 大气压和干湿表常数
        P = 101.3 * ((293 - 0.0065 * elevation) / 293)**5.26
        gamma = 0.000665 * P
        
        # 净辐射（简化）
        Rn = solar_radiation * 0.75  # 假设净辐射≈0.75*总辐射
        G = 0
        
        # 潜热汽化热
        lambda_heat = 2.45  # MJ kg-1
        
        # PET
        pet = alpha * (delta / (delta + gamma)) * (Rn - G) / lambda_heat
        return np.maximum(pet, 0)


# FAO-56 Penman–Monteith helpers
def _svp(T):
    # saturation vapor pressure (kPa)
    return 0.6108 * np.exp((17.27 * T) / (T + 237.3))

def _slope_vp_curve(T):
    es = _svp(T)
    return 4098.0 * es / np.power(T + 237.3, 2)  # kPa/°C

def _psychrometric_const(elevation_m):
    # atmospheric pressure (kPa)
    P = 101.3 * np.power((293.0 - 0.0065 * elevation_m) / 293.0, 5.26)
    return 0.000665 * P  # kPa/°C

def _extraterrestrial_radiation(day_of_year, latitude_deg):
    # Ra (MJ m-2 day-1)
    phi = np.deg2rad(latitude_deg)
    J = np.asarray(day_of_year, dtype=float)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * J / 365.0)
    delta = 0.409 * np.sin(2.0 * np.pi * J / 365.0 - 1.39)
    omega_s = np.arccos(-np.tan(phi) * np.tan(delta))
    Gsc = 0.0820  # MJ m-2 min-1
    Ra = (24.0 * 60.0 / np.pi) * Gsc * dr * (
        omega_s * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(omega_s)
    )
    return Ra

class PenmanMonteithPET:
    """
    FAO-56 Penman–Monteith reference ET0 (mm/day).
    Expects daily-scale inputs; for annual series, pass arrays of equal length.
    """
    def __init__(self, albedo: float = 0.23):
        self.albedo = float(albedo)

    def calculate(self,
                  temp_avg,
                  temp_max,
                  temp_min,
                  rh_mean,
                  wind_speed,
                  solar_radiation,  # Rs (MJ m-2 day-1)
                  latitude,
                  elevation,
                  day_of_year):
        # to arrays (robust to scalar inputs)
        T = np.atleast_1d(np.asarray(temp_avg, dtype=float))
        Tmax = np.atleast_1d(np.asarray(temp_max, dtype=float))
        Tmin = np.atleast_1d(np.asarray(temp_min, dtype=float))
        RH = np.clip(np.atleast_1d(np.asarray(rh_mean, dtype=float)), 1.0, 100.0)
        u2 = np.maximum(np.atleast_1d(np.asarray(wind_speed, dtype=float)), 0.0)
        Rs = np.maximum(np.atleast_1d(np.asarray(solar_radiation, dtype=float)), 0.0)
        lat = float(latitude)
        elev = float(elevation)
        J = np.atleast_1d(np.asarray(day_of_year, dtype=float))
        n = T.size
        if J.size == 1:
            J = np.full(n, J.item(), dtype=float)

        # radiation terms
        Ra = _extraterrestrial_radiation(J, lat)  # MJ m-2 day-1
        Rso = (0.75 + 2e-5 * elev) * Ra  # clear-sky radiation
        Rns = (1.0 - self.albedo) * Rs
        sigma = 4.903e-9  # MJ K-4 m-2 day-1
        Tk4_mean = (np.power(Tmax + 273.16, 4) + np.power(Tmin + 273.16, 4)) / 2.0
        # cloudiness factor
        f_cloud = np.clip(np.divide(Rs, Rso, out=np.ones_like(Rs), where=Rso > 0), 0.0, 1.0)
        ea = _svp(T) * (RH / 100.0)  # kPa
        Rnl = sigma * Tk4_mean * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0))) * (1.35 * f_cloud - 0.35)
        Rnl = np.maximum(Rnl, 0.0)
        Rn = np.maximum(Rns - Rnl, 0.0)  # MJ m-2 day-1

        # PM equation
        delta = _slope_vp_curve(T)
        gamma = _psychrometric_const(elev)
        es = _svp(T)
        G = 0.0  # daily-scale soil heat flux ~ 0
        num = 0.408 * delta * (Rn - G) + gamma * (900.0 / (T + 273.0)) * u2 * (es - ea)
        den = delta + gamma * (1.0 + 0.34 * u2)
        et0 = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        et0 = np.nan_to_num(np.maximum(et0, 0.0), nan=0.0, posinf=0.0, neginf=0.0)
        # return scalar if scalar inputs were provided
        return float(et0[0]) if n == 1 else et0

class PETModelFactory:
    """PET模型工厂"""
    
    MODELS = {
        'hargreaves': HargreavesSamani,
        'penman_monteith': PenmanMonteith,
        'priestley_taylor': PriestleyTaylor
    }
    
    @classmethod
    def create(cls, model_name: str) -> PETModel:
        """
        创建PET模型实例
        
        Parameters
        ----------
        model_name : str
            模型名称: 'hargreaves', 'penman_monteith', 'priestley_taylor'
            
        Returns
        -------
        PETModel
            模型实例
        """
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown PET model: {model_name}. "
                           f"Available: {list(cls.MODELS.keys())}")
        
        return cls.MODELS[model_name]()
    
    @classmethod
    def compare_methods(cls,
                       data: Dict,
                       models: List[str] = None) -> pd.DataFrame:
        """
        比较不同PET方法的结果
        
        Parameters
        ----------
        data : Dict
            包含所有可能需要的气象数据
        models : List[str], optional
            要比较的模型列表
            
        Returns
        -------
        pd.DataFrame
            PET比较结果
        """
        if models is None:
            models = list(cls.MODELS.keys())
        
        results = {}
        
        for model_name in models:
            try:
                model = cls.create(model_name)
                
                # 检查所需输入
                required = set(model.required_inputs)
                available = set(data.keys())
                
                if not required.issubset(available):
                    missing = required - available
                    print(f"Skipping {model_name}: missing {missing}")
                    continue
                
                # 提取所需数据
                inputs = {key: data[key] for key in required}
                
                # 计算PET
                pet = model.calculate(**inputs)
                results[model_name] = pet
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                continue
        
        # 转换为DataFrame
        if results:
            df = pd.DataFrame(results)
            df['date'] = data.get('date', range(len(next(iter(results.values())))))
            return df
        else:
            return pd.DataFrame()