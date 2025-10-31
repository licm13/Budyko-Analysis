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
from typing import Dict, Tuple

def _to_array(x, n: int):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    return arr

class PETWithLAICO2:
    """
    简化的 FAO-56 Penman-Monteith PET, 并加入 LAI 与 CO2 调制因子
    输出单位：mm/day（与测试约束一致：< 20 mm/day）
    """
    def __init__(self, elevation: float = 0.0, latitude: float = 0.0):
        self.elevation = float(elevation)
        self.latitude = float(latitude)

    def calculate(self,
                  temperature,
                  humidity,
                  wind_speed,
                  radiation,
                  lai,
                  co2):
        """
        参数（标量或等长数组）：
        - temperature: °C
        - humidity: 相对湿度 % (0-100)
        - wind_speed: m/s
        - radiation: Rn 净辐射，W/m^2
        - lai: 叶面积指数
        - co2: 大气CO2浓度 ppm
        返回：np.ndarray，mm/day
        """
        # 统一为数组
        n = np.size(temperature)
        T = _to_array(temperature, n)
        RH = np.clip(_to_array(humidity, n), 1.0, 100.0)
        u2 = np.maximum(_to_array(wind_speed, n), 0.0)
        Rn_w = np.maximum(_to_array(radiation, n), 0.0)  # W/m^2
        LAI = np.maximum(_to_array(lai, n), 0.0)
        CO2 = np.maximum(_to_array(co2, n), 0.0)

        # 单位换算：W/m^2 → MJ/m^2/day
        Rn_MJ = Rn_w * 0.0864

        # 饱和水汽压 (kPa)
        es = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
        ea = es * (RH / 100.0)

        # 斜率 Δ (kPa/°C)
        delta = 4098.0 * es / np.power(T + 237.3, 2)

        # 气压与心理常数 γ (kPa/°C)
        P_atm = 101.3 * np.power((293.0 - 0.0065 * self.elevation) / 293.0, 5.26)
        gamma = 0.000665 * P_atm

        # 简化 FAO-56 PM (日尺度，地表通量，G≈0)
        # ET0 = [0.408*Δ*(Rn) + γ*(900/(T+273))*u2*(es-ea)] / [Δ + γ*(1+0.34*u2)]
        numerator = 0.408 * delta * Rn_MJ + gamma * (900.0 / (T + 273.0)) * u2 * (es - ea)
        denominator = delta + gamma * (1.0 + 0.34 * u2)
        et0 = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

        # LAI 与 CO2 调制（经验型，限制在合理范围）
        # LAI 增大 → 蒸腾增强；CO2 增大 → 气孔关闭→蒸腾减弱
        lai_factor = np.clip(1.0 + 0.10 * (LAI - 3.0) / 3.0, 0.7, 1.3)
        co2_factor = np.clip(1.0 - 0.05 * ((CO2 - 400.0) / 100.0), 0.7, 1.1)

        pet = et0 * lai_factor * co2_factor
        pet = np.nan_to_num(pet, nan=0.0, posinf=0.0, neginf=0.0)
        pet = np.maximum(pet, 0.0)

        return pet


class PETComparator:
    """PET结果对比工具"""
    @staticmethod
    def compare(a, b) -> Dict[str, float]:
        A = np.asarray(a, dtype=float)
        B = np.asarray(b, dtype=float)
        diff = A - B
        rmse = float(np.sqrt(np.nanmean(diff ** 2)))
        corr = float(np.corrcoef(A, B)[0, 1]) if A.size > 1 else float("nan")
        return {
            "mean_A": float(np.nanmean(A)),
            "mean_B": float(np.nanmean(B)),
            "mean_diff": float(np.nanmean(diff)),
            "rmse": rmse,
            "corr": corr
        }