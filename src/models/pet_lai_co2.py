# src/models/pet_lai_co2.py
"""
创新PET方法：考虑LAI和CO2的Penman-Monteith模型

核心创新点：
1. LAI动态调整表面阻抗和辐射吸收
2. CO2浓度影响气孔导度（CO2施肥效应）
3. 更准确反映变化环境下的蒸发需求

References
----------
- Ball, J. T., Woodrow, I. E., & Berry, J. A. (1987). A model predicting
  stomatal conductance and its contribution to the control of photosynthesis
  under different environmental conditions.
- Medlyn, B. E., et al. (2001). Stomatal conductance of forest species
  after long-term exposure to elevated CO2 concentration: a synthesis.
- Yang, Y., et al. (2019). Evapotranspiration on a greening Earth.
  Nature Reviews Earth & Environment.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class PETComponents:
    """PET计算组分数据类"""
    pet_total: np.ndarray
    radiation_term: np.ndarray
    aerodynamic_term: np.ndarray
    surface_resistance: np.ndarray
    aerodynamic_resistance: np.ndarray


class PETWithLAICO2:
    """
    考虑LAI和CO2的Penman-Monteith PET模型

    物理机制：
    -----------
    1. LAI影响：
       - 增加蒸腾面积 → 表面阻抗降低
       - 增加辐射截取 → 可用能量增加

    2. CO2影响：
       - CO2浓度升高 → 气孔部分关闭
       - 气孔导度降低 → 表面阻抗增加 → PET降低

    参数：
    ------
    elevation : float
        海拔高度 (m)
    latitude : float
        纬度 (度)
    lai_method : str
        LAI调整方法 ('linear', 'exponential', 'logistic')
    co2_method : str
        CO2调整方法 ('linear_log', 'ball_berry', 'medlyn')
    """

    def __init__(self,
                 elevation: float = 0,
                 latitude: float = 30,
                 lai_method: str = 'exponential',
                 co2_method: str = 'linear_log'):
        """
        初始化PET计算器

        Parameters
        ----------
        elevation : float
            海拔高度 (m), 默认0（海平面）
        latitude : float
            纬度 (度), 默认30°N
        lai_method : str
            LAI调整方法:
            - 'linear': 线性关系
            - 'exponential': 指数衰减（推荐）
            - 'logistic': 逻辑斯谛函数
        co2_method : str
            CO2调整方法:
            - 'linear_log': 对数线性（推荐）
            - 'ball_berry': Ball-Berry模型
            - 'medlyn': Medlyn模型
        """
        self.elevation = elevation
        self.latitude = latitude
        self.lai_method = lai_method
        self.co2_method = co2_method

        # 参考值
        self.rs_min_ref = 70    # 参考最小表面阻抗 (s/m)
        self.lai_ref = 3.0      # 参考LAI
        self.co2_ref = 380      # 参考CO2浓度 (ppm)

        # CO2敏感性参数（文献范围：0.15-0.25）
        self.k_co2 = 0.20

    def calculate(self,
                  temperature: np.ndarray,
                  humidity: np.ndarray,
                  wind_speed: np.ndarray,
                  radiation: np.ndarray,
                  lai: np.ndarray,
                  co2: np.ndarray,
                  pressure: Optional[np.ndarray] = None,
                  return_components: bool = False) -> np.ndarray:
        """
        计算考虑LAI和CO2的PET

        Parameters
        ----------
        temperature : np.ndarray
            空气温度 (°C)
        humidity : np.ndarray
            相对湿度 (%)
        wind_speed : np.ndarray
            风速 (m/s), 2m高度
        radiation : np.ndarray
            净辐射 (W/m²)
        lai : np.ndarray
            叶面积指数 (无量纲)
        co2 : np.ndarray
            大气CO2浓度 (ppm)
        pressure : np.ndarray, optional
            大气压 (kPa), 如不提供则根据海拔计算
        return_components : bool
            是否返回各组分

        Returns
        -------
        np.ndarray or PETComponents
            PET (mm/day) 或包含各组分的对象
        """
        # 确保输入为numpy数组
        T = np.asarray(temperature)
        RH = np.asarray(humidity)
        u2 = np.asarray(wind_speed)
        Rn = np.asarray(radiation)
        LAI = np.asarray(lai)
        CO2 = np.asarray(co2)

        # 气象变量计算
        if pressure is None:
            P = self._calculate_pressure(self.elevation)
        else:
            P = np.asarray(pressure)

        # 饱和水汽压及斜率
        es = self._saturation_vapor_pressure(T)
        delta = self._vapor_pressure_slope(T)

        # 实际水汽压
        ea = es * RH / 100.0

        # 干湿表常数
        gamma = 0.000665 * P

        # ============ 创新1: LAI调整表面阻抗 ============
        rs_lai = self._adjust_surface_resistance_lai(LAI)

        # ============ 创新2: CO2调整气孔导度 ============
        rs_co2_factor = self._adjust_for_co2(CO2)

        # 综合表面阻抗
        rs = rs_lai * rs_co2_factor

        # 空气动力学阻抗
        ra = self._aerodynamic_resistance(u2)

        # LAI对辐射的调整（可选）
        Rn_eff = self._adjust_radiation_lai(Rn, LAI)

        # ============ Penman-Monteith公式 ============
        # 辐射项
        radiation_term = (delta * Rn_eff) / (delta + gamma * (1 + rs/ra))

        # 空气动力项
        rho_cp = 1.013  # 空气密度*比热 (kJ m-3 K-1)
        aerodynamic_term = (rho_cp * (es - ea) / ra) / (delta + gamma * (1 + rs/ra))

        # 总PET (W/m² → mm/day)
        lambda_heat = 2.45  # MJ/kg
        pet_mj = (radiation_term + aerodynamic_term) / 1e6  # W to MJ
        pet_mm = pet_mj / lambda_heat  # MJ/m²/day to mm/day

        # 确保非负
        pet_mm = np.maximum(pet_mm, 0)

        if return_components:
            return PETComponents(
                pet_total=pet_mm,
                radiation_term=radiation_term / 1e6 / lambda_heat,
                aerodynamic_term=aerodynamic_term / 1e6 / lambda_heat,
                surface_resistance=rs,
                aerodynamic_resistance=ra
            )
        else:
            return pet_mm

    def _adjust_surface_resistance_lai(self, lai: np.ndarray) -> np.ndarray:
        """
        根据LAI调整表面阻抗

        物理机制：
        - LAI增加 → 蒸腾叶面积增加 → 总阻抗降低
        - 使用指数衰减关系模拟叶片重叠效应

        Parameters
        ----------
        lai : np.ndarray
            叶面积指数

        Returns
        -------
        np.ndarray
            表面阻抗 (s/m)
        """
        lai_safe = np.maximum(lai, 0.1)  # 避免除零

        if self.lai_method == 'linear':
            # 线性反比关系
            rs = self.rs_min_ref * (self.lai_ref / lai_safe)

        elif self.lai_method == 'exponential':
            # 指数关系（推荐）
            # rs = rs_min * exp(-k * (LAI - LAI_ref))
            k_lai = 0.5
            rs = self.rs_min_ref * np.exp(-k_lai * (lai_safe - self.lai_ref))

        elif self.lai_method == 'logistic':
            # 逻辑斯谛函数
            lai_max = 8.0
            rs = self.rs_min_ref * (1 + (lai_max - lai_safe) / lai_safe)

        else:
            raise ValueError(f"Unknown LAI method: {self.lai_method}")

        # 限制在合理范围
        rs = np.clip(rs, 30, 500)

        return rs

    def _adjust_for_co2(self, co2: np.ndarray) -> np.ndarray:
        """
        根据CO2浓度调整气孔导度（表面阻抗因子）

        物理机制：
        - CO2浓度升高 → 气孔部分关闭（减少水分损失）
        - 气孔导度降低 → 表面阻抗增加 → PET降低

        Parameters
        ----------
        co2 : np.ndarray
            大气CO2浓度 (ppm)

        Returns
        -------
        np.ndarray
            阻抗调整因子 (>1 表示阻抗增加)
        """
        if self.co2_method == 'linear_log':
            # 对数线性关系（推荐，基于多项研究）
            # rs_factor = 1 + k * ln(CO2/CO2_ref)
            # 文献范围：k = 0.15-0.25
            factor = 1 + self.k_co2 * np.log(co2 / self.co2_ref)

        elif self.co2_method == 'ball_berry':
            # Ball-Berry模型简化版
            # g_s ∝ 1 / (1 + k * CO2)
            factor = (1 + self.k_co2 * self.co2_ref) / (1 + self.k_co2 * co2)
            factor = 1 / factor  # 转换为阻抗因子

        elif self.co2_method == 'medlyn':
            # Medlyn模型
            # g_s ∝ 1 / sqrt(CO2)
            factor = np.sqrt(co2 / self.co2_ref)

        else:
            raise ValueError(f"Unknown CO2 method: {self.co2_method}")

        # 限制在合理范围（0.8-1.5）
        factor = np.clip(factor, 0.8, 1.5)

        return factor

    def _adjust_radiation_lai(self,
                             radiation: np.ndarray,
                             lai: np.ndarray) -> np.ndarray:
        """
        根据LAI调整有效辐射

        LAI越大，截取的辐射越多（但有饱和效应）

        Parameters
        ----------
        radiation : np.ndarray
            净辐射 (W/m²)
        lai : np.ndarray
            叶面积指数

        Returns
        -------
        np.ndarray
            有效辐射 (W/m²)
        """
        # Beer-Lambert定律
        k_ext = 0.5  # 消光系数
        fraction_intercepted = 1 - np.exp(-k_ext * lai)

        # 有效辐射
        rn_eff = radiation * (0.5 + 0.5 * fraction_intercepted)

        return rn_eff

    def _saturation_vapor_pressure(self, temperature: np.ndarray) -> np.ndarray:
        """
        计算饱和水汽压（Tetens公式）

        Parameters
        ----------
        temperature : np.ndarray
            温度 (°C)

        Returns
        -------
        np.ndarray
            饱和水汽压 (kPa)
        """
        es = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))
        return es

    def _vapor_pressure_slope(self, temperature: np.ndarray) -> np.ndarray:
        """
        计算饱和水汽压曲线斜率

        Parameters
        ----------
        temperature : np.ndarray
            温度 (°C)

        Returns
        -------
        np.ndarray
            斜率 (kPa/°C)
        """
        delta = 4098 * self._saturation_vapor_pressure(temperature) / \
                (temperature + 237.3)**2
        return delta

    def _calculate_pressure(self, elevation: float) -> float:
        """
        根据海拔计算大气压

        Parameters
        ----------
        elevation : float
            海拔高度 (m)

        Returns
        -------
        float
            大气压 (kPa)
        """
        P = 101.3 * ((293 - 0.0065 * elevation) / 293)**5.26
        return P

    def _aerodynamic_resistance(self, wind_speed: np.ndarray) -> np.ndarray:
        """
        计算空气动力学阻抗

        Parameters
        ----------
        wind_speed : np.ndarray
            风速 (m/s), 2m高度

        Returns
        -------
        np.ndarray
            空气动力学阻抗 (s/m)
        """
        # 简化公式（假设草地参考表面）
        z = 2.0      # 测量高度 (m)
        zh = 0.12    # 作物高度 (m)
        d = 2/3 * zh # 零平面位移 (m)
        zom = 0.123 * zh  # 动量粗糙度 (m)
        zoh = 0.1 * zom   # 热量粗糙度 (m)
        k = 0.41     # von Karman常数

        # 确保风速不为零
        u2 = np.maximum(wind_speed, 0.1)

        ra = (np.log((z - d) / zom) * np.log((z - d) / zoh)) / (k**2 * u2)

        return ra


class PETComparator:
    """
    PET方法对比分析器

    用于对比传统PET方法与LAI+CO2改进方法的差异
    """

    def __init__(self):
        self.results = {}

    def compare_methods(self,
                       data: Dict[str, np.ndarray],
                       methods: list = None) -> Dict:
        """
        对比多种PET方法

        Parameters
        ----------
        data : Dict[str, np.ndarray]
            气象数据字典，包含：
            - temperature, humidity, wind_speed, radiation
            - lai (可选), co2 (可选)
        methods : list, optional
            要对比的方法列表

        Returns
        -------
        Dict
            对比结果
        """
        from .pet_models import PETModelFactory

        if methods is None:
            methods = ['standard_pm', 'lai_co2']

        results = {}

        # 标准Penman-Monteith
        if 'standard_pm' in methods:
            from .pet_models import PenmanMonteith
            pm_standard = PenmanMonteith()
            results['standard_pm'] = pm_standard.calculate(
                temp_avg=data['temperature'],
                temp_max=data.get('temperature_max', data['temperature'] + 5),
                temp_min=data.get('temperature_min', data['temperature'] - 5),
                rh_mean=data['humidity'],
                wind_speed=data['wind_speed'],
                solar_radiation=data['radiation'] * 0.0864,  # W/m² to MJ/m²/day
                latitude=data.get('latitude', 30),
                elevation=data.get('elevation', 0),
                day_of_year=data.get('day_of_year', np.arange(1, len(data['temperature'])+1))
            )

        # LAI+CO2方法
        if 'lai_co2' in methods and 'lai' in data and 'co2' in data:
            pet_advanced = PETWithLAICO2(
                elevation=data.get('elevation', 0),
                latitude=data.get('latitude', 30)
            )
            results['lai_co2'] = pet_advanced.calculate(
                temperature=data['temperature'],
                humidity=data['humidity'],
                wind_speed=data['wind_speed'],
                radiation=data['radiation'],
                lai=data['lai'],
                co2=data['co2']
            )

        # 计算统计量
        comparison = self._calculate_comparison_stats(results)

        return {
            'pet_values': results,
            'statistics': comparison
        }

    def _calculate_comparison_stats(self, results: Dict) -> Dict:
        """计算对比统计量"""
        stats = {}

        method_names = list(results.keys())

        for method in method_names:
            pet = results[method]
            stats[method] = {
                'mean': np.nanmean(pet),
                'std': np.nanstd(pet),
                'min': np.nanmin(pet),
                'max': np.nanmax(pet),
                'median': np.nanmedian(pet)
            }

        # 方法间差异
        if len(method_names) >= 2:
            method1 = method_names[0]
            method2 = method_names[1]

            diff = results[method2] - results[method1]
            stats['difference'] = {
                'mean_diff': np.nanmean(diff),
                'mean_pct_diff': 100 * np.nanmean(diff / results[method1]),
                'rmse': np.sqrt(np.nanmean(diff**2)),
                'correlation': np.corrcoef(
                    results[method1][~np.isnan(results[method1])],
                    results[method2][~np.isnan(results[method2])]
                )[0, 1]
            }

        return stats


class CO2SensitivityAnalyzer:
    """
    CO2敏感性分析器

    用于分析PET对CO2浓度变化的响应
    """

    def __init__(self):
        self.pet_calculator = PETWithLAICO2()

    def analyze_co2_response(self,
                            baseline_data: Dict,
                            co2_scenarios: np.ndarray) -> Dict:
        """
        分析不同CO2情景下的PET响应

        Parameters
        ----------
        baseline_data : Dict
            基准气象数据
        co2_scenarios : np.ndarray
            CO2情景数组 (ppm)，如 [350, 400, 450, 500, 550, 600]

        Returns
        -------
        Dict
            包含各情景PET及敏感性指标
        """
        results = {}

        for co2 in co2_scenarios:
            # 创建CO2数组（与其他变量长度一致）
            co2_array = np.full_like(baseline_data['temperature'], co2)

            # 计算PET
            pet = self.pet_calculator.calculate(
                temperature=baseline_data['temperature'],
                humidity=baseline_data['humidity'],
                wind_speed=baseline_data['wind_speed'],
                radiation=baseline_data['radiation'],
                lai=baseline_data['lai'],
                co2=co2_array
            )

            results[f'co2_{int(co2)}'] = {
                'co2': co2,
                'pet': pet,
                'mean_pet': np.nanmean(pet)
            }

        # 计算敏感性
        co2_values = list(co2_scenarios)
        pet_means = [results[f'co2_{int(co2)}']['mean_pet'] for co2 in co2_values]

        # 线性回归
        z = np.polyfit(co2_values, pet_means, 1)
        sensitivity = z[0]  # PET变化率 (mm/day per ppm)

        results['sensitivity_analysis'] = {
            'co2_range': [min(co2_values), max(co2_values)],
            'pet_range': [min(pet_means), max(pet_means)],
            'absolute_sensitivity': sensitivity,
            'relative_sensitivity': 100 * sensitivity / pet_means[0],  # %/(100 ppm)
            'co2_values': co2_values,
            'pet_means': pet_means
        }

        return results


# 使用示例
if __name__ == '__main__':
    # 创建示例数据
    n_days = 365

    data = {
        'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 365),
        'humidity': 60 + 20 * np.cos(2 * np.pi * np.arange(n_days) / 365),
        'wind_speed': 2 + 1 * np.random.rand(n_days),
        'radiation': 200 + 100 * np.sin(2 * np.pi * np.arange(n_days) / 365),
        'lai': 3 + 2 * np.sin(2 * np.pi * np.arange(n_days) / 365),
        'co2': 400 * np.ones(n_days)
    }

    # 计算PET
    pet_calculator = PETWithLAICO2(elevation=500, latitude=35)

    pet = pet_calculator.calculate(
        temperature=data['temperature'],
        humidity=data['humidity'],
        wind_speed=data['wind_speed'],
        radiation=data['radiation'],
        lai=data['lai'],
        co2=data['co2']
    )

    print("PET统计量:")
    print(f"  均值: {np.mean(pet):.2f} mm/day")
    print(f"  标准差: {np.std(pet):.2f} mm/day")
    print(f"  年总量: {np.sum(pet):.2f} mm/year")

    # CO2敏感性分析
    analyzer = CO2SensitivityAnalyzer()
    co2_response = analyzer.analyze_co2_response(
        baseline_data=data,
        co2_scenarios=np.array([350, 400, 450, 500, 550, 600])
    )

    print("\nCO2敏感性分析:")
    sens = co2_response['sensitivity_analysis']
    print(f"  CO2范围: {sens['co2_range'][0]}-{sens['co2_range'][1]} ppm")
    print(f"  PET范围: {sens['pet_range'][0]:.2f}-{sens['pet_range'][1]:.2f} mm/day")
    print(f"  绝对敏感性: {sens['absolute_sensitivity']:.4f} mm/day/ppm")
    print(f"  相对敏感性: {sens['relative_sensitivity']:.2f} %/(100ppm)")
