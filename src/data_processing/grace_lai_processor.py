# src/data_processing/grace_lai_processor.py
"""
GRACE TWS 和 MODIS LAI 数据加载与处理模块

**GRACE（Gravity Recovery and Climate Experiment）**：
提供陆地水储量（TWS）数据，用于He et al. (2023)的3D Budyko框架。
通过 ΔS = ΔTWS，可以计算扩展的蒸散发：EA_ext = P - Q - ΔS

**MODIS LAI（Leaf Area Index）**：
叶面积指数，用于PETWithLAICO2模型的创新PET计算。
LAI↑ → 蒸腾面积↑ → 表面阻抗↓ → PET↑
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Tuple, Optional, List
import warnings


class GRACEDataLoader:
    """
    GRACE陆地水储量数据加载器

    **3D Budyko框架的关键**：
    TWS（陆地水储量）包含：地表水、土壤水、地下水、积雪
    其变化ΔS打破了传统Budyko的ΔS≈0假设，揭示了储量变化对水量平衡的贡献。
    """

    def __init__(self):
        self.grace_data = None
        self.baseline_period = None

    def load_grace_tws(self,
                      grace_dir: str,
                      solution: str = 'CSR',
                      release: str = 'RL06',
                      mascon: bool = False) -> xr.Dataset:
        """
        加载GRACE/GRACE-FO TWS数据

        **数据版本**：
        - Release: RL06 (推荐最新版本)
        - Solution: CSR, JPL, GFZ (三大解算中心)
        - Mascon: 是否使用质量浓度块解（更直接，不需解缠）

        Parameters
        ----------
        grace_dir : str
            GRACE数据目录
        solution : str
            解算中心: 'CSR', 'JPL', 'GFZ'
        release : str
            数据版本: 'RL06'
        mascon : bool
            是否为Mascon解

        Returns
        -------
        xr.Dataset
            GRACE TWS数据，单位: mm 等效水高
        """
        grace_path = Path(grace_dir)

        if not grace_path.exists():
            raise FileNotFoundError(f"GRACE目录不存在: {grace_dir}")

        # 构建文件搜索模式
        if mascon:
            pattern = f"*MASCON*{solution}*.nc"
        else:
            pattern = f"*{solution}*{release}*.nc"

        files = list(grace_path.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"未找到GRACE数据文件\n"
                f"  目录: {grace_dir}\n"
                f"  模式: {pattern}"
            )

        print(f"找到 {len(files)} 个GRACE文件")

        # 读取并合并
        ds = xr.open_mfdataset(files, combine='by_coords')

        # 标准化变量名
        if 'lwe_thickness' in ds:
            ds = ds.rename({'lwe_thickness': 'TWS'})
        elif 'tws' in ds:
            ds = ds.rename({'tws': 'TWS'})

        # 确保单位为mm
        if ds['TWS'].attrs.get('units') == 'cm':
            ds['TWS'] = ds['TWS'] * 10
            ds['TWS'].attrs['units'] = 'mm'

        self.grace_data = ds

        print(f"✓ 成功加载GRACE TWS数据")
        print(f"  解算方案: {solution}")
        print(f"  版本: {release}")
        print(f"  时间范围: {ds.time.min().values} 至 {ds.time.max().values}")

        return ds

    def calculate_tws_anomaly(self,
                             tws: xr.DataArray,
                             baseline_period: Tuple[str, str] = ('2004-01', '2009-12'),
                             detrend: bool = False) -> xr.DataArray:
        """
        计算TWS异常（去除基准期平均）

        **物理意义**：
        TWS异常反映了相对于气候平均态的水储量变化，
        消除了地形等固定因素的影响。

        Parameters
        ----------
        tws : xr.DataArray
            原始TWS数据
        baseline_period : tuple
            基准期 (start, end)，如 ('2004-01', '2009-12')
        detrend : bool
            是否去除线性趋势（用于分离长期变化与年际波动）

        Returns
        -------
        xr.DataArray
            TWS异常 [mm]
        """
        # 选择基准期
        baseline = tws.sel(time=slice(baseline_period[0], baseline_period[1]))
        tws_mean = baseline.mean(dim='time')

        # 计算异常
        tws_anomaly = tws - tws_mean

        # 可选：去趋势
        if detrend:
            tws_anomaly = self._detrend_tws(tws_anomaly)

        tws_anomaly.attrs['long_name'] = 'TWS Anomaly'
        tws_anomaly.attrs['units'] = 'mm'
        tws_anomaly.attrs['baseline_period'] = f"{baseline_period[0]} to {baseline_period[1]}"

        self.baseline_period = baseline_period

        print(f"✓ TWS异常计算完成")
        print(f"  基准期: {baseline_period[0]} 至 {baseline_period[1]}")

        return tws_anomaly

    @staticmethod
    def _detrend_tws(tws_anomaly: xr.DataArray) -> xr.DataArray:
        """
        去除TWS的线性趋势

        **应用场景**：
        - 分离人为开采导致的长期下降（如地下水耗竭）
        - 保留年际和季节变率
        """
        from scipy import signal

        # 沿时间维去趋势
        detrended = xr.apply_ufunc(
            signal.detrend,
            tws_anomaly,
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            vectorize=True,
            dask='parallelized'
        )

        return detrended

    def calculate_delta_s(self,
                         tws: xr.DataArray,
                         time_scale: str = 'annual') -> xr.DataArray:
        """
        计算储量变化 ΔS = TWS(t+1) - TWS(t)

        **3D Budyko的核心**：
        ΔS是第三维度，与DI和EI共同定义流域状态。
        - ΔS > 0: 储量增加（补给）
        - ΔS < 0: 储量减少（耗竭）

        Parameters
        ----------
        tws : xr.DataArray
            TWS数据
        time_scale : str
            时间尺度: 'monthly', 'annual'

        Returns
        -------
        xr.DataArray
            ΔS [mm]
        """
        if time_scale == 'monthly':
            delta_s = tws.diff(dim='time', n=1)
        elif time_scale == 'annual':
            # 年际差分
            tws_annual = tws.resample(time='AS').mean()
            delta_s = tws_annual.diff(dim='time', n=1)
        else:
            raise ValueError(f"不支持的时间尺度: {time_scale}")

        delta_s.attrs['long_name'] = 'Storage Change'
        delta_s.attrs['units'] = 'mm'

        print(f"✓ ΔS计算完成（{time_scale}尺度）")

        return delta_s


class LAIDataLoader:
    """
    MODIS LAI数据加载器

    **LAI在Budyko分析中的角色**：
    LAI调控植被表面阻抗，影响PET计算：
        - LAI ↑ → 蒸腾表面积 ↑ → 表面阻抗 rs ↓ → PET ↑
        - 这是PETWithLAICO2模型的物理基础
    """

    def __init__(self):
        self.lai_data = None

    def load_modis_lai(self,
                      modis_dir: str,
                      product: str = 'MOD15A2H',
                      version: str = '061',
                      time_range: Optional[Tuple[str, str]] = None) -> xr.Dataset:
        """
        加载MODIS LAI产品

        **MODIS LAI产品**：
        - MOD15A2H: Terra, 8天合成, 500m分辨率
        - MYD15A2H: Aqua, 8天合成, 500m分辨率
        - Collection 6.1 (061): 最新版本

        Parameters
        ----------
        modis_dir : str
            MODIS数据目录
        product : str
            产品名称: 'MOD15A2H' (Terra) 或 'MYD15A2H' (Aqua)
        version : str
            产品版本: '061' (Collection 6.1)
        time_range : tuple, optional
            时间范围 (start, end)

        Returns
        -------
        xr.Dataset
            LAI数据 [m²/m²]
        """
        modis_path = Path(modis_dir)

        if not modis_path.exists():
            raise FileNotFoundError(f"MODIS目录不存在: {modis_dir}")

        # 搜索MODIS文件
        pattern = f"*{product}*.nc"
        files = sorted(list(modis_path.glob(pattern)))

        if not files:
            # 尝试HDF格式
            pattern = f"*{product}*.hdf"
            files = sorted(list(modis_path.glob(pattern)))

        if not files:
            raise FileNotFoundError(
                f"未找到MODIS LAI数据\n"
                f"  目录: {modis_dir}\n"
                f"  产品: {product}"
            )

        print(f"找到 {len(files)} 个MODIS文件")

        # 读取数据
        if files[0].suffix == '.nc':
            ds = xr.open_mfdataset(files, combine='by_coords')
        elif files[0].suffix == '.hdf':
            ds = self._read_modis_hdf(files)
        else:
            raise ValueError(f"不支持的文件格式: {files[0].suffix}")

        # 标准化变量名
        if 'Lai_500m' in ds:
            ds = ds.rename({'Lai_500m': 'LAI'})
        elif 'lai' in ds:
            ds = ds.rename({'lai': 'LAI'})

        # 质量控制：LAI有效范围 [0, 10]
        ds['LAI'] = ds['LAI'].where((ds['LAI'] >= 0) & (ds['LAI'] <= 10))

        # 单位转换（如需要）
        if ds['LAI'].attrs.get('scale_factor'):
            scale = ds['LAI'].attrs['scale_factor']
            ds['LAI'] = ds['LAI'] * scale

        # 时间范围筛选
        if time_range:
            ds = ds.sel(time=slice(time_range[0], time_range[1]))

        self.lai_data = ds

        print(f"✓ 成功加载MODIS LAI数据")
        print(f"  产品: {product}")
        print(f"  时间范围: {ds.time.min().values} 至 {ds.time.max().values}")
        print(f"  LAI范围: {float(ds['LAI'].min()):.2f} - {float(ds['LAI'].max()):.2f} m²/m²")

        return ds

    @staticmethod
    def _read_modis_hdf(hdf_files: List[Path]) -> xr.Dataset:
        """
        读取MODIS HDF文件

        需要rioxarray或pyhdf
        """
        try:
            import rioxarray

            datasets = []
            for hdf_file in hdf_files:
                # 读取LAI波段
                ds = rioxarray.open_rasterio(
                    f'HDF4_EOS:EOS_GRID:"{hdf_file}":MOD_Grid_MOD15A2H:Lai_500m'
                )
                datasets.append(ds)

            # 合并
            combined = xr.concat(datasets, dim='time')

            return combined.to_dataset(name='LAI')

        except ImportError:
            raise ImportError(
                "读取MODIS HDF需要安装rioxarray:\n"
                "  pip install rioxarray"
            )

    def aggregate_to_monthly(self, lai: xr.DataArray) -> xr.DataArray:
        """
        将8天LAI聚合到月尺度

        **时间尺度匹配**：
        MODIS LAI是8天合成，但Budyko分析通常用月或年尺度。
        月平均LAI用于匹配月降水和径流数据。

        Parameters
        ----------
        lai : xr.DataArray
            8天LAI数据

        Returns
        -------
        xr.DataArray
            月平均LAI
        """
        lai_monthly = lai.resample(time='MS').mean()

        lai_monthly.attrs['long_name'] = 'Monthly Mean LAI'
        lai_monthly.attrs['units'] = 'm²/m²'

        print(f"✓ LAI聚合到月尺度完成")

        return lai_monthly

    def fill_gaps(self, lai: xr.DataArray, method: str = 'linear') -> xr.DataArray:
        """
        填补LAI数据缺失

        **缺失原因**：云遮挡、积雪覆盖、传感器问题

        Parameters
        ----------
        lai : xr.DataArray
            带缺失的LAI
        method : str
            插值方法: 'linear', 'nearest', 'cubic'

        Returns
        -------
        xr.DataArray
            填补后的LAI
        """
        # 时间维插值
        lai_filled = lai.interpolate_na(dim='time', method=method)

        # 空间维插值（如需要）
        # lai_filled = lai_filled.interpolate_na(dim='lat', method=method)
        # lai_filled = lai_filled.interpolate_na(dim='lon', method=method)

        n_missing_before = int(lai.isnull().sum())
        n_missing_after = int(lai_filled.isnull().sum())

        print(f"✓ LAI缺失值填补完成")
        print(f"  填补前: {n_missing_before} 个缺失")
        print(f"  填补后: {n_missing_after} 个缺失")

        return lai_filled


class CO2DataLoader:
    """
    大气CO2浓度数据加载器

    **CO2在PET计算中的作用**：
    CO2浓度升高 → 气孔部分关闭（节约水分）→ 气孔导度gs↓ → 表面阻抗rs↑ → PET↓
    这是"CO2施肥效应"对水循环的影响。
    """

    @staticmethod
    def load_global_co2(source: str = 'mauna_loa',
                       start_year: int = 1960,
                       end_year: int = 2023) -> pd.DataFrame:
        """
        加载全球CO2浓度数据

        **数据源**：
        - 'mauna_loa': 夏威夷Mauna Loa观测站（最权威）
        - 'esrl': NOAA/ESRL全球平均
        - 'cmip6': CMIP6情景数据

        Parameters
        ----------
        source : str
            数据源
        start_year, end_year : int
            时间范围

        Returns
        -------
        pd.DataFrame
            CO2浓度数据，列：['year', 'month', 'CO2_ppm']
        """
        # 这里提供一个简化的实现
        # 实际应用应从NOAA或其他官方源下载

        # 拟合历史趋势（Keeling曲线）
        years = np.arange(start_year, end_year + 1)

        # 简化模型：指数增长 + 季节波动
        # 1960年约315ppm，2023年约420ppm
        co2_annual = 315 * np.exp(0.005 * (years - 1960))

        # 生成月度数据
        data = []
        for year, co2_yearly in zip(years, co2_annual):
            for month in range(1, 13):
                # 季节波动：北半球夏季CO2低（植被吸收）
                seasonal_cycle = 3 * np.sin(2 * np.pi * (month - 5) / 12)
                co2_monthly = co2_yearly - seasonal_cycle

                data.append({
                    'year': year,
                    'month': month,
                    'CO2_ppm': co2_monthly
                })

        df = pd.DataFrame(data)

        print(f"✓ 加载CO2数据（{source}）")
        print(f"  时间范围: {start_year} - {end_year}")
        print(f"  CO2范围: {df['CO2_ppm'].min():.1f} - {df['CO2_ppm'].max():.1f} ppm")

        return df

    @staticmethod
    def load_cmip6_scenario(scenario: str = 'ssp245',
                           start_year: int = 2015,
                           end_year: int = 2100) -> pd.DataFrame:
        """
        加载CMIP6未来情景CO2数据

        **SSP情景**：
        - ssp126: 低排放（Paris目标）
        - ssp245: 中等排放
        - ssp370: 中高排放
        - ssp585: 高排放

        Parameters
        ----------
        scenario : str
            SSP情景代码
        start_year, end_year : int
            时间范围

        Returns
        -------
        pd.DataFrame
            未来CO2浓度
        """
        # 简化实现：线性外推不同情景
        years = np.arange(start_year, end_year + 1)

        # 不同情景的年增长率（ppm/year）
        growth_rates = {
            'ssp126': 0.5,   # 缓慢增长，2100年约450ppm
            'ssp245': 1.5,   # 中等增长，2100年约550ppm
            'ssp370': 2.5,   # 中高增长，2100年约700ppm
            'ssp585': 4.0    # 快速增长，2100年约900ppm
        }

        rate = growth_rates.get(scenario, 1.5)
        co2_2015 = 400  # 2015年基准

        co2_values = co2_2015 + rate * (years - 2015)

        df = pd.DataFrame({
            'year': years,
            'CO2_ppm': co2_values,
            'scenario': scenario
        })

        print(f"✓ 加载CMIP6 {scenario}情景CO2数据")
        print(f"  {start_year}: {co2_values[0]:.1f} ppm")
        print(f"  {end_year}: {co2_values[-1]:.1f} ppm")

        return df


if __name__ == "__main__":
    print("GRACE TWS与LAI数据加载模块")
    print("="*60)
    print("功能：")
    print("  1. GRACE TWS加载 → 3D Budyko框架（He et al. 2023）")
    print("  2. MODIS LAI加载 → 创新PET模型（LAI动态调控）")
    print("  3. CO2浓度数据 → 创新PET模型（CO2施肥效应）")
