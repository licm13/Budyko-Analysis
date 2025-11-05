# src/data_processing/basin_processor.py
"""
流域数据处理模块

**核心功能**：
1. 加载流域径流观测数据（Q） - 这是Budyko分析的基石！
2. 加载流域边界矢量数据（Shapefile）
3. 从格点气象数据中提取流域平均值
4. 时间尺度聚合（日→月→年）
5. 数据质量控制

**科学背景**：
观测径流Q是整个Budyko分析的锚点。通过水量平衡方程 EA = P - Q，
我们可以直接计算实际蒸散发EA，进而得到蒸发指数 IE = EA/P。
这避免了直接测量EA的困难，是Budyko框架的核心优势之一。
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings


class BasinDataProcessor:
    """
    流域数据处理器

    处理：
    - 站点径流数据（多流域、多年）
    - 格点气象数据（CMFD、ERA5等）
    - 流域边界矢量数据

    输出：
    - 流域尺度的年/月时间序列数据
    """

    def __init__(self, basin_shapefile: Optional[str] = None):
        """
        初始化流域数据处理器

        Parameters
        ----------
        basin_shapefile : str, optional
            流域边界shapefile路径。如提供，可用于空间聚合。
        """
        self.basin_shapefile = basin_shapefile
        self.basins = None
        self.runoff_data = None
        self.met_data = None

    def load_runoff_observations(self,
                                 file_path: str,
                                 file_format: str = 'csv',
                                 date_column: str = 'date',
                                 basin_column: str = 'basin_id',
                                 runoff_column: str = 'runoff') -> pd.DataFrame:
        """
        加载流域径流观测数据（核心数据！）

        **科学重要性**：
        径流Q是Budyko分析的基石。它来自水文站的实测数据，
        通过水量平衡 EA = P - Q 计算实际蒸散发，锚定流域在Budyko空间的真实位置。

        Parameters
        ----------
        file_path : str
            数据文件路径
        file_format : str
            文件格式: 'csv', 'excel', 'netcdf'
        date_column : str
            日期列名
        basin_column : str
            流域ID列名
        runoff_column : str
            径流列名（单位应为mm或mm/day）

        Returns
        -------
        pd.DataFrame
            标准化的径流数据，列名：['date', 'basin_id', 'Q']
        """
        # 读取数据
        if file_format == 'csv':
            data = pd.read_csv(file_path, parse_dates=[date_column])
        elif file_format == 'excel':
            data = pd.read_excel(file_path, parse_dates=[date_column])
        elif file_format == 'netcdf':
            ds = xr.open_dataset(file_path)
            data = ds.to_dataframe().reset_index()
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")

        # 标准化列名
        data = data.rename(columns={
            date_column: 'date',
            basin_column: 'basin_id',
            runoff_column: 'Q'
        })

        # 确保时间排序
        data = data.sort_values(['basin_id', 'date']).reset_index(drop=True)

        # 数据质量控制
        data = self._quality_control_runoff(data)

        self.runoff_data = data
        print(f"✓ 成功加载 {data['basin_id'].nunique()} 个流域的径流数据")
        print(f"  时间范围: {data['date'].min()} 至 {data['date'].max()}")

        return data

    def _quality_control_runoff(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        径流数据质量控制

        **质控步骤**：
        1. 移除负值（物理上不合理）
        2. 检测异常值（使用IQR方法）
        3. 标记质量flag但不删除数据（保留原始信息）

        Parameters
        ----------
        data : pd.DataFrame
            原始径流数据

        Returns
        -------
        pd.DataFrame
            添加'Q_flag'列的数据
        """
        # 初始化质量标志
        data['Q_flag'] = 'good'

        # 1. 标记负值
        negative_mask = data['Q'] < 0
        if negative_mask.any():
            data.loc[negative_mask, 'Q_flag'] = 'negative'
            data.loc[negative_mask, 'Q'] = np.nan
            print(f"  警告: 发现 {negative_mask.sum()} 个负值，已设为NaN")

        # 2. 逐流域检测异常值（IQR方法） - Optimized using groupby
        grouped = data.groupby('basin_id')['Q']
        
        # Vectorized percentile calculation
        q25 = grouped.transform(lambda x: np.nanpercentile(x, 25))
        q75 = grouped.transform(lambda x: np.nanpercentile(x, 75))
        iqr = q75 - q25
        
        # Vectorized outlier detection
        lower_bound = q25 - 3 * iqr
        upper_bound = q75 + 3 * iqr
        outlier_mask = (data['Q'] < lower_bound) | (data['Q'] > upper_bound)
        data.loc[outlier_mask, 'Q_flag'] = 'outlier'

        n_outliers = (data['Q_flag'] == 'outlier').sum()
        if n_outliers > 0:
            print(f"  信息: 标记了 {n_outliers} 个可能的异常值")

        return data

    def load_cmfd_gridded_data(self,
                              cmfd_dir: str,
                              variables: List[str] = ['prec', 'temp', 'wind', 'pres', 'shum'],
                              start_year: int = 1960,
                              end_year: int = 2020) -> xr.Dataset:
        """
        加载CMFD（中国气象同化驱动数据集）0.1°格点数据

        **数据说明**：
        CMFD提供高分辨率气象要素，用于：
        - 降水P：水量平衡的输入
        - 温度、风速、辐射等：计算PET的驱动变量

        Parameters
        ----------
        cmfd_dir : str
            CMFD数据目录
        variables : list
            需要的变量列表
        start_year, end_year : int
            时间范围

        Returns
        -------
        xr.Dataset
            合并后的CMFD数据集
        """
        cmfd_path = Path(cmfd_dir)

        if not cmfd_path.exists():
            raise FileNotFoundError(f"CMFD目录不存在: {cmfd_dir}")

        datasets = []

        for var in variables:
            var_files = []

            for year in range(start_year, end_year + 1):
                # CMFD文件命名模式（可能需要根据实际情况调整）
                pattern = f"*{var}*{year}*.nc"
                files = list(cmfd_path.glob(pattern))

                if files:
                    var_files.extend(files)

            if var_files:
                print(f"  读取变量 {var}: {len(var_files)} 个文件")
                ds_var = xr.open_mfdataset(var_files, combine='by_coords')
                datasets.append(ds_var)
            else:
                warnings.warn(f"未找到变量 {var} 的数据文件")

        # 合并所有变量
        if datasets:
            ds = xr.merge(datasets)
            self.met_data = ds
            print(f"✓ 成功加载CMFD数据，变量: {list(ds.data_vars)}")
            return ds
        else:
            raise FileNotFoundError(f"在 {cmfd_dir} 未找到任何CMFD数据")

    def extract_basin_average(self,
                             gridded_data: xr.Dataset,
                             basin_geometries: 'gpd.GeoDataFrame',
                             method: str = 'area_weighted') -> pd.DataFrame:
        """
        从格点数据中提取流域平均值

        **科学意义**：
        将0.1°格点气象数据聚合到流域尺度，匹配站点径流Q的空间尺度，
        确保水量平衡分析的空间一致性。

        Parameters
        ----------
        gridded_data : xr.Dataset
            格点气象数据（如CMFD）
        basin_geometries : gpd.GeoDataFrame
            流域边界矢量数据，需包含'basin_id'列
        method : str
            平均方法:
            - 'simple': 简单平均
            - 'area_weighted': 面积加权平均（推荐，考虑纬度变化）

        Returns
        -------
        pd.DataFrame
            流域平均数据
        """
        try:
            import geopandas as gpd
            import rasterio
            from rasterio import features
        except ImportError:
            raise ImportError(
                "需要安装geopandas和rasterio:\n"
                "  pip install geopandas rasterio"
            )

        results = []

        print(f"开始提取 {len(basin_geometries)} 个流域的平均数据...")

        for idx, basin in basin_geometries.iterrows():
            basin_id = basin['basin_id']
            geometry = basin.geometry

            # 1. 裁剪到流域范围
            bbox = geometry.bounds  # (minx, miny, maxx, maxy)
            subset = gridded_data.sel(
                lon=slice(bbox[0], bbox[2]),
                lat=slice(bbox[1], bbox[3])
            )

            # 2. 创建流域mask
            mask = self._create_basin_mask(subset, geometry)

            # 3. 计算平均
            if method == 'simple':
                basin_mean = subset.where(mask).mean(dim=['lon', 'lat'])
            elif method == 'area_weighted':
                # 面积权重（纬度加权）
                weights = self._calculate_area_weights(subset.lat)
                basin_mean = subset.where(mask).weighted(weights).mean(dim=['lon', 'lat'])
            else:
                raise ValueError(f"未知方法: {method}")

            # 转为DataFrame
            basin_df = basin_mean.to_dataframe().reset_index()
            basin_df['basin_id'] = basin_id

            results.append(basin_df)

            if (idx + 1) % 100 == 0:
                print(f"  进度: {idx + 1}/{len(basin_geometries)}")

        result_df = pd.concat(results, ignore_index=True)
        print(f"✓ 完成流域平均提取")

        return result_df

    @staticmethod
    def _create_basin_mask(dataset: xr.Dataset, geometry) -> xr.DataArray:
        """
        创建流域mask（栅格化矢量边界）

        Parameters
        ----------
        dataset : xr.Dataset
            目标格点数据集
        geometry : shapely.geometry
            流域几何边界

        Returns
        -------
        xr.DataArray
            布尔mask数组
        """
        from rasterio import features
        from affine import Affine

        # 获取坐标
        lon = dataset.lon.values
        lat = dataset.lat.values

        # 计算分辨率
        res_lon = np.mean(np.diff(lon))
        res_lat = np.mean(np.diff(lat))

        # 创建仿射变换
        transform = Affine.translation(lon[0] - res_lon/2, lat[0] - res_lat/2) * \
                   Affine.scale(res_lon, res_lat)

        # 栅格化几何
        mask_arr = features.geometry_mask(
            [geometry],
            out_shape=(len(lat), len(lon)),
            transform=transform,
            invert=True  # True表示流域内
        )

        # 转为DataArray
        mask = xr.DataArray(
            mask_arr,
            coords={'lat': lat, 'lon': lon},
            dims=['lat', 'lon']
        )

        return mask

    @staticmethod
    def _calculate_area_weights(lat: xr.DataArray) -> xr.DataArray:
        """
        计算面积权重（考虑球面纬度效应）

        **物理原理**：
        在球面上，单位经纬度格网的实际面积随纬度变化：
            A(φ) ∝ cos(φ)
        因此需要纬度加权以获得正确的面积平均。

        Parameters
        ----------
        lat : xr.DataArray
            纬度坐标

        Returns
        -------
        xr.DataArray
            归一化的面积权重
        """
        weights = np.cos(np.deg2rad(lat))
        weights = weights / weights.sum()  # 归一化
        return weights

    def aggregate_to_timescale(self,
                               data: pd.DataFrame,
                               time_scale: str = 'annual',
                               variables: List[str] = ['Q', 'P', 'T']) -> pd.DataFrame:
        """
        时间尺度聚合（日/月 → 年/季节）

        **Budyko分析的时间尺度**：
        Budyko假设在长时间尺度（年际或多年平均）上 ΔS≈0，
        因此通常使用年度数据。但季节尺度分析（湿季/干季）也很有价值。

        Parameters
        ----------
        data : pd.DataFrame
            原始数据，需包含'date'列
        time_scale : str
            目标时间尺度:
            - 'annual': 年度
            - 'seasonal': 季节（DJF, MAM, JJA, SON）
            - 'monthly': 月度
        variables : list
            需要聚合的变量

        Returns
        -------
        pd.DataFrame
            聚合后的数据
        """
        if 'date' not in data.columns:
            raise ValueError("数据需要包含'date'列")

        data = data.copy()
        data['year'] = data['date'].dt.year

        # 定义分组列
        if time_scale == 'annual':
            group_cols = ['basin_id', 'year']
        elif time_scale == 'seasonal':
            data['season'] = data['date'].dt.month.map(self._month_to_season)
            group_cols = ['basin_id', 'year', 'season']
        elif time_scale == 'monthly':
            data['month'] = data['date'].dt.month
            group_cols = ['basin_id', 'year', 'month']
        else:
            raise ValueError(f"不支持的时间尺度: {time_scale}")

        # 聚合规则
        agg_dict = {}
        for var in variables:
            if var in data.columns:
                # 水量变量：累积（sum）
                if var in ['P', 'Q', 'PET', 'ET', 'EA']:
                    agg_dict[var] = 'sum'
                # 状态变量：平均（mean）
                elif var in ['T', 'RH', 'u2', 'LAI', 'CO2']:
                    agg_dict[var] = 'mean'
                else:
                    # 默认平均
                    agg_dict[var] = 'mean'

        aggregated = data.groupby(group_cols).agg(agg_dict).reset_index()

        print(f"✓ 聚合到{time_scale}尺度，共 {len(aggregated)} 条记录")

        return aggregated

    @staticmethod
    def _month_to_season(month: int) -> str:
        """
        月份转季节（北半球定义）

        Returns
        -------
        str
            季节代码: 'DJF', 'MAM', 'JJA', 'SON'
        """
        if month in [12, 1, 2]:
            return 'DJF'  # 冬季
        elif month in [3, 4, 5]:
            return 'MAM'  # 春季
        elif month in [6, 7, 8]:
            return 'JJA'  # 夏季
        else:  # 9, 10, 11
            return 'SON'  # 秋季

    def filter_basins(self,
                     data: pd.DataFrame,
                     min_data_years: int = 10,
                     min_basin_area: float = 100,  # km²
                     exclude_regulated: bool = True) -> pd.DataFrame:
        """
        筛选满足条件的流域

        **质量要求**：
        - 足够的数据年限（≥10年）以捕捉年际变率
        - 避免过小流域（易受局地扰动影响）
        - 排除强人类调控流域（如有大坝的流域）

        Parameters
        ----------
        data : pd.DataFrame
            流域数据
        min_data_years : int
            最少数据年限
        min_basin_area : float
            最小流域面积 [km²]
        exclude_regulated : bool
            是否排除受调控流域（需数据中有'regulated'标志）

        Returns
        -------
        pd.DataFrame
            筛选后的数据
        """
        # 1. 数据年限筛选
        data_years = data.groupby('basin_id')['year'].nunique()
        valid_basins = set(data_years[data_years >= min_data_years].index)

        # 2. 流域面积筛选（如果有'area'列）
        if 'area' in data.columns:
            area_valid = set(data[data['area'] >= min_basin_area]['basin_id'].unique())
            valid_basins = valid_basins.intersection(area_valid)

        # 3. 排除调控流域（如果有'regulated'列）
        if exclude_regulated and 'regulated' in data.columns:
            unregulated = set(data[~data['regulated']]['basin_id'].unique())
            valid_basins = valid_basins.intersection(unregulated)

        # 应用筛选
        filtered = data[data['basin_id'].isin(valid_basins)].copy()

        print(f"\n流域筛选结果:")
        print(f"  筛选前: {data['basin_id'].nunique()} 个流域")
        print(f"  筛选后: {filtered['basin_id'].nunique()} 个流域")
        print(f"  筛选条件: 数据≥{min_data_years}年, 面积≥{min_basin_area}km²")

        return filtered


if __name__ == "__main__":
    print("流域数据处理模块")
    print("="*50)
    print("功能：")
    print("  1. 加载径流观测数据（Q - Budyko分析的基石）")
    print("  2. 加载格点气象数据（CMFD等）")
    print("  3. 提取流域平均值（空间聚合）")
    print("  4. 时间尺度聚合（日→月→年）")
    print("  5. 数据质量控制与流域筛选")
