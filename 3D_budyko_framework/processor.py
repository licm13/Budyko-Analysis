"""
Data Processing Module for Budyko Framework
处理流域数据、气象数据、遥感数据
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
    
    处理中国6000+小流域的径流和气象数据
    """
    
    def __init__(self, basin_shapefile: Optional[str] = None):
        """
        初始化
        
        Parameters:
        -----------
        basin_shapefile : str, optional
            流域边界shapefile路径
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
        加载流域径流观测数据
        
        Parameters:
        -----------
        file_path : str
            数据文件路径
        file_format : str
            文件格式: 'csv', 'excel', 'netcdf'
        date_column : str
            日期列名
        basin_column : str
            流域ID列名
        runoff_column : str
            径流列名
            
        Returns:
        --------
        data : DataFrame
            整理后的径流数据
        """
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
        data = data.sort_values(['basin_id', 'date'])
        
        # 质量控制
        data = self._quality_control_runoff(data)
        
        self.runoff_data = data
        return data
    
    def _quality_control_runoff(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        径流数据质量控制
        
        - 移除负值
        - 标记异常值
        - 插补缺失值（可选）
        """
        # 移除负值
        data.loc[data['Q'] < 0, 'Q'] = np.nan
        
        # 检测异常值（使用IQR方法）
        for basin_id in data['basin_id'].unique():
            mask = data['basin_id'] == basin_id
            Q = data.loc[mask, 'Q']
            
            Q1 = Q.quantile(0.25)
            Q3 = Q.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = (Q < lower_bound) | (Q > upper_bound)
            data.loc[mask & outliers, 'Q_flag'] = 'outlier'
        
        return data
    
    def load_cmfd_gridded_data(self,
                              cmfd_dir: str,
                              variables: List[str] = ['prec', 'temp', 'wind', 'pres', 'shum'],
                              start_year: int = 1960,
                              end_year: int = 2020) -> xr.Dataset:
        """
        加载CMFD 0.1°格点气象数据
        
        Parameters:
        -----------
        cmfd_dir : str
            CMFD数据目录
        variables : list
            需要的变量
        start_year, end_year : int
            时间范围
            
        Returns:
        --------
        ds : Dataset
            CMFD数据集
        """
        cmfd_path = Path(cmfd_dir)
        
        datasets = []
        
        for var in variables:
            var_files = []
            
            for year in range(start_year, end_year + 1):
                # CMFD文件命名规则可能需要调整
                pattern = f"*{var}*{year}*.nc"
                files = list(cmfd_path.glob(pattern))
                
                if files:
                    var_files.extend(files)
            
            if var_files:
                ds_var = xr.open_mfdataset(var_files, combine='by_coords')
                datasets.append(ds_var)
        
        # 合并所有变量
        if datasets:
            ds = xr.merge(datasets)
            self.met_data = ds
            return ds
        else:
            raise FileNotFoundError(f"在 {cmfd_dir} 未找到CMFD数据")
    
    def extract_basin_average(self, 
                             gridded_data: xr.Dataset,
                             basin_geometries: gpd.GeoDataFrame,
                             method: str = 'area_weighted') -> pd.DataFrame:
        """
        提取流域平均气象数据
        
        Parameters:
        -----------
        gridded_data : Dataset
            格点气象数据
        basin_geometries : GeoDataFrame
            流域矢量数据
        method : str
            平均方法: 'simple', 'area_weighted'
            
        Returns:
        --------
        basin_avg : DataFrame
            流域平均数据
        """
        try:
            import geopandas as gpd
            import rasterio
            from rasterio import features
        except ImportError:
            raise ImportError("需要安装geopandas和rasterio: pip install geopandas rasterio")
        
        results = []
        
        for idx, basin in basin_geometries.iterrows():
            basin_id = basin['basin_id']
            geometry = basin.geometry
            
            # 裁剪格点数据到流域范围
            bbox = geometry.bounds
            subset = gridded_data.sel(
                lon=slice(bbox[0], bbox[2]),
                lat=slice(bbox[1], bbox[3])
            )
            
            # 创建mask
            mask = self._create_basin_mask(subset, geometry)
            
            # 计算平均
            if method == 'simple':
                basin_mean = subset.where(mask).mean(dim=['lon', 'lat'])
            elif method == 'area_weighted':
                weights = self._calculate_area_weights(subset.lat)
                basin_mean = subset.where(mask).weighted(weights).mean(dim=['lon', 'lat'])
            
            basin_df = basin_mean.to_dataframe()
            basin_df['basin_id'] = basin_id
            
            results.append(basin_df)
        
        return pd.concat(results, ignore_index=True)
    
    @staticmethod
    def _create_basin_mask(dataset: xr.Dataset, geometry) -> xr.DataArray:
        """创建流域mask"""
        # 简化实现，实际应用需要更精确的栅格化
        from rasterio import features
        from affine import Affine
        
        # 创建仿射变换
        lon = dataset.lon.values
        lat = dataset.lat.values
        
        res_lon = np.mean(np.diff(lon))
        res_lat = np.mean(np.diff(lat))
        
        transform = Affine.translation(lon[0], lat[0]) * Affine.scale(res_lon, res_lat)
        
        # 栅格化
        mask_arr = features.geometry_mask(
            [geometry],
            out_shape=(len(lat), len(lon)),
            transform=transform,
            invert=True
        )
        
        mask = xr.DataArray(
            mask_arr,
            coords={'lat': lat, 'lon': lon},
            dims=['lat', 'lon']
        )
        
        return mask
    
    @staticmethod
    def _calculate_area_weights(lat: xr.DataArray) -> xr.DataArray:
        """
        计算面积权重（考虑纬度）
        
        面积 ∝ cos(lat)
        """
        weights = np.cos(np.deg2rad(lat))
        weights = weights / weights.sum()
        return weights
    
    def aggregate_to_timescale(self,
                               data: pd.DataFrame,
                               time_scale: str = 'annual',
                               variables: List[str] = ['Q', 'P', 'T']) -> pd.DataFrame:
        """
        时间尺度聚合
        
        Parameters:
        -----------
        data : DataFrame
            原始数据（日或月尺度）
        time_scale : str
            目标时间尺度: 'annual', 'seasonal', 'monthly'
        variables : list
            需要聚合的变量
            
        Returns:
        --------
        aggregated : DataFrame
            聚合后的数据
        """
        if 'date' not in data.columns:
            raise ValueError("数据需要包含'date'列")
        
        data = data.copy()
        data['year'] = data['date'].dt.year
        
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
                if var in ['P', 'Q', 'PET', 'ET']:  # 累积变量
                    agg_dict[var] = 'sum'
                elif var in ['T', 'RH', 'u2']:     # 平均变量
                    agg_dict[var] = 'mean'
        
        aggregated = data.groupby(group_cols).agg(agg_dict).reset_index()
        
        return aggregated
    
    @staticmethod
    def _month_to_season(month: int) -> str:
        """月份转季节"""
        if month in [12, 1, 2]:
            return 'DJF'
        elif month in [3, 4, 5]:
            return 'MAM'
        elif month in [6, 7, 8]:
            return 'JJA'
        else:  # 9, 10, 11
            return 'SON'
    
    def filter_basins(self,
                     data: pd.DataFrame,
                     min_data_years: int = 10,
                     min_basin_area: float = 100,  # km²
                     exclude_regulated: bool = True) -> pd.DataFrame:
        """
        筛选满足条件的流域
        
        Parameters:
        -----------
        data : DataFrame
            流域数据
        min_data_years : int
            最少数据年限
        min_basin_area : float
            最小流域面积
        exclude_regulated : bool
            是否排除受调控流域
            
        Returns:
        --------
        filtered : DataFrame
            筛选后的数据
        """
        # 计算每个流域的数据年限
        data_years = data.groupby('basin_id')['year'].nunique()
        valid_basins = data_years[data_years >= min_data_years].index
        
        filtered = data[data['basin_id'].isin(valid_basins)]
        
        print(f"筛选前流域数: {data['basin_id'].nunique()}")
        print(f"筛选后流域数: {filtered['basin_id'].nunique()}")
        
        return filtered


class RemoteSensingDataProcessor:
    """
    遥感数据处理器
    
    处理GRACE、MODIS等遥感数据
    """
    
    def __init__(self):
        self.grace_data = None
        self.ndvi_data = None
        
    def load_grace_tws(self,
                      grace_dir: str,
                      solution: str = 'CSR',
                      release: str = 'RL06') -> xr.Dataset:
        """
        加载GRACE陆地水储量数据
        
        Parameters:
        -----------
        grace_dir : str
            GRACE数据目录
        solution : str
            解算中心: 'CSR', 'JPL', 'GFZ'
        release : str
            数据版本: 'RL06'
            
        Returns:
        --------
        ds : Dataset
            GRACE TWS数据
        """
        grace_path = Path(grace_dir)
        
        # 查找GRACE文件
        pattern = f"*{solution}*{release}*.nc"
        files = list(grace_path.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"未找到GRACE数据: {pattern}")
        
        # 读取并合并
        ds = xr.open_mfdataset(files, combine='by_coords')
        
        # 标准化变量名
        if 'lwe_thickness' in ds:
            ds = ds.rename({'lwe_thickness': 'TWS'})
        
        self.grace_data = ds
        return ds
    
    def calculate_tws_anomaly(self, 
                             tws: xr.DataArray,
                             baseline_period: Tuple[str, str] = ('2004-01', '2009-12')) -> xr.DataArray:
        """
        计算TWS异常
        
        Parameters:
        -----------
        tws : DataArray
            TWS数据
        baseline_period : tuple
            基准期 (start, end)
            
        Returns:
        --------
        tws_anomaly : DataArray
            TWS异常
        """
        baseline = tws.sel(time=slice(baseline_period[0], baseline_period[1]))
        tws_mean = baseline.mean(dim='time')
        
        tws_anomaly = tws - tws_mean
        
        return tws_anomaly
    
    def load_modis_lai(self,
                      modis_dir: str,
                      product: str = 'MOD15A2H',
                      version: str = '006') -> xr.Dataset:
        """
        加载MODIS LAI数据
        
        Parameters:
        -----------
        modis_dir : str
            MODIS数据目录
        product : str
            产品名称
        version : str
            版本号
            
        Returns:
        --------
        ds : Dataset
            LAI数据
        """
        modis_path = Path(modis_dir)
        
        # 查找MODIS文件
        pattern = f"*{product}*{version}*.hdf"
        files = list(modis_path.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"未找到MODIS数据: {pattern}")
        
        # 使用rioxarray读取HDF
        try:
            import rioxarray
            datasets = []
            
            for f in files:
                ds = rioxarray.open_rasterio(f)
                datasets.append(ds)
            
            ds = xr.concat(datasets, dim='time')
            
        except ImportError:
            raise ImportError("需要安装rioxarray: pip install rioxarray")
        
        return ds


class SnowDataProcessor:
    """
    积雪数据处理器
    
    研究积雪对Budyko关系的影响
    """
    
    def __init__(self):
        self.snow_data = None
        
    def calculate_snowmelt_runoff(self,
                                  snow_depth: np.ndarray,
                                  temperature: np.ndarray,
                                  melt_factor: float = 2.5) -> np.ndarray:
        """
        计算融雪径流
        
        使用度日模型:
        Melt = melt_factor * (T - T_threshold)  if T > T_threshold
        
        Parameters:
        -----------
        snow_depth : array
            雪深 [mm]
        temperature : array
            气温 [°C]
        melt_factor : float
            融雪系数 [mm/°C/day]
            
        Returns:
        --------
        melt_runoff : array
            融雪径流 [mm]
        """
        T_threshold = 0  # 融雪阈值温度
        
        # 计算融雪
        positive_temp = np.maximum(temperature - T_threshold, 0)
        potential_melt = melt_factor * positive_temp
        
        # 融雪不能超过积雪量
        melt_runoff = np.minimum(potential_melt, snow_depth)
        
        return melt_runoff
    
    def identify_snow_dominated_basins(self,
                                      data: pd.DataFrame,
                                      snow_ratio_threshold: float = 0.2) -> List:
        """
        识别积雪主导的流域
        
        Parameters:
        -----------
        data : DataFrame
            包含snow_ratio列的数据
        snow_ratio_threshold : float
            积雪径流比例阈值
            
        Returns:
        --------
        snow_basins : list
            积雪主导的流域ID列表
        """
        if 'snow_ratio' not in data.columns:
            raise ValueError("数据需要包含'snow_ratio'列")
        
        snow_basins = data[data['snow_ratio'] > snow_ratio_threshold]['basin_id'].unique().tolist()
        
        print(f"识别出 {len(snow_basins)} 个积雪主导流域")
        
        return snow_basins


if __name__ == "__main__":
    print("数据处理模块加载成功")
    print("可用类:")
    print("- BasinDataProcessor: 流域数据处理")
    print("- RemoteSensingDataProcessor: 遥感数据处理")
    print("- SnowDataProcessor: 积雪数据处理")