# src/data_processing/cmip6_processor.py
"""
CMIP6数据处理与情景分析
"""
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

class CMIP6Processor:
    """
    CMIP6模型数据处理器
    
    支持的情景：
    - historical (1850-2014)
    - ssp126 (低排放)
    - ssp245 (中排放)
    - ssp585 (高排放，对应Jaramillo的SSP5-8.5)
    """
    
    SCENARIOS = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    # Jaramillo et al. (2022) 使用的模型
    JARAMILLO_MODELS = [
        'ACCESS-CM2', 'ACCESS-ESM1-5', 'CNRM-CM6-1',
        'EC-Earth3', 'GFDL-ESM4', 'IPSL-CM6A-LR', 'MRI-ESM2-0'
    ]
    
    def __init__(self, 
                 data_dir: Path,
                 models: List[str] = None,
                 scenarios: List[str] = None):
        """
        Parameters
        ----------
        data_dir : Path
            CMIP6数据根目录
        models : List[str], optional
            使用的模型列表
        scenarios : List[str], optional
            使用的情景列表
        """
        self.data_dir = Path(data_dir)
        self.models = models if models else self.JARAMILLO_MODELS
        self.scenarios = scenarios if scenarios else self.SCENARIOS
        
        self.data_cache = {}
    
    def load_model_scenario(self,
                           model: str,
                           scenario: str,
                           variable: str,
                           time_range: Tuple[int, int] = None) -> xr.DataArray:
        """
        加载特定模型和情景的变量
        
        Parameters
        ----------
        model : str
            模型名称
        scenario : str
            情景名称
        variable : str
            变量名 ('pr', 'tas', 'evspsbl', etc.)
        time_range : Tuple[int, int], optional
            时间范围 (start_year, end_year)
            
        Returns
        -------
        xr.DataArray
            变量数据
        """
        # 构建文件路径（根据实际CMIP6文件命名调整）
        file_pattern = f"{variable}*{model}*{scenario}*.nc"
        files = list(self.data_dir.glob(f"**/{file_pattern}"))
        
        if not files:
            raise FileNotFoundError(f"No files found for {model}/{scenario}/{variable}")
        
        # 打开数据集
        ds = xr.open_mfdataset(files, combine='by_coords')
        da = ds[variable]
        
        # 时间筛选
        if time_range:
            start_year, end_year = time_range
            da = da.sel(time=slice(f'{start_year}', f'{end_year}'))
        
        return da
    
    def aggregate_to_catchment(self,
                              data: xr.DataArray,
                              catchment_shapefile: str,
                              catchment_id_field: str = 'catchment_id') -> pd.DataFrame:
        """
        将栅格数据聚合到流域尺度
        
        Parameters
        ----------
        data : xr.DataArray
            栅格数据
        catchment_shapefile : str
            流域边界矢量文件
        catchment_id_field : str
            流域ID字段名
            
        Returns
        -------
        pd.DataFrame
            流域尺度时间序列
        """
        import geopandas as gpd
        from rasterstats import zonal_stats
        
        # 读取流域边界
        catchments = gpd.read_file(catchment_shapefile)
        
        # 逐时间步聚合
        results = []
        
        for time_idx in range(len(data.time)):
            time_slice = data.isel(time=time_idx)
            
            # 转换为numpy数组
            array = time_slice.values
            transform = data.rio.transform()
            
            # 区域统计
            stats = zonal_stats(
                catchments,
                array,
                affine=transform,
                stats=['mean', 'count'],
                nodata=np.nan
            )
            
            # 提取结果
            for idx, stat in enumerate(stats):
                results.append({
                    'catchment_id': catchments.iloc[idx][catchment_id_field],
                    'time': time_slice.time.values,
                    'value': stat['mean']
                })
        
        df = pd.DataFrame(results)
        df = df.pivot(index='time', columns='catchment_id', values='value')
        
        return df
    
    def calculate_budyko_inputs(self,
                               model: str,
                               scenario: str,
                               catchments: gpd.GeoDataFrame,
                               period: Tuple[int, int],
                               pet_method: str = 'hargreaves') -> pd.DataFrame:
        """
        计算Budyko框架所需的IA和IE
        
        Parameters
        ----------
        model : str
            CMIP6模型
        scenario : str
            排放情景
        catchments : gpd.GeoDataFrame
            流域边界
        period : Tuple[int, int]
            时间段 (start_year, end_year)
        pet_method : str
            PET计算方法
            
        Returns
        -------
        pd.DataFrame
            每个流域的IA和IE
        """
        start_year, end_year = period
        
        # 1. 加载降水
        pr = self.load_model_scenario(model, scenario, 'pr', period)
        pr_catchment = self.aggregate_to_catchment(pr, catchments)
        
        # 2. 加载温度（用于PET）
        tas = self.load_model_scenario(model, scenario, 'tas', period)
        tas_catchment = self.aggregate_to_catchment(tas, catchments)
        
        tasmax = self.load_model_scenario(model, scenario, 'tasmax', period)
        tasmax_catchment = self.aggregate_to_catchment(tasmax, catchments)
        
        tasmin = self.load_model_scenario(model, scenario, 'tasmin', period)
        tasmin_catchment = self.aggregate_to_catchment(tasmin, catchments)
        
        # 3. 计算PET
        from ..models.pet_models import PETModelFactory
        pet_model = PETModelFactory.create(pet_method)
        
        pet_results = {}
        for catchment_id in pr_catchment.columns:
            # 提取该流域数据
            catchment_lat = catchments[catchments['catchment_id'] == catchment_id].geometry.centroid.y.iloc[0]
            
            data = {
                'temp_avg': tas_catchment[catchment_id].values - 273.15,  # K to °C
                'temp_max': tasmax_catchment[catchment_id].values - 273.15,
                'temp_min': tasmin_catchment[catchment_id].values - 273.15,
                'latitude': catchment_lat,
                'day_of_year': pd.to_datetime(tas_catchment.index).dayofyear.values
            }
            
            pet = pet_model.calculate(**data)
            pet_results[catchment_id] = pet
        
        pet_df = pd.DataFrame(pet_results, index=pr_catchment.index)
        
        # 4. 加载蒸散发（如果有）
        try:
            et = self.load_model_scenario(model, scenario, 'evspsbl', period)
            et_catchment = self.aggregate_to_catchment(et, catchments)
        except:
            # 如果没有直接的ET数据，使用水量平衡估算
            warnings.warn("No ET data, estimating from water balance")
            # 这里需要径流数据，或者使用Budyko假设
            et_catchment = None
        
        # 5. 计算年度IA和IE
        results = []
        for catchment_id in pr_catchment.columns:
            pr_annual = pr_catchment[catchment_id].resample('Y').sum()
            pet_annual = pet_df[catchment_id].resample('Y').sum()
            
            if et_catchment is not None:
                et_annual = et_catchment[catchment_id].resample('Y').sum()
            else:
                # 使用Budyko假设估算
                ia_annual = pet_annual / pr_annual
                # 这里需要假设一个ω，或者使用历史期拟合的ω
                # 简化：使用全球平均ω≈2.6
                from ..budyko.curves import BudykoCurves
                ie_annual_estimated = BudykoCurves.tixeront_fu(ia_annual, omega=2.6)
                et_annual = ie_annual_estimated * pr_annual
            
            # 计算多年平均
            ia_mean = (pet_annual / pr_annual).mean()
            ie_mean = (et_annual / pr_annual).mean()
            
            results.append({
                'catchment_id': catchment_id,
                'model': model,
                'scenario': scenario,
                'period': f"{start_year}-{end_year}",
                'IA': ia_mean,
                'IE': ie_mean,
                'P_mean': pr_annual.mean(),
                'PET_mean': pet_annual.mean(),
                'ET_mean': et_annual.mean()
            })
        
        return pd.DataFrame(results)
    
    def multi_model_ensemble(self,
                            results_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        多模型集成
        
        Parameters
        ----------
        results_list : List[pd.DataFrame]
            各模型结果列表
            
        Returns
        -------
        pd.DataFrame
            集成结果（包含均值、标准差等）
        """
        # 合并所有模型
        all_results = pd.concat(results_list, ignore_index=True)
        
        # 按流域和情景分组
        ensemble = all_results.groupby(['catchment_id', 'scenario', 'period']).agg({
            'IA': ['mean', 'std', 'min', 'max'],
            'IE': ['mean', 'std', 'min', 'max'],
            'P_mean': 'mean',
            'PET_mean': 'mean',
            'ET_mean': 'mean'
        }).reset_index()
        
        # 展平多级列名
        ensemble.columns = ['_'.join(col).strip('_') for col in ensemble.columns.values]
        
        return ensemble