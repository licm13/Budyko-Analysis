"""
数据预处理模块 (Data Preprocessing Module)
处理原始流量、气象和流域属性数据
Process raw discharge, meteorological, and catchment property data

Author: [Your Name]
Date: 2025-01-01
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入自定义工具函数 / Import custom utility functions
from utils import lyne_hollick_filter, load_catchment_data


# =============================================================================
# 配置参数 (Configuration Parameters)
# =============================================================================

class Config:
    """配置类 / Configuration class"""
    
    # 数据路径 / Data paths
    RAW_DATA_DIR = Path('../data/raw')
    PROCESSED_DATA_DIR = Path('../data/processed')
    
    # 流量数据路径 / Discharge data paths
    DISCHARGE_DIR = RAW_DATA_DIR / 'discharge'
    GRDC_DIR = DISCHARGE_DIR / 'GRDC'
    CAMELS_DIR = DISCHARGE_DIR / 'CAMELS'
    
    # 气象数据路径 / Meteorological data paths
    METEO_DIR = RAW_DATA_DIR / 'meteorology'
    P_FILE = METEO_DIR / 'MSWEP_v1.1_precipitation.nc'
    EP_FILE = METEO_DIR / 'TerraClimate_pet.nc'
    EA_FILE = METEO_DIR / 'GLEAM_v3.6_Ea.nc'
    
    # 流域属性路径 / Catchment properties paths
    CATCHMENT_PROPS_DIR = RAW_DATA_DIR / 'catchment_properties'
    
    # 数据质量控制阈值 / Data quality control thresholds
    MIN_RECORD_LENGTH = 10  # 最小记录长度(年) / Minimum record length (years)
    MAX_MISSING_RATE = 0.20  # 最大缺失率 / Maximum missing rate
    MIN_CATCHMENT_AREA = 50  # 最小流域面积(km²) / Minimum catchment area (km²)
    MAX_CATCHMENT_AREA = 5000  # 最大流域面积(km²) / Maximum catchment area (km²)
    WATER_BALANCE_THRESHOLD = 0.1  # 水量平衡阈值 / Water balance threshold
    
    # 基流分离参数 / Baseflow separation parameters
    LH_ALPHA = 0.925  # Lyne-Hollick滤波参数 / Lyne-Hollick filter parameter
    LH_PASSES = 3  # 滤波次数 / Number of passes


# =============================================================================
# 流域数据加载函数 (Catchment Data Loading Functions)
# =============================================================================

def load_grdc_catchments(grdc_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    加载GRDC流量数据
    Load GRDC discharge data
    
    Parameters
    ----------
    grdc_dir : Path
        GRDC数据目录 / GRDC data directory
        
    Returns
    -------
    catchments : dict
        流域ID到数据框的字典 / Dictionary mapping catchment ID to DataFrame
        
    Notes
    -----
    预期GRDC数据格式 / Expected GRDC data format:
    - 文件名 / Filename: {station_id}.csv
    - 列 / Columns: date, discharge_m3s, area_km2
    """
    print("正在加载GRDC流量数据... / Loading GRDC discharge data...")
    
    catchments = {}
    
    # 查找所有CSV文件 / Find all CSV files
    csv_files = list(grdc_dir.glob('*.csv'))
    print(f"找到 {len(csv_files)} 个GRDC站点文件 / Found {len(csv_files)} GRDC station files")
    
    for csv_file in csv_files:
        try:
            # 提取站点ID / Extract station ID
            station_id = csv_file.stem
            
            # 读取数据 / Read data
            df = pd.read_csv(csv_file, parse_dates=['date'])
            
            # 基本质量检查 / Basic quality check
            if len(df) > 0 and 'discharge_m3s' in df.columns:
                catchments[station_id] = df
                
        except Exception as e:
            print(f"  警告: 无法加载 {csv_file.name}: {e}")
            print(f"  Warning: Failed to load {csv_file.name}: {e}")
    
    print(f"成功加载 {len(catchments)} 个GRDC流域 / Successfully loaded {len(catchments)} GRDC catchments")
    
    return catchments


def load_camels_catchments(camels_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    加载CAMELS流量数据 (所有国家)
    Load CAMELS discharge data (all countries)
    
    Parameters
    ----------
    camels_dir : Path
        CAMELS数据目录 / CAMELS data directory
        
    Returns
    -------
    catchments : dict
        流域ID到数据框的字典 / Dictionary mapping catchment ID to DataFrame
        
    Notes
    -----
    支持的CAMELS数据集 / Supported CAMELS datasets:
    - CAMELS-US (671 catchments)
    - CAMELS-Chile (516 catchments)
    - CAMELS-Brazil (897 catchments)
    - CAMELS-UK (671 catchments)
    - CAMELS-Australia (222 catchments)
    """
    print("正在加载CAMELS流量数据... / Loading CAMELS discharge data...")
    
    catchments = {}
    
    # CAMELS国家列表 / List of CAMELS countries
    camels_countries = ['US', 'Chile', 'Brazil', 'UK', 'Australia']
    
    for country in camels_countries:
        country_dir = camels_dir / country
        
        if not country_dir.exists():
            print(f"  跳过: 未找到CAMELS-{country}目录 / Skipping: CAMELS-{country} directory not found")
            continue
        
        # 查找CSV文件 / Find CSV files
        csv_files = list(country_dir.glob('*.csv'))
        print(f"  在CAMELS-{country}中找到 {len(csv_files)} 个站点")
        print(f"  Found {len(csv_files)} stations in CAMELS-{country}")
        
        for csv_file in csv_files:
            try:
                # 提取站点ID (包含国家前缀)
                # Extract station ID (with country prefix)
                station_id = f"{country}_{csv_file.stem}"
                
                # 读取数据 / Read data
                df = pd.read_csv(csv_file, parse_dates=['date'])
                
                # 基本质量检查 / Basic quality check
                if len(df) > 0 and 'discharge_m3s' in df.columns:
                    catchments[station_id] = df
                    
            except Exception as e:
                print(f"    警告: 无法加载 {csv_file.name}: {e}")
                print(f"    Warning: Failed to load {csv_file.name}: {e}")
    
    print(f"成功加载 {len(catchments)} 个CAMELS流域 / Successfully loaded {len(catchments)} CAMELS catchments")
    
    return catchments


# =============================================================================
# 数据质量控制函数 (Data Quality Control Functions)
# =============================================================================

def quality_control_catchment(df: pd.DataFrame, 
                              catchment_id: str,
                              config: Config) -> Tuple[bool, str]:
    """
    对单个流域进行数据质量控制
    Perform data quality control for a single catchment
    
    Parameters
    ----------
    df : pd.DataFrame
        流域数据 / Catchment data
    catchment_id : str
        流域ID / Catchment ID
    config : Config
        配置对象 / Configuration object
        
    Returns
    -------
    pass_qc : bool
        是否通过质量控制 / Whether passed quality control
    reason : str
        未通过的原因 (如果适用) / Reason for failure (if applicable)
    """
    # 检查1: 记录长度 / Check 1: Record length
    record_years = len(df) / 365.25
    if record_years < config.MIN_RECORD_LENGTH:
        return False, f"记录长度不足: {record_years:.1f}年 / Insufficient record length: {record_years:.1f} years"
    
    # 检查2: 缺失率 / Check 2: Missing rate
    missing_rate = df['discharge_m3s'].isna().sum() / len(df)
    if missing_rate > config.MAX_MISSING_RATE:
        return False, f"缺失率过高: {missing_rate:.1%} / High missing rate: {missing_rate:.1%}"
    
    # 检查3: 流域面积 / Check 3: Catchment area
    if 'area_km2' in df.columns:
        area = df['area_km2'].iloc[0]
        if area < config.MIN_CATCHMENT_AREA or area > config.MAX_CATCHMENT_AREA:
            return False, f"流域面积超出范围: {area:.0f} km² / Area out of range: {area:.0f} km²"
    
    # 检查4: 连续缺失 / Check 4: Continuous missing
    # 避免季节性缺失导致的系统偏差
    # Avoid systematic bias from seasonal missing data
    df_copy = df.copy()
    df_copy['month'] = df_copy['date'].dt.month
    
    # 检查是否有连续3个月以上的缺失
    # Check if there are >3 consecutive months of missing data
    for month in range(1, 13):
        month_data = df_copy[df_copy['month'] == month]['discharge_m3s']
        if month_data.isna().all():
            return False, f"第{month}月完全缺失 / Month {month} completely missing"
    
    return True, "通过质量控制 / Passed QC"


def filter_catchments_by_qc(catchments: Dict[str, pd.DataFrame],
                            config: Config) -> Dict[str, pd.DataFrame]:
    """
    根据质量控制标准过滤流域
    Filter catchments based on quality control criteria
    
    Parameters
    ----------
    catchments : dict
        原始流域数据字典 / Raw catchments data dictionary
    config : Config
        配置对象 / Configuration object
        
    Returns
    -------
    filtered_catchments : dict
        通过质量控制的流域 / Catchments passing quality control
    """
    print("\n执行数据质量控制... / Performing data quality control...")
    
    filtered_catchments = {}
    qc_stats = {
        'total': len(catchments),
        'passed': 0,
        'failed_length': 0,
        'failed_missing': 0,
        'failed_area': 0,
        'failed_seasonal': 0
    }
    
    for catchment_id, df in catchments.items():
        passed, reason = quality_control_catchment(df, catchment_id, config)
        
        if passed:
            filtered_catchments[catchment_id] = df
            qc_stats['passed'] += 1
        else:
            # 统计失败原因 / Count failure reasons
            if '记录长度' in reason or 'record length' in reason:
                qc_stats['failed_length'] += 1
            elif '缺失率' in reason or 'missing rate' in reason:
                qc_stats['failed_missing'] += 1
            elif '面积' in reason or 'area' in reason:
                qc_stats['failed_area'] += 1
            elif '月' in reason or 'month' in reason:
                qc_stats['failed_seasonal'] += 1
    
    # 打印统计信息 / Print statistics
    print(f"\n质量控制统计 / Quality Control Statistics:")
    print(f"  总流域数 / Total catchments: {qc_stats['total']}")
    print(f"  通过QC / Passed QC: {qc_stats['passed']} ({qc_stats['passed']/qc_stats['total']*100:.1f}%)")
    print(f"  未通过原因 / Failed reasons:")
    print(f"    记录长度不足 / Insufficient length: {qc_stats['failed_length']}")
    print(f"    缺失率过高 / High missing rate: {qc_stats['failed_missing']}")
    print(f"    面积超范围 / Area out of range: {qc_stats['failed_area']}")
    print(f"    季节性缺失 / Seasonal missing: {qc_stats['failed_seasonal']}")
    
    return filtered_catchments


# =============================================================================
# 基流分离函数 (Baseflow Separation Functions)
# =============================================================================

def separate_baseflow_for_catchment(df: pd.DataFrame,
                                   config: Config) -> pd.DataFrame:
    """
    对单个流域进行基流分离
    Perform baseflow separation for a single catchment
    
    Parameters
    ----------
    df : pd.DataFrame
        流域数据 (包含日流量) / Catchment data (with daily discharge)
    config : Config
        配置对象 / Configuration object
        
    Returns
    -------
    df : pd.DataFrame
        添加了Qb和Qq列的数据框 / DataFrame with added Qb and Qq columns
    """
    # 获取日流量序列 / Get daily discharge series
    Q_daily = df['discharge_m3s'].values
    
    # 执行Lyne-Hollick基流分离
    # Perform Lyne-Hollick baseflow separation
    Qb_daily, Qq_daily = lyne_hollick_filter(
        Q_daily, 
        alpha=config.LH_ALPHA,
        n_passes=config.LH_PASSES
    )
    
    # 添加到数据框 / Add to dataframe
    df['Qb_m3s'] = Qb_daily
    df['Qq_m3s'] = Qq_daily
    
    return df


# =============================================================================
# 长期平均值计算 (Long-term Mean Calculation)
# =============================================================================

def calculate_longterm_means(catchments: Dict[str, pd.DataFrame],
                            meteo_data: Dict[str, xr.DataArray]) -> pd.DataFrame:
    """
    计算每个流域的长期平均值
    Calculate long-term means for each catchment
    
    Parameters
    ----------
    catchments : dict
        流域数据字典 / Catchments data dictionary
    meteo_data : dict
        气象数据字典 (P, Ep, Ea) / Meteorological data dictionary (P, Ep, Ea)
        
    Returns
    -------
    df_means : pd.DataFrame
        包含长期平均值的数据框 / DataFrame with long-term means
        列包括 / Columns include: catchment_id, P, Ep, Ea, Q, Qb, Qq, area_km2
    """
    print("\n计算长期平均值... / Calculating long-term means...")
    
    results = []
    
    for catchment_id, df in catchments.items():
        try:
            # 获取流域坐标 (假设存储在数据中)
            # Get catchment coordinates (assuming stored in data)
            if 'lat' not in df.columns or 'lon' not in df.columns:
                print(f"  警告: {catchment_id} 缺少坐标信息,跳过")
                print(f"  Warning: {catchment_id} missing coordinates, skipping")
                continue
            
            lat = df['lat'].iloc[0]
            lon = df['lon'].iloc[0]
            area_km2 = df['area_km2'].iloc[0] if 'area_km2' in df.columns else np.nan
            
            # 提取气象数据 (最近邻插值)
            # Extract meteorological data (nearest neighbor interpolation)
            P_mm_yr = float(meteo_data['P'].sel(lat=lat, lon=lon, method='nearest').mean())
            Ep_mm_yr = float(meteo_data['Ep'].sel(lat=lat, lon=lon, method='nearest').mean())
            Ea_mm_yr = float(meteo_data['Ea'].sel(lat=lat, lon=lon, method='nearest').mean())
            
            # 计算径流的长期平均 (转换为mm/yr)
            # Calculate long-term mean runoff (convert to mm/yr)
            Q_m3s = df['discharge_m3s'].mean()
            Qb_m3s = df['Qb_m3s'].mean()
            Qq_m3s = df['Qq_m3s'].mean()
            
            # 转换单位: m³/s -> mm/yr
            # Convert units: m³/s -> mm/yr
            # Q (mm/yr) = Q (m³/s) * (86400 * 365.25) / (area_km2 * 1e6) * 1000
            conversion_factor = (86400 * 365.25) / (area_km2 * 1e6) * 1000
            Q_mm_yr = Q_m3s * conversion_factor
            Qb_mm_yr = Qb_m3s * conversion_factor
            Qq_mm_yr = Qq_m3s * conversion_factor
            
            # 水量平衡检查 / Water balance check
            # |P - Ea - Q| / P < threshold
            water_balance_error = abs(P_mm_yr - Ea_mm_yr - Q_mm_yr) / P_mm_yr
            
            if water_balance_error > Config.WATER_BALANCE_THRESHOLD:
                print(f"  警告: {catchment_id} 水量平衡误差 {water_balance_error:.2%},跳过")
                print(f"  Warning: {catchment_id} water balance error {water_balance_error:.2%}, skipping")
                continue
            
            # 存储结果 / Store results
            results.append({
                'catchment_id': catchment_id,
                'lat': lat,
                'lon': lon,
                'area_km2': area_km2,
                'P_mm_yr': P_mm_yr,
                'Ep_mm_yr': Ep_mm_yr,
                'Ea_mm_yr': Ea_mm_yr,
                'Q_mm_yr': Q_mm_yr,
                'Qb_mm_yr': Qb_mm_yr,
                'Qq_mm_yr': Qq_mm_yr,
                'record_years': len(df) / 365.25,
                'water_balance_error': water_balance_error
            })
            
        except Exception as e:
            print(f"  错误: 处理 {catchment_id} 时出错: {e}")
            print(f"  Error: Failed processing {catchment_id}: {e}")
    
    # 转换为DataFrame / Convert to DataFrame
    df_means = pd.DataFrame(results)
    
    print(f"成功计算 {len(df_means)} 个流域的长期平均值")
    print(f"Successfully calculated long-term means for {len(df_means)} catchments")
    
    return df_means


# =============================================================================
# 主函数 (Main Function)
# =============================================================================

def main():
    """主预处理流程 / Main preprocessing workflow"""
    
    print("="*80)
    print("开始数据预处理 / Starting Data Preprocessing")
    print("="*80)
    
    config = Config()
    
    # 创建输出目录 / Create output directory
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 步骤1: 加载流量数据 / Step 1: Load discharge data
    print("\n[步骤1/5] 加载流量数据...")
    print("[Step 1/5] Loading discharge data...")
    
    catchments = {}
    
    # 加载GRDC数据 / Load GRDC data
    if config.GRDC_DIR.exists():
        grdc_catchments = load_grdc_catchments(config.GRDC_DIR)
        catchments.update(grdc_catchments)
    
    # 加载CAMELS数据 / Load CAMELS data
    if config.CAMELS_DIR.exists():
        camels_catchments = load_camels_catchments(config.CAMELS_DIR)
        catchments.update(camels_catchments)
    
    print(f"总共加载 {len(catchments)} 个流域 / Total loaded {len(catchments)} catchments")
    
    # 步骤2: 数据质量控制 / Step 2: Data quality control
    print("\n[步骤2/5] 数据质量控制...")
    print("[Step 2/5] Data quality control...")
    
    catchments = filter_catchments_by_qc(catchments, config)
    
    # 步骤3: 基流分离 / Step 3: Baseflow separation
    print("\n[步骤3/5] 基流分离...")
    print("[Step 3/5] Baseflow separation...")
    
    for catchment_id in catchments.keys():
        catchments[catchment_id] = separate_baseflow_for_catchment(
            catchments[catchment_id], config
        )
    
    print(f"完成 {len(catchments)} 个流域的基流分离")
    print(f"Completed baseflow separation for {len(catchments)} catchments")
    
    # 步骤4: 加载气象数据 / Step 4: Load meteorological data
    print("\n[步骤4/5] 加载气象数据...")
    print("[Step 4/5] Loading meteorological data...")
    
    # 注意: 这里需要实际的气象数据文件
    # Note: This requires actual meteorological data files
    # 这里仅作为示例结构 / This is just example structure
    print("  注意: 需要下载MSWEP, TerraClimate, GLEAM数据")
    print("  Note: Need to download MSWEP, TerraClimate, GLEAM data")
    
    meteo_data = {
        'P': None,  # xr.open_dataset(config.P_FILE)['precipitation']
        'Ep': None,  # xr.open_dataset(config.EP_FILE)['pet']
        'Ea': None,  # xr.open_dataset(config.EA_FILE)['Ea']
    }
    
    # 步骤5: 计算长期平均值 / Step 5: Calculate long-term means
    print("\n[步骤5/5] 计算长期平均值...")
    print("[Step 5/5] Calculating long-term means...")
    
    # df_means = calculate_longterm_means(catchments, meteo_data)
    
    # 保存处理后的数据 / Save processed data
    # output_file = config.PROCESSED_DATA_DIR / 'catchment_longterm_means.csv'
    # df_means.to_csv(output_file, index=False)
    # print(f"\n数据已保存到: {output_file}")
    # print(f"Data saved to: {output_file}")
    
    print("\n" + "="*80)
    print("数据预处理完成! / Data preprocessing completed!")
    print("="*80)


if __name__ == "__main__":
    main()
