"""
Configuration File for Budyko Framework
Budyko框架配置文件
"""

import os
from pathlib import Path

# ==================== 路径配置 ====================
# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = DATA_DIR / 'output'

# 数据子目录
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# 确保目录存在
for dir_path in [OUTPUT_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== 数据文件路径 ====================
# 流域数据
BASIN_SHAPEFILE = RAW_DATA_DIR / 'china_basins.shp'
RUNOFF_DATA_FILE = RAW_DATA_DIR / 'china_basins_runoff.csv'

# CMFD气象数据
CMFD_DIR = RAW_DATA_DIR / 'CMFD'

# GRACE数据
GRACE_DIR = RAW_DATA_DIR / 'GRACE'

# MODIS数据
MODIS_LAI_DIR = RAW_DATA_DIR / 'MODIS_LAI'
MODIS_NDVI_DIR = RAW_DATA_DIR / 'MODIS_NDVI'

# 积雪数据
SNOW_DATA_DIR = RAW_DATA_DIR / 'Snow'

# ==================== 分析参数 ====================
# 时间范围
START_YEAR = 1960
END_YEAR = 2020

# GRACE时间范围
GRACE_START = '2002-04'
GRACE_END = '2017-06'

# 时间尺度
TIME_SCALES = ['annual', 'seasonal', 'monthly']

# 季节划分
SEASONS = {
    'wet': [6, 7, 8, 9, 10],      # 湿季: 6-10月
    'dry': [11, 12, 1, 2, 3, 4, 5]  # 干季: 11-5月
}

# ==================== Budyko参数 ====================
# 默认ω参数
DEFAULT_OMEGA = 2.6

# ω参数范围
OMEGA_MIN = 0.5
OMEGA_MAX = 10.0

# 优化方法
OMEGA_OPTIMIZATION_METHOD = 'L-BFGS-B'

# ==================== PET计算参数 ====================
# PET方法列表
PET_METHODS = [
    'PM_FAO56',           # Penman-Monteith FAO-56
    'PM_LAI_CO2',         # 改进PM（考虑LAI和CO2）
    'Priestley_Taylor',   # Priestley-Taylor
    'Hargreaves',         # Hargreaves
]

# 默认PET方法
DEFAULT_PET_METHOD = 'PM_LAI_CO2'

# PM方法参数
PM_PARAMS = {
    'rs_min': 70,      # 最小气孔阻抗 [s/m]
    'beta_co2': 0.15,  # CO2敏感性参数
    'co2_ref': 400,    # 参考CO2浓度 [ppm]
}

# Priestley-Taylor系数
PT_ALPHA = 1.26

# ==================== 流域筛选标准 ====================
# 最小数据年限
MIN_DATA_YEARS = 10

# 最小流域面积 [km²]
MIN_BASIN_AREA = 100

# 是否排除受调控流域
EXCLUDE_REGULATED = True

# 数据质量控制
QC_PARAMS = {
    'outlier_threshold': 3,  # IQR倍数
    'max_missing_rate': 0.2,  # 最大缺失率
}

# ==================== 可视化参数 ====================
# 默认图像大小
DEFAULT_FIGSIZE = (10, 8)

# 默认DPI
DEFAULT_DPI = 300

# 配色方案
COLORMAPS = {
    'deviation': 'RdBu_r',
    'precipitation': 'Blues',
    'temperature': 'RdYlBu_r',
    'et': 'Greens',
    'tws': 'BrBG',
}

# 绘图风格
PLOT_STYLE = 'seaborn-v0_8-whitegrid'

# ==================== 输出格式 ====================
# 支持的输出格式
OUTPUT_FORMATS = ['csv', 'xlsx', 'netcdf', 'geojson']

# 默认输出格式
DEFAULT_OUTPUT_FORMAT = 'csv'

# 数值精度
DECIMAL_PRECISION = 4

# ==================== 计算参数 ====================
# 并行计算
USE_PARALLEL = True
N_JOBS = -1  # -1表示使用所有CPU核心

# 缓存
USE_CACHE = True
CACHE_DIR = PROJECT_ROOT / '.cache'

# ==================== 研究创新参数 ====================
# 是否使用考虑LAI和CO2的PET
USE_IMPROVED_PET = True

# 是否进行三维Budyko分析
USE_3D_BUDYKO = True

# 是否分析积雪影响
ANALYZE_SNOW_IMPACT = True

# 是否进行季节性分析
ANALYZE_SEASONALITY = True

# ==================== 日志配置 ====================
# 日志级别
LOG_LEVEL = 'INFO'

# 日志文件
LOG_FILE = OUTPUT_DIR / 'budyko_analysis.log'

# ==================== 常数 ====================
# 物理常数
LATENT_HEAT_VAPORIZATION = 2.45  # MJ/kg
AIR_DENSITY = 1.2  # kg/m³
SPECIFIC_HEAT_AIR = 1.01  # kJ/kg/K
STEFAN_BOLTZMANN = 4.903e-9  # MJ/K⁴/m²/day

# 转换因子
MM_TO_M = 0.001
M_TO_MM = 1000
DAY_TO_YEAR = 365.25

# ==================== 打印配置信息 ====================
def print_config():
    """打印当前配置"""
    print("=" * 60)
    print("Budyko Framework 配置信息")
    print("=" * 60)
    print(f"\n项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"\n时间范围: {START_YEAR} - {END_YEAR}")
    print(f"默认ω参数: {DEFAULT_OMEGA}")
    print(f"默认PET方法: {DEFAULT_PET_METHOD}")
    print(f"使用改进PET: {USE_IMPROVED_PET}")
    print(f"使用三维Budyko: {USE_3D_BUDYKO}")
    print(f"分析季节性: {ANALYZE_SEASONALITY}")
    print(f"分析积雪影响: {ANALYZE_SNOW_IMPACT}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()