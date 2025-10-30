# Budyko Framework 使用指南

## 项目概述

本项目实现了 He et al. (2023) 提出的三维Budyko框架，并结合您的研究思路进行了扩展创新。

### 核心功能

1. **传统Budyko分析**
   - 计算干旱指数 (DI) 和蒸发指数 (EI)
   - 估计景观参数 ω
   - 偏离分析

2. **三维Budyko框架**（He et al. 2023创新）
   - 整合陆地水储量 (TWS)
   - 储量指数 (SI) 和储量变化指数 (SCI)
   - DI/EI与SI/SCI关系分析

3. **改进PET方法**（您的研究创新）
   - 考虑LAI影响的PET
   - 考虑CO2浓度变化的PET
   - 多种PET方法对比

4. **综合分析**
   - 季节性分析（湿季/干季）
   - 积雪影响分析
   - 偏离归因分析

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv budyko_env
source budyko_env/bin/activate  # Linux/Mac
# 或
budyko_env\Scripts\activate  # Windows

# 安装依赖包
pip install -r requirements.txt
```

### 2. 运行示例

```bash
# 运行快速开始示例（使用模拟数据）
python quick_start.py
```

这将：
- 生成模拟流域数据
- 计算多种PET方法
- 进行传统和三维Budyko分析
- 生成可视化图表
- 输出分析报告

### 3. 查看结果

分析结果保存在 `data/output/` 目录：
- `budyko_space_traditional.png`: 传统Budyko空间图
- `3d_budyko_space.png`: 三维Budyko空间图
- `di_si_relationships.png`: DI/EI与SI/SCI关系图
- `deviation_distribution.png`: 偏离分布图
- `budyko_analysis_results.csv`: 完整数据
- `analysis_report.txt`: 分析报告

## 使用您自己的数据

### 数据准备

将您的数据放在 `data/raw/` 目录：

```
data/raw/
├── china_basins.shp         # 流域矢量边界
├── china_basins_runoff.csv  # 径流观测数据
├── CMFD/                     # CMFD气象数据
│   ├── prec_*.nc
│   ├── temp_*.nc
│   └── ...
├── GRACE/                    # GRACE TWS数据（可选）
│   └── CSR_RL06_*.nc
└── MODIS_LAI/               # MODIS LAI数据（可选）
    └── MOD15A2H_*.hdf
```

### 数据格式要求

#### 1. 径流数据 CSV格式
```csv
date,basin_id,runoff
1960-01-01,1,50.2
1960-01-02,1,48.5
...
```

#### 2. CMFD气象数据
- NetCDF格式
- 变量：prec (降水), temp (气温), wind (风速), pres (气压), shum (比湿), srad (辐射)
- 分辨率：0.1°
- 时间：日尺度

#### 3. 流域边界 Shapefile
- 必须包含字段：basin_id, area
- 坐标系：WGS84 (EPSG:4326)

### 修改配置

编辑 `config/settings.py` 调整参数：

```python
# 时间范围
START_YEAR = 1960
END_YEAR = 2020

# 季节划分
SEASONS = {
    'wet': [6, 7, 8, 9, 10],
    'dry': [11, 12, 1, 2, 3, 4, 5]
}

# 流域筛选
MIN_DATA_YEARS = 10
MIN_BASIN_AREA = 100
```

### 运行完整分析

```python
import sys
sys.path.append('src')

from budyko.analyzer import BudykoAnalyzer
from pet.calculator import PETCalculator
from data_processing.processor import BasinDataProcessor
from visualization.plotter import BudykoVisualizer

# 1. 加载数据
processor = BasinDataProcessor()
runoff_data = processor.load_runoff_observations(
    'data/raw/china_basins_runoff.csv'
)

# 2. 加载CMFD数据
cmfd_data = processor.load_cmfd_gridded_data(
    cmfd_dir='data/raw/CMFD',
    start_year=1960,
    end_year=2020
)

# 3. 提取流域平均
import geopandas as gpd
basins = gpd.read_file('data/raw/china_basins.shp')
basin_met = processor.extract_basin_average(
    cmfd_data, basins
)

# 4. 计算PET
pet_calc = PETCalculator()
PET = pet_calc.penman_monteith_with_lai_co2(
    T=basin_met['temp'],
    RH=basin_met['rh'],
    u2=basin_met['wind'],
    Rn=basin_met['rad'],
    LAI=basin_met['lai'],
    CO2=basin_met['co2']
)

# 5. Budyko分析
budyko = BudykoAnalyzer()
budyko.load_basin_data(
    P=basin_met['prec'],
    Q=runoff_data['runoff'],
    PET=PET,
    basin_ids=basin_met['basin_id']
)

# 6. 三维分析（如果有GRACE数据）
budyko.data['TWS'] = tws_data
budyko.calculate_3d_indices()

# 7. 可视化
viz = BudykoVisualizer()
viz.plot_budyko_space(
    budyko.data['DI'],
    budyko.data['EI'],
    omega=budyko.results['omega'],
    save_path='data/output/my_budyko_space.png'
)
```

## 研究框架对应

根据您的研究思路，本框架提供：

### 1. 基础验证
```python
# 检验流域是否遵循Budyko曲线
budyko.calculate_budyko_deviation()
```

### 2. 雪的影响
```python
from data_processing.processor import SnowDataProcessor
snow_processor = SnowDataProcessor()
snow_basins = snow_processor.identify_snow_dominated_basins(data)
```

### 3. 偏离原因分析
```python
# 分类流域
limit_type = budyko.classify_water_energy_limit()

# 归因分析
# 可以加入土地利用、水库、灌溉等因子
```

### 4. 创新PET方法
```python
# 使用考虑LAI和CO2的PET
pet_calc.penman_monteith_with_lai_co2(
    T, RH, u2, Rn, LAI, CO2
)
```

### 5. 其他改进
- 多尺度分析
- 季节性分析
- 未来情景（CMIP6）

## 进阶使用

### Jupyter Notebook

查看 `notebooks/01_complete_budyko_analysis.md` 获取完整的分析示例。

可以转换为Jupyter notebook:
```bash
# 安装jupytext
pip install jupytext

# 转换
jupytext --to notebook notebooks/01_complete_budyko_analysis.md
```

### 批量处理

```python
# 并行处理多个流域
from joblib import Parallel, delayed

def process_basin(basin_id):
    # 你的分析代码
    pass

results = Parallel(n_jobs=-1)(
    delayed(process_basin)(bid) for bid in basin_ids
)
```

### 自定义PET方法

```python
from pet.calculator import PETCalculator

class MyPETCalculator(PETCalculator):
    def my_custom_pet(self, **kwargs):
        # 您的PET计算方法
        pass
```

## 常见问题

### Q1: 如何处理缺失数据？
A: 在 `data_processing/processor.py` 中有质量控制函数，可以设置缺失值处理策略。

### Q2: 如何添加新的影响因子？
A: 在 `BudykoAnalyzer` 的 `load_basin_data` 中添加新列即可。

### Q3: 如何导出结果？
A: 使用 `budyko.data.to_csv()` 或 `budyko.data.to_excel()`。

### Q4: 支持哪些时间尺度？
A: 支持年、季节、月尺度，在 `time_scale` 参数中设置。

### Q5: 如何引用本框架？
A: 引用论文 He et al. (2023) 并注明使用了本框架。

## 贡献与支持

如有问题或建议：
1. 检查文档和示例代码
2. 查看配置文件 `config/settings.py`
3. 参考论文 He et al. (2023)

## 下一步计划

根据您的研究目标，建议：

1. **短期**：
   - 验证框架在您的数据上的效果
   - 对比传统PET和改进PET的差异
   - 分析季节性特征

2. **中期**：
   - 加入积雪影响分析
   - 多尺度分析（流域大小）
   - 人类活动影响量化

3. **长期**：
   - CMIP6未来情景分析
   - 发表研究成果
   - 扩展到其他区域

## 参考文献

- He et al. (2023). Three-dimensional Budyko framework incorporating terrestrial water storage. *Science of the Total Environment*, 904, 166380.
- Budyko (1974). Climate and Life.
- Fu (1981). On the calculation of the evaporation from land surface.

## 许可证

本项目供学术研究使用。商业使用请联系作者。

---

祝研究顺利！如有任何问题，欢迎随时询问。