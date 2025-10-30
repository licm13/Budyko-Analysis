# Budyko框架完整分析示例
# 基于He et al. (2023)的三维Budyko框架

## 1. 环境准备

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 添加项目路径
sys.path.append('../src')

from budyko.analyzer import BudykoAnalyzer, SeasonalBudykoAnalyzer
from pet.calculator import PETCalculator, PETUncertaintyAnalyzer
from data_processing.processor import BasinDataProcessor, RemoteSensingDataProcessor

# 设置绘图风格
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False

%matplotlib inline
```

## 2. 数据加载

### 2.1 加载流域径流数据

```python
# 初始化数据处理器
processor = BasinDataProcessor()

# 加载径流观测数据
# 假设数据格式: date, basin_id, runoff (mm/day)
runoff_data = processor.load_runoff_observations(
    file_path='../data/raw/china_basins_runoff.csv',
    file_format='csv',
    date_column='date',
    basin_column='basin_id',
    runoff_column='runoff'
)

print(f"加载了 {runoff_data['basin_id'].nunique()} 个流域的径流数据")
print(f"时间范围: {runoff_data['date'].min()} 至 {runoff_data['date'].max()}")
```

### 2.2 加载CMFD气象数据

```python
# 加载CMFD 0.1°格点数据
cmfd_data = processor.load_cmfd_gridded_data(
    cmfd_dir='../data/raw/CMFD',
    variables=['prec', 'temp', 'wind', 'pres', 'shum', 'srad'],
    start_year=1960,
    end_year=2020
)

print("CMFD数据变量:", list(cmfd_data.data_vars))
```

### 2.3 提取流域平均气象数据

```python
import geopandas as gpd

# 加载流域矢量边界
basins_shp = gpd.read_file('../data/raw/china_basins.shp')

# 提取流域平均
basin_met_data = processor.extract_basin_average(
    gridded_data=cmfd_data,
    basin_geometries=basins_shp,
    method='area_weighted'
)
```

## 3. 计算PET

### 3.1 使用多种PET方法

```python
pet_calc = PETCalculator(elevation=500)  # 平均海拔500m

# 准备输入数据
T = basin_met_data['temp'].values  # 气温
RH = basin_met_data['shum'].values * 100  # 相对湿度
u2 = basin_met_data['wind'].values  # 风速
Rn = basin_met_data['srad'].values * 0.0864  # 辐射转换为MJ/m²/day

# 计算多种PET
pet_results = {}

# 1. Penman-Monteith FAO-56
pet_results['PM_FAO56'] = pet_calc.penman_monteith_fao56(
    T=T, RH=RH, u2=u2, Rn=Rn
)

# 2. Priestley-Taylor
pet_results['PT'] = pet_calc.priestley_taylor(
    T=T, Rn=Rn
)

# 3. Hargreaves (需要T_max, T_min)
# pet_results['Hargreaves'] = pet_calc.hargreaves(...)
```

### 3.2 创新：考虑LAI和CO2的PET

```python
# 加载MODIS LAI数据
rs_processor = RemoteSensingDataProcessor()
lai_data = rs_processor.load_modis_lai(
    modis_dir='../data/raw/MODIS_LAI',
    product='MOD15A2H'
)

# 提取流域LAI
# ... (类似气象数据提取)

# CO2浓度数据（可从全球数据集获取）
co2 = 400 * np.ones_like(T)  # 简化为常数

# 计算考虑LAI和CO2的PET
pet_results['PM_LAI_CO2'] = pet_calc.penman_monteith_with_lai_co2(
    T=T, RH=RH, u2=u2, Rn=Rn,
    LAI=lai_values,
    CO2=co2
)

# 绘制不同PET方法对比
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('不同PET方法对比', fontsize=16)

for idx, (method, pet) in enumerate(pet_results.items()):
    ax = axes.flatten()[idx]
    ax.hist(pet[np.isfinite(pet)], bins=50, edgecolor='black', alpha=0.7)
    ax.set_title(method)
    ax.set_xlabel('PET (mm/day)')
    ax.set_ylabel('频数')
    ax.axvline(np.nanmean(pet), color='red', linestyle='--', label=f'均值: {np.nanmean(pet):.2f}')
    ax.legend()

plt.tight_layout()
plt.savefig('../data/output/pet_methods_comparison.png', dpi=300)
```

### 3.3 PET不确定性分析

```python
uncertainty_analyzer = PETUncertaintyAnalyzer()

for method, pet in pet_results.items():
    uncertainty_analyzer.add_pet_method(method, pet)

# 计算方法间差异
spread = uncertainty_analyzer.calculate_spread()

print("PET方法不确定性:")
print(f"平均值: {np.nanmean(spread['mean']):.2f} mm/day")
print(f"标准差: {np.nanmean(spread['std']):.2f} mm/day")
print(f"变异系数: {np.nanmean(spread['cv']):.2f}%")
```

## 4. 传统Budyko分析

### 4.1 年尺度分析

```python
# 聚合到年尺度
annual_data = processor.aggregate_to_timescale(
    data=basin_met_data,
    time_scale='annual',
    variables=['Q', 'P', 'T']
)

# 合并径流数据
# ... 

# 初始化Budyko分析器
budyko = BudykoAnalyzer(time_scale='annual')

# 加载数据（使用PM_LAI_CO2作为PET）
budyko.load_basin_data(
    P=annual_data['P'].values,
    Q=annual_data['Q'].values,
    PET=pet_results['PM_LAI_CO2'][:len(annual_data)],
    basin_ids=annual_data['basin_id'].values
)

print("\n基本Budyko指数:")
print(budyko.data[['basin_id', 'P', 'Q', 'ET', 'PET', 'DI', 'EI']].head())
```

### 4.2 估计ω参数

```python
# 估计景观参数ω
omega_optimal = budyko.estimate_omega(
    budyko.data['DI'].values,
    budyko.data['EI'].values
)

print(f"\n最优ω参数: {omega_optimal:.2f}")

# 计算偏离
deviation = budyko.calculate_budyko_deviation(omega=omega_optimal)
budyko.data['deviation'] = deviation
```

### 4.3 绘制Budyko空间

```python
fig, ax = plt.subplots(figsize=(10, 8))

# 数据点
scatter = ax.scatter(
    budyko.data['DI'],
    budyko.data['EI'],
    c=budyko.data['deviation'],
    cmap='RdBu_r',
    s=50,
    alpha=0.6,
    edgecolor='black',
    linewidth=0.5
)

# Budyko曲线
DI_range = np.linspace(0, 5, 100)
EI_budyko = budyko.budyko_curve(DI_range, omega=omega_optimal)

ax.plot(DI_range, EI_budyko, 'r-', linewidth=2, label=f'Budyko曲线 (ω={omega_optimal:.2f})')

# 能量和水分限制边界
ax.plot([0, 5], [0, 1], 'k--', alpha=0.5, label='能量限制')
ax.plot([1, 5], [1, 1], 'k--', alpha=0.5, label='水分限制')

# 设置
ax.set_xlabel('干旱指数 (DI = PET/P)', fontsize=12)
ax.set_ylabel('蒸发指数 (EI = ET/P)', fontsize=12)
ax.set_title('中国流域Budyko空间分布', fontsize=14)
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.2)
ax.legend()
ax.grid(True, alpha=0.3)

# 颜色条
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Budyko偏离', fontsize=12)

plt.tight_layout()
plt.savefig('../data/output/budyko_space_annual.png', dpi=300)
```

## 5. 三维Budyko分析（论文核心创新）

### 5.1 加载GRACE TWS数据

```python
# 加载GRACE TWS
grace_tws = rs_processor.load_grace_tws(
    grace_dir='../data/raw/GRACE',
    solution='CSR',
    release='RL06'
)

# 提取流域TWS
# ... (类似气象数据提取)

# 加载到Budyko分析器
budyko.data['TWS'] = tws_values

# 计算三维指数
budyko.calculate_3d_indices()

print("\n三维Budyko指数:")
print(budyko.data[['SI', 'SCI', 'EI_extended', 'DI_extended']].head())
```

### 5.2 DI/EI与SI/SCI关系

```python
# 计算相关关系
relationships = budyko.calculate_si_sci_relationships()

print("\nDI/EI与SI/SCI的关系:")
for key, value in relationships.items():
    print(f"{key}: {value:.3f}")
```

### 5.3 绘制三维Budyko空间

```python
fig = plt.figure(figsize=(15, 5))

# 子图1: DI vs SI
ax1 = fig.add_subplot(131)
scatter1 = ax1.scatter(
    budyko.data['DI'],
    budyko.data['SI'],
    c=budyko.data['P'],
    cmap='Blues',
    s=30,
    alpha=0.6
)
ax1.set_xlabel('干旱指数 (DI)')
ax1.set_ylabel('储量指数 (SI = TWS/P)')
ax1.set_title('DI vs SI')
plt.colorbar(scatter1, ax=ax1, label='降水 (mm)')

# 子图2: DI vs SCI
ax2 = fig.add_subplot(132)
scatter2 = ax2.scatter(
    budyko.data['DI'],
    budyko.data['SCI'],
    c=budyko.data['EI'],
    cmap='Greens',
    s=30,
    alpha=0.6
)
ax2.set_xlabel('干旱指数 (DI)')
ax2.set_ylabel('储量变化指数 (SCI = ΔS/P)')
ax2.set_title('DI vs SCI')
ax2.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='蒸发指数 (EI)')

# 子图3: 传统vs扩展Budyko
ax3 = fig.add_subplot(133)
ax3.scatter(
    budyko.data['DI'],
    budyko.data['EI'],
    label='传统 (无TWS)',
    s=30,
    alpha=0.4,
    color='blue'
)
ax3.scatter(
    budyko.data['DI_extended'],
    budyko.data['EI_extended'],
    label='扩展 (含TWS)',
    s=30,
    alpha=0.4,
    color='red'
)

# Budyko曲线
DI_range = np.linspace(0, 5, 100)
ax3.plot(DI_range, budyko.budyko_curve(DI_range, omega_optimal), 
         'k-', linewidth=2, label=f'Budyko曲线')

ax3.set_xlabel('干旱指数 (DI)')
ax3.set_ylabel('蒸发指数 (EI)')
ax3.set_title('传统 vs 扩展Budyko')
ax3.set_xlim(0, 5)
ax3.set_ylim(0, 1.2)
ax3.legend()

plt.tight_layout()
plt.savefig('../data/output/3d_budyko_framework.png', dpi=300)
```

## 6. 季节性分析

### 6.1 湿季vs干季

```python
seasonal_budyko = SeasonalBudykoAnalyzer()

# 聚合到季节
seasonal_data = processor.aggregate_to_timescale(
    data=basin_met_data,
    time_scale='seasonal'
)

# 加载数据
seasonal_budyko.load_basin_data(
    P=seasonal_data['P'].values,
    Q=seasonal_data['Q'].values,
    PET=seasonal_pet,  # 季节PET
    basin_ids=seasonal_data['basin_id'].values,
    TWS=seasonal_tws  # 季节TWS
)

# 划分季节（湿季6-10月，干季11-5月）
seasonal_budyko.split_seasons(
    dates=seasonal_data['date'],
    wet_months=[6, 7, 8, 9, 10],
    dry_months=[11, 12, 1, 2, 3, 4, 5]
)

# 计算三维指数
seasonal_budyko.calculate_3d_indices()

# 按季节分析
season_results = seasonal_budyko.analyze_by_season()
```

### 6.2 季节对比图

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

seasons = ['wet', 'dry']
colors = ['blue', 'red']

for idx, season in enumerate(seasons):
    ax = axes[idx]
    data_season = season_results[season]
    
    # 数据点
    scatter = ax.scatter(
        data_season['DI'],
        data_season['EI'],
        c=data_season['SCI'],
        cmap='RdBu_r',
        s=50,
        alpha=0.6,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Budyko曲线
    omega_season = data_season['omega'].iloc[0]
    DI_range = np.linspace(0, 5, 100)
    ax.plot(DI_range, budyko.budyko_curve(DI_range, omega_season),
            color=colors[idx], linewidth=2, 
            label=f'Budyko (ω={omega_season:.2f})')
    
    # 边界
    ax.plot([0, 5], [0, 1], 'k--', alpha=0.3)
    ax.plot([1, 5], [1, 1], 'k--', alpha=0.3)
    
    ax.set_xlabel('干旱指数 (DI)', fontsize=12)
    ax.set_ylabel('蒸发指数 (EI)', fontsize=12)
    ax.set_title(f'{"湿季" if season == "wet" else "干季"} Budyko空间', fontsize=14)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='SCI')

plt.tight_layout()
plt.savefig('../data/output/seasonal_budyko_comparison.png', dpi=300)
```

## 7. 偏离归因分析

### 7.1 分类流域

```python
# 水分/能量限制分类
budyko.data['limit_type'] = budyko.classify_water_energy_limit()

# 统计
limit_stats = budyko.data.groupby('limit_type').agg({
    'basin_id': 'count',
    'DI': 'mean',
    'EI': 'mean',
    'deviation': 'mean'
})

print("\n流域分类统计:")
print(limit_stats)
```

### 7.2 偏离影响因子分析

```python
# 加载可能的影响因子
# - 植被类型/NDVI
# - 土地利用变化
# - 水库/大坝
# - 灌溉
# - 积雪

# 简化示例：NDVI影响
from scipy import stats

# 相关分析
corr, pval = stats.pearsonr(
    budyko.data['NDVI'].dropna(),
    budyko.data['deviation'].dropna()
)

print(f"\nNDVI与Budyko偏离的相关性:")
print(f"相关系数: {corr:.3f}")
print(f"p值: {pval:.4f}")

# 绘图
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(budyko.data['NDVI'], budyko.data['deviation'], 
           alpha=0.5, s=30)
ax.set_xlabel('NDVI', fontsize=12)
ax.set_ylabel('Budyko偏离', fontsize=12)
ax.set_title('植被与Budyko偏离的关系', fontsize=14)
ax.axhline(0, color='red', linestyle='--', alpha=0.3)

# 拟合线
z = np.polyfit(budyko.data['NDVI'].dropna(), 
               budyko.data['deviation'].dropna(), 1)
p = np.poly1d(z)
ax.plot(budyko.data['NDVI'].dropna(), 
        p(budyko.data['NDVI'].dropna()), 
        "r-", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.savefig('../data/output/ndvi_deviation_relationship.png', dpi=300)
```

## 8. 汇总与输出

### 8.1 生成汇总报告

```python
summary = budyko.get_summary_statistics()

print("\n=== Budyko分析汇总报告 ===")
print(f"\n流域数量: {budyko.data['basin_id'].nunique()}")
print(f"时间范围: {annual_data['year'].min()} - {annual_data['year'].max()}")
print(f"\n最优ω参数: {omega_optimal:.2f}")
print(f"\n基本统计:")
print(summary)

# 保存
summary.to_csv('../data/output/budyko_summary_statistics.csv')
budyko.data.to_csv('../data/output/budyko_analysis_results.csv', index=False)
```

### 8.2 创新点总结

```python
print("\n=== 本研究创新点 ===")
print("\n1. 考虑LAI和CO2的PET方法")
print(f"   - 传统PM PET均值: {np.nanmean(pet_results['PM_FAO56']):.2f} mm/day")
print(f"   - 改进PM PET均值: {np.nanmean(pet_results['PM_LAI_CO2']):.2f} mm/day")
print(f"   - 差异: {np.nanmean(pet_results['PM_LAI_CO2'] - pet_results['PM_FAO56']):.2f} mm/day")

print("\n2. 三维Budyko框架整合TWS")
print(f"   - DI-SI相关系数: {relationships['corr_DI_SI']:.3f}")
print(f"   - DI-SCI相关系数: {relationships['corr_DI_SCI']:.3f}")

print("\n3. 季节性差异")
wet_omega = season_results['wet']['omega'].iloc[0]
dry_omega = season_results['dry']['omega'].iloc[0]
print(f"   - 湿季ω参数: {wet_omega:.2f}")
print(f"   - 干季ω参数: {dry_omega:.2f}")
```

## 结论

本示例展示了：
1. 传统二维Budyko框架的完整分析流程
2. He et al. (2023)提出的三维Budyko框架（整合TWS）
3. **研究创新**：考虑LAI和CO2的改进PET方法
4. 季节性对比分析
5. Budyko偏离的归因分析

**下一步工作**：
- 研究积雪对Budyko关系的影响
- 多尺度（流域大小）分析
- 未来情景预测（CMIP6）
- 人类活动影响量化