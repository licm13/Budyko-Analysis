# Budyko-Analysis: 中国流域水文能量平衡分析框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

一个全面的Budyko框架分析工具，专门针对中国6000+小流域的水文能量平衡研究。本框架整合了传统Budyko理论、Ibrahim偏差分析方法和Jaramillo轨迹分析方法，并创新性地引入了**考虑LAI和CO2的PET估算方法**。

---

## 核心研究框架 (Framework-WBET)

本项目基于以下研究思路：

### 1. 基础验证：流域是否遵循Budyko曲线？
- **数据基础**：中国6000+小流域观测径流数据（尽量小，无人类活动影响）
- **气象数据**：0.1°CMFD数据（1960-2020）
- **核心原理**：
  - 干旱指数：`IA = PET/P`
  - 蒸发指数：`IE = EA/P`
  - 实际蒸发：`EA = P - Q`（径流数据Q是关键）

### 2. 雪的影响有多大？
- 识别积雪主导流域
- 量化积雪对水量平衡的贡献
- 分析融雪期与非融雪期的Budyko关系差异

### 3. 流域偏离Budyko曲线的原因？
- **三大研究方向**：
  1. **高精度验证与图谱绘制**（径流是"尺子"）
     - 确定理论位置：利用PET和P计算理论IA和IE
     - 确定实际位置：**通过径流Q计算实际EA = P - Q**
     - 计算偏离：`ε = IE,obs - IE,theory`
     - 绘制中国流域响应稳定性图谱

  2. **定量归因分析**（径流是"病人"）
     - **径流数据揭示偏离"症状"**
     - 引入驱动因子：土地利用、水库、灌溉等
     - 机器学习建立"病因-症状"联系
     - 示例：华北平原径流显示严重偏离 → 归因于灌溉

  3. **方法论深化**（径流是"参照物"）
     - 检验储量变化：`EA' = P - Q - ΔS`（GRACE数据）
     - PET公式不确定性：多种PET方法对比

### 4. **核心创新：考虑LAI和CO2的PET方法**
- **这是前人未做过的工作**
- 传统PET忽略了植被动态和CO2施肥效应
- 本框架实现：
  - Penman-Monteith with LAI adjustment
  - Stomatal conductance response to CO2
  - 结合MODIS LAI和CO2浓度数据

### 5. 其他创新方面
- 多尺度分析（流域大小效应）
- 季节性分析（湿季/干季）
- 未来情景预测（CMIP6/TRENDY）

---

## 径流数据（Q）的核心地位

### 为什么径流数据是基石？

**没有径流数据Q，我们无法进行任何Budyko分析！**

```
水量平衡方程：P - Q = EA + ΔS
长时间尺度：EA ≈ P - Q （ΔS ≈ 0）

干旱指数：IA = PET / P     （X轴，由气象数据决定）
蒸发指数：IE = EA / P      （Y轴，由径流数据Q决定！）
```

### 径流数据的三重角色

1. **"尺子"角色**：衡量流域真实的水分消耗
   - 理论IE = f(IA, ω)
   - 实际IE = (P - Q) / P  ← **完全依赖径流观测Q**
   - 偏差 = 实际IE - 理论IE

2. **"病人"角色**：揭示流域"健康状态"
   - 径流异常 → 水量平衡偏离 → "诊断"病因
   - 示例：Q减少 → IE增加 → 可能是灌溉取水

3. **"参照物"角色**：检验理论和方法
   - 哪种PET方法最好？→ 看哪个最接近Q揭示的真实状态
   - GRACE数据准确吗？→ 对比P-Q和P-Q-ΔS

---

## 项目结构

```
Budyko-Analysis/
├── README.md                          # 本文件
├── 研究思路.md                        # 详细研究思路（中文）
├── 代码库结构.md                      # 原始结构说明
│
├── src/                               # 核心代码模块
│   ├── budyko/                        # Budyko分析核心
│   │   ├── curves.py                  # Budyko曲线公式
│   │   ├── deviation.py               # Ibrahim偏差分析
│   │   ├── trajectory_jaramillo.py    # Jaramillo轨迹分析
│   │   └── water_balance.py           # 水量平衡计算（新增）
│   │
│   ├── models/                        # 模型模块
│   │   ├── pet_models.py              # PET模型集合
│   │   └── pet_lai_co2.py             # LAI+CO2 PET模型（核心创新）
│   │
│   ├── analysis/                      # 分析模块
│   │   ├── basin_screening.py         # 流域筛选
│   │   ├── deviation_attribution.py   # 偏差归因分析
│   │   └── snow_analysis.py           # 积雪影响分析
│   │
│   ├── data_processing/               # 数据处理
│   │   ├── cmip6_processor.py         # CMIP6数据处理
│   │   └── caravan_loader.py          # Caravan数据加载
│   │
│   ├── visualization/                 # 可视化
│   │   ├── budyko_plots.py            # Budyko空间图
│   │   └── direction_rose.py          # 方向玫瑰图
│   │
│   └── utils/                         # 工具函数
│       └── parallel_processing.py     # 并行计算（万级流域）
│
├── examples/                          # 完整示例（新增）
│   ├── 01_basic_budyko_analysis.py    # 基础Budyko分析
│   ├── 02_pet_comparison.py           # PET方法对比
│   ├── 03_deviation_analysis.py       # 偏差分析
│   ├── 04_trajectory_analysis.py      # 轨迹分析
│   ├── 05_attribution_analysis.py     # 归因分析
│   └── 06_complete_workflow.py        # 完整工作流
│
├── tests/                             # 测试文件
│   ├── unit/                          # 单元测试
│   │   ├── test_pet_models.py
│   │   ├── test_water_balance.py
│   │   └── test_deviation.py
│   └── integration/                   # 集成测试
│       └── test_full_workflow.py
│
├── docs/                              # 文档（新增）
│   ├── methodology.md                 # 方法论详解
│   ├── data_requirements.md           # 数据需求说明
│   ├── api_reference.md               # API文档
│   └── case_studies.md                # 案例研究
│
├── notebooks/                         # Jupyter教程（保留）
│   └── tutorial.ipynb                 # 交互式教程
│
└── scripts/                           # 批处理脚本
    ├── batch_processing/              # 批量处理
    └── parallel_analysis.py           # 并行分析
```

---

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/Budyko-Analysis.git
cd Budyko-Analysis

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 基础示例

```python
import numpy as np
import pandas as pd
from src.budyko.curves import BudykoCurves
from src.budyko.water_balance import WaterBalanceCalculator
from src.models.pet_lai_co2 import PETWithLAICO2

# ============ 1. 准备数据 ============
# 假设你有流域数据
P = np.array([800, 850, 900])  # 降水 (mm/yr)
Q = np.array([200, 220, 240])  # 径流 (mm/yr) - 核心数据！
T = np.array([15, 16, 14])     # 温度 (°C)
LAI = np.array([3.5, 3.8, 3.2]) # 叶面积指数
CO2 = np.array([380, 390, 400]) # CO2浓度 (ppm)

# ============ 2. 计算PET（创新方法） ============
pet_calculator = PETWithLAICO2()
PET = pet_calculator.calculate(
    temperature=T,
    lai=LAI,
    co2=CO2,
    # ... 其他气象变量
)

# ============ 3. 水量平衡计算（径流Q是核心） ============
wb_calc = WaterBalanceCalculator()
results = wb_calc.calculate_budyko_indices(
    P=P,
    Q=Q,      # 径流观测 - 决定实际蒸发！
    PET=PET
)

print("干旱指数 IA:", results['aridity_index'])
print("蒸发指数 IE:", results['evaporation_index'])
print("实际蒸发 EA:", results['actual_evaporation'])  # = P - Q

# ============ 4. Budyko曲线拟合 ============
budyko = BudykoCurves()
omega, fit_stats = budyko.fit_omega(
    ia_values=results['aridity_index'],
    ie_values=results['evaporation_index']
)

print(f"流域参数 ω: {omega:.2f}")
print(f"拟合R²: {fit_stats['r2']:.3f}")

# ============ 5. 计算偏差 ============
ie_theory = budyko.tixeront_fu(results['aridity_index'], omega)
deviation = results['evaporation_index'] - ie_theory

print("Budyko偏差:", deviation)
```

### 完整工作流示例

查看 `examples/06_complete_workflow.py` 获取：
- 数据加载（CMFD、径流、LAI、CO2）
- 流域筛选（面积、数据质量）
- 多种PET方法对比
- 偏差分析和归因
- 可视化输出

---

## 核心功能

### 1. 水量平衡计算（基于径流Q）

```python
from src.budyko.water_balance import WaterBalanceCalculator

wb = WaterBalanceCalculator()

# 基础计算：EA = P - Q
results = wb.calculate_budyko_indices(P, Q, PET)

# 考虑储量变化：EA = P - Q - ΔS
results_with_storage = wb.calculate_with_storage(P, Q, PET, delta_S)
```

**关键**：没有径流Q，实际蒸发EA无法确定，Budyko分析无从谈起！

### 2. 多种PET方法（含创新方法）

```python
from src.models.pet_models import PETModelFactory
from src.models.pet_lai_co2 import PETWithLAICO2

# 传统方法
pet_pm = PETModelFactory.create('penman_monteith')
pet_hs = PETModelFactory.create('hargreaves')
pet_pt = PETModelFactory.create('priestley_taylor')

# 创新方法：考虑LAI和CO2
pet_advanced = PETWithLAICO2()
PET_new = pet_advanced.calculate(
    temperature=T,
    humidity=RH,
    wind_speed=u2,
    radiation=Rn,
    lai=LAI,          # MODIS LAI
    co2=CO2,          # 大气CO2浓度
    latitude=lat
)
```

**创新点**：
- LAI动态调整表面阻抗
- CO2浓度影响气孔导度
- 更准确反映植被-大气相互作用

### 3. Ibrahim偏差分析

```python
from src.budyko.deviation import DeviationAnalysis

analyzer = DeviationAnalysis(period_length=20)

# 计算时段间偏差（基于径流Q）
distribution = analyzer.calculate_deviations(
    ia_i=period1_aridity,
    ie_obs_i=period1_evaporation,  # 来自Q: (P-Q)/P
    omega_i=period1_omega,
    ia_i_plus_1=period2_aridity,
    ie_obs_i_plus_1=period2_evaporation,  # 来自Q
    period_pair='Δ1-2'
)

# Wilcoxon检验
test_result = analyzer.wilcoxon_test(distribution)
```

### 4. Jaramillo轨迹分析

```python
from src.budyko.trajectory_jaramillo import TrajectoryAnalyzer

trajectory = TrajectoryAnalyzer()

# 计算Budyko空间运动（径流Q决定起点和终点）
movement = trajectory.calculate_movement(
    catchment_id='basin_001',
    period_1={'IA': ia1, 'IE': ie1, 'name': '1980-2000'},  # IE来自Q
    period_2={'IA': ia2, 'IE': ie2, 'name': '2000-2020'}   # IE来自Q
)

print(f"运动强度: {movement.intensity:.3f}")
print(f"方向角: {movement.direction_angle:.1f}°")
print(f"遵循曲线: {movement.follows_curve}")
```

### 5. 偏差归因分析

```python
from src.analysis.deviation_attribution import DeviationAttribution

attributor = DeviationAttribution()

# 添加驱动因子
attributor.add_drivers({
    'land_use_change': land_use_data,
    'irrigation': irrigation_data,
    'reservoir': reservoir_data,
    'snow_fraction': snow_data
})

# 归因分析（径流Q揭示的偏差是因变量）
attribution_results = attributor.attribute_deviation(
    deviation=budyko_deviation,  # 基于径流Q的偏差
    method='random_forest'
)

print("驱动因子重要性:")
print(attribution_results['importance'])
```

### 6. 并行处理（万级流域）

```python
from src.utils.parallel_processing import ParallelBudykoAnalyzer

parallel_analyzer = ParallelBudykoAnalyzer(n_jobs=-1)

# 批量处理6000+流域
results_all = parallel_analyzer.analyze_catchments(
    catchment_data=basin_data,
    pet_method='lai_co2',  # 使用创新PET方法
    n_catchments=6000
)
```

---

## 数据需求

### 必需数据

1. **径流数据 (Q)** - **核心！**
   - 格式：CSV/NetCDF
   - 字段：date, basin_id, runoff (mm/day 或 mm/month)
   - 来源：国家水文站网、Caravan数据集
   - **重要性**：决定实际蒸发EA，是Budyko分析的基石

2. **降水数据 (P)**
   - CMFD 0.1°格点数据
   - 时间：1960-2020
   - 单位：mm/day

3. **气象数据（计算PET）**
   - 温度、湿度、风速、辐射等
   - 来源：CMFD数据集

### 可选数据（用于创新分析）

4. **LAI数据**
   - MODIS MOD15A2H
   - 用于改进PET计算

5. **CO2浓度数据**
   - 全球CO2观测（Mauna Loa等）
   - 用于改进PET计算

6. **GRACE TWS数据**
   - 用于检验储量变化假设
   - EA' = P - Q - ΔS

7. **其他驱动因子**
   - 土地利用、NDVI、水库、灌溉、积雪等
   - 用于偏差归因分析

---

## 方法论概述

### 传统Budyko框架

```
Budyko假设：
在长时间尺度上，流域蒸发由供水（P）和需水（PET）共同决定

Fu-Budyko公式：
IE = 1 + IA - (1 + IA^ω)^(1/ω)

其中：
- IA = PET/P（干旱指数，X轴）
- IE = EA/P（蒸发指数，Y轴）
- EA = P - Q（实际蒸发，由径流Q确定！）
- ω = 流域特征参数
```

### Ibrahim偏差分析（2023）

```
Step 1: 拟合时段i的ω参数
        使用径流Q计算IE_obs,i = (P-Q)/P

Step 2: 计算时段i+1的偏差
        ε_IE,ω = IE_obs,i+1 - IE_theory,i+1(ω_i)

Step 3: 拟合偏态正态分布
        f(ε) ~ SkewNormal(ξ, λ, α)

Step 4: 时间稳定性分类
        Stable / Variable / Alternating / Shift

Step 5: 边际分布聚合
```

### Jaramillo轨迹分析（2022）

```
运动向量：
v = (ΔIA, ΔIE)
其中 ΔIE = IE_t2 - IE_t1 （由两个时期的径流Q决定）

运动强度：
I = |v| = sqrt(ΔIA² + ΔIE²)

运动方向：
θ = arctan2(ΔIA, ΔIE)

遵循曲线判断：
45° < θ < 90° 或 225° < θ < 270°
```

### 创新：LAI+CO2 PET方法

```
基于Penman-Monteith方程，改进气孔阻抗：

rs = rs_min * f(LAI) * f(CO2)

其中：
- f(LAI)：叶面积指数影响
  rs_LAI = rs_ref / max(LAI, 0.5)

- f(CO2)：CO2浓度影响
  rs_CO2 = 1 + k_co2 * log(CO2/CO2_ref)
  k_co2 ≈ 0.15-0.25（文献范围）
```

**物理机制**：
- LAI增加 → 蒸腾面积增大 → PET增加
- CO2增加 → 气孔部分关闭 → PET减少（CO2施肥效应）

---

## 应用场景

### 场景1：流域筛选

```python
from src.analysis.basin_screening import BasinScreener

screener = BasinScreener()

# 筛选标准
selected_basins = screener.select_basins(
    min_area=100,           # 最小面积 km²
    max_area=5000,          # 最大面积 km²
    min_data_years=20,      # 最少20年径流数据
    human_impact='low',     # 低人类活动影响
    data_quality='high'     # 高质量径流观测
)

print(f"筛选出 {len(selected_basins)} 个流域")
```

### 场景2：雪影响分析

```python
from src.analysis.snow_analysis import SnowImpactAnalyzer

snow_analyzer = SnowImpactAnalyzer()

# 识别积雪主导流域
snow_basins = snow_analyzer.identify_snow_basins(
    temperature_data=T,
    precipitation_data=P,
    threshold_temp=0  # 0°C以下视为降雪
)

# 量化雪对径流的贡献
snow_contribution = snow_analyzer.quantify_snow_contribution(
    basins=snow_basins,
    runoff_data=Q,  # 径流观测
    method='degree_day'
)
```

### 场景3：未来情景预测

```python
from src.data_processing.cmip6_processor import CMIP6Processor

cmip6 = CMIP6Processor()

# 加载CMIP6数据
future_climate = cmip6.load_scenario(
    scenario='ssp585',
    variables=['pr', 'tas', 'co2'],
    period='2020-2100'
)

# 预测未来Budyko关系（使用LAI+CO2 PET）
future_budyko = budyko_analyzer.project_future(
    climate_data=future_climate,
    pet_method='lai_co2',
    baseline_omega=current_omega
)
```

---

## 可视化

### Budyko空间图

```python
from src.visualization.budyko_plots import BudykoPlotter

plotter = BudykoPlotter()

# 绘制Budyko空间
fig, ax = plotter.plot_budyko_space(
    aridity_index=IA,
    evaporation_index=IE,  # 基于径流Q
    omega=2.6,
    color_by='deviation',
    size_by='basin_area'
)

plotter.add_budyko_curves(ax, omega_range=[1.5, 2.0, 2.5, 3.0])
plotter.add_water_energy_limits(ax)

plt.savefig('budyko_space.png', dpi=300)
```

### 轨迹方向玫瑰图

```python
from src.visualization.direction_rose import DirectionRosePlotter

rose_plotter = DirectionRosePlotter()

# 绘制方向玫瑰图
fig = rose_plotter.plot_direction_rose(
    angles=movement_angles,
    intensities=movement_intensities,
    n_bins=8,
    color_by_intensity=True
)

plt.savefig('trajectory_rose.png', dpi=300)
```

---

## 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/unit/test_pet_models.py
pytest tests/unit/test_water_balance.py

# 运行集成测试
pytest tests/integration/test_full_workflow.py

# 覆盖率报告
pytest --cov=src tests/
```

---

## 引用

如果你使用本框架，请引用：

### 方法论引用

1. **Ibrahim偏差分析**:
   ```
   Ibrahim, B., et al. (2023). On the Need to Update the Water-Energy Balance
   Framework for Predicting Catchment Runoff. Water Resources Research, 59(1).
   ```

2. **Jaramillo轨迹分析**:
   ```
   Jaramillo, F., et al. (2022). Fewer Basins Will Follow Their Budyko Curves
   Under Global Warming. Water Resources Research, 58(3).
   ```

3. **原始Budyko理论**:
   ```
   Budyko, M. I. (1974). Climate and Life. Academic Press.
   Fu, B. P. (1981). On the calculation of the evaporation from land surface.
   ```

### 数据引用

如果使用Caravan数据集:
```
Kratzert, F., et al. (2023). Caravan - A global community dataset for
large-sample hydrology. Scientific Data, 10(1), 61.
```

---

## 常见问题

### Q1: 为什么强调径流数据Q如此重要？

**A**: 因为在Budyko框架中，实际蒸发EA无法直接测量，只能通过水量平衡方程间接计算：
```
EA = P - Q - ΔS ≈ P - Q （长时间尺度）
```
没有径流Q，我们就无法得到蒸发指数IE = EA/P，整个Budyko分析无法进行。**Q是连接理论与现实的唯一桥梁**。

### Q2: LAI+CO2 PET方法的优势是什么？

**A**: 传统PET方法将植被作为静态参数，忽略了：
- 植被动态变化（LAI季节性、年际变化）
- CO2浓度上升导致的气孔响应
我们的方法动态考虑这两个因素，更准确反映变化环境下的蒸发需求。

### Q3: 如何选择合适的流域？

**A**: 建议标准：
- 面积：100-5000 km²（避免太小或太大）
- 数据：至少20年高质量径流观测
- 人类影响：尽量选择自然流域
- 地形：避免极端地形（如高山冰川）

### Q4: 多长的时间尺度合适？

**A**:
- 年尺度：最常用，消除季节波动
- 20年窗口：Ibrahim方法推荐，平滑年际变率
- 季节尺度：用于湿季/干季对比

### Q5: 如何处理缺失数据？

**A**:
- 径流Q：严格质控，缺失>10%的年份剔除
- 气象数据：可用临近站点插值
- LAI：可用气候态平均值填充

---

## 贡献指南

欢迎贡献！请遵循：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 代码规范

- 遵循PEP 8
- 添加docstring（Google风格）
- 编写单元测试
- 更新文档

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 联系方式

- Issues: [GitHub Issues](https://github.com/yourusername/Budyko-Analysis/issues)
- Email: your.email@example.com

---

## 致谢

- 感谢Ibrahim et al. (2023)和Jaramillo et al. (2022)的开创性工作
- 感谢CMFD、Caravan、MODIS团队提供高质量数据
- 感谢所有贡献者

---

## 更新日志

### v1.0.0 (2025-10-31)
- 初始版本发布
- 实现Ibrahim偏差分析
- 实现Jaramillo轨迹分析
- **创新：LAI+CO2 PET方法**
- 完整示例和文档

---

**祝研究顺利！如有问题，请随时提出Issue。**
