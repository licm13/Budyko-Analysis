# Budyko框架方法论详解

本文档详细说明Budyko-Analysis框架的理论基础和计算方法。

---

## 目录

1. [核心理论](#核心理论)
2. [径流数据的核心地位](#径流数据的核心地位)
3. [水量平衡方程](#水量平衡方程)
4. [Budyko曲线](#budyko曲线)
5. [Ibrahim偏差分析](#ibrahim偏差分析)
6. [Jaramillo轨迹分析](#jaramillo轨迹分析)
7. [创新：LAI+CO2 PET方法](#创新laiCO2-pet方法)
8. [驱动因子归因](#驱动因子归因)

---

## 核心理论

### Budyko假设

Budyko (1974)提出：在长时间尺度上（如年尺度），流域的蒸发由两个因素共同决定：
- **供水能力**：降水P（水分供应）
- **需水能力**：潜在蒸发PET（能量需求）

### Budyko空间

Budyko框架使用二维空间描述流域水分平衡：

**X轴：干旱指数 (Aridity Index)**
```
IA = PET / P
```
- IA < 1：湿润气候（水分充足）
- IA = 1：过渡气候
- IA > 1：干旱气候（能量充足）

**Y轴：蒸发指数 (Evaporation Index)**
```
IE = EA / P
```
- EA: 实际蒸发（mm）
- IE: 降水中被蒸发的比例

### 物理限制

1. **能量限制** (Energy Limit)
   ```
   IE = IA  (斜率为1的直线)
   ```
   当IA很小时（湿润区），蒸发受能量限制

2. **水分限制** (Water Limit)
   ```
   IE = 1  (水平线)
   ```
   当IA很大时（干旱区），蒸发受水分限制

---

## 径流数据的核心地位

### 为什么径流Q如此重要？

**实际蒸发EA无法直接测量！**

对于一个流域，我们无法像测量降水和径流那样直接测量其总的实际蒸发量EA。我们只能通过水量平衡方程间接计算：

```
水量平衡方程：P = Q + EA + ΔS

移项得到：EA = P - Q - ΔS
```

在长时间尺度上（如年尺度、20年平均），我们通常假设储量变化ΔS ≈ 0：

```
EA ≈ P - Q
```

**因此，径流Q是计算实际蒸发EA、进而得到蒸发指数IE的唯一途径！**

### 径流数据的三重角色

**1. "尺子"角色：衡量流域真实的水分消耗**

```python
# 理论位置（由气候决定）
IA_theory = PET / P
IE_theory = f(IA, ω)  # Budyko公式

# 实际位置（由径流Q决定！）
EA_actual = P - Q
IE_actual = EA_actual / P = (P - Q) / P

# 偏差
deviation = IE_actual - IE_theory
```

**2. "病人"角色：揭示流域"健康状态"**

- 径流异常减少（Q↓）→ IE增加 → 可能是灌溉取水、水库蓄水
- 径流异常增加（Q↑）→ IE减少 → 可能是森林砍伐、城市化

**3. "参照物"角色：检验理论和方法**

- 哪种PET方法最好？→ 看哪个最接近Q揭示的真实状态
- GRACE储量数据准确吗？→ 对比P-Q和P-Q-ΔS

### 没有径流Q会怎样？

如果没有径流观测Q：
- ❌ 无法计算实际蒸发EA
- ❌ 无法确定蒸发指数IE
- ❌ 无法在Budyko空间中定位流域
- ❌ 无法计算Budyko偏差
- ❌ 整个Budyko分析无法进行！

**结论：径流Q是Budyko分析的绝对基石！**

---

## 水量平衡方程

### 基本形式

```
P - Q = EA + ΔS
```

**变量说明：**
- P: 降水（mm）- 可直接测量或从气象数据获取
- Q: 径流（mm）- **核心观测数据**
- EA: 实际蒸发（mm）- 无法直接测量，通过Q计算
- ΔS: 储量变化（mm）- 包括土壤水、地下水、积雪等

### 长时间尺度简化

在年尺度或多年平均：
```
ΔS ≈ 0

因此：EA ≈ P - Q
```

这是Budyko框架最常用的假设。

### 考虑储量变化（GRACE数据）

如果有GRACE卫星观测的陆地水储量（TWS）：
```
EA = P - Q - ΔS

其中 ΔS = TWS(t) - TWS(t-1)
```

这可以检验"ΔS ≈ 0"假设的有效性。

---

## Budyko曲线

### Fu-Budyko参数化公式（推荐）

```
IE = 1 + IA - (1 + IA^ω)^(1/ω)
```

**参数：**
- ω: 流域特征参数（通常1.5 < ω < 3.5）
  - ω小 → 曲线向下，蒸发能力弱
  - ω大 → 曲线向上，蒸发能力强

**物理意义：**
- ω反映流域的蒸发能力
- 受植被、土壤、地形等影响

### 原始Budyko公式（1948）

```
IE = sqrt(IA * tanh(1/IA) * (1 - exp(-IA)))
```

非参数形式，适用于全球平均。

### ω参数的拟合

给定观测数据(IA_obs, IE_obs)，通过最小二乘法拟合ω：

```python
def objective(ω):
    IE_pred = budyko_formula(IA_obs, ω)
    residuals = IE_obs - IE_pred
    return sum(residuals^2)

ω_optimal = minimize(objective, initial_guess=2.6)
```

---

## Ibrahim偏差分析

Ibrahim et al. (2023) 提出了系统化的Budyko偏差分析方法。

### 步骤1：时段划分

将长时间序列（如60年）划分为多个时段（如20年窗口）：
- T1: 1960-1979
- T2: 1980-1999
- T3: 2000-2019

### 步骤2：ω参数拟合

对每个时段Ti，拟合其特定的ω参数：

```python
for period in periods:
    IA_i = PET_i / P_i
    IE_i = (P_i - Q_i) / P_i  # 关键：基于径流Q
    ω_i = fit_omega(IA_i, IE_i)
```

### 步骤3：偏差计算

使用时段i的ω参数，计算时段i+1的偏差：

```
ε_IE,ω = IE_obs,i+1 - IE_expected,i+1(ω_i)

其中：
- IE_obs,i+1 = (P_i+1 - Q_i+1) / P_i+1  （基于Q观测）
- IE_expected,i+1 = Budyko_formula(IA_i+1, ω_i)
```

**核心思想**：如果流域稳定遵循Budyko假设，ω_i应该能准确预测i+1时段的IE。

### 步骤4：偏差分布拟合

对每个时段对的年度偏差，拟合偏态正态分布：

```
f(ε) = (2/λ) * φ((ε-ξ)/λ) * Φ(α(ε-ξ)/λ)
```

参数：
- ξ: 位置参数（偏差中心）
- λ: 尺度参数（偏差幅度）
- α: 形状参数（偏态程度）

### 步骤5：时间稳定性分类

通过Kolmogorov-Smirnov检验和趋势分析，将流域分为4类：

| 类别 | 描述 | 特征 |
|------|------|------|
| **Stable** | 稳定 | 偏差分布在时段间无显著差异 |
| **Variable** | 变化 | 偏差分布随机变化，无规律 |
| **Alternating** | 交替 | 偏差随气候条件（如干旱/湿润）交替 |
| **Shift** | 漂移 | 偏差呈系统性单向变化趋势 |

---

## Jaramillo轨迹分析

Jaramillo et al. (2022) 提出了Budyko空间轨迹分析方法。

### 核心概念

流域在Budyko空间中的位置随时间变化，形成轨迹：

```
时期1: (IA_t1, IE_t1)
时期2: (IA_t2, IE_t2)

运动向量: v = (ΔIA, ΔIE)
其中：
  ΔIA = IA_t2 - IA_t1
  ΔIE = IE_t2 - IE_t1  （由两个时期的Q决定）
```

### 运动特征

**1. 运动强度 (Intensity)**
```
I = |v| = sqrt(ΔIA² + ΔIE²)
```

**2. 运动方向 (Direction)**
```
θ = arctan2(ΔIA, ΔIE)
```
从垂直方向（IE轴）顺时针计算，单位：度

### 遵循曲线判断

根据方向角θ判断流域是否遵循Budyko曲线：

**遵循曲线 (Following)**：
- 45° < θ < 90°（第一象限，向右上）
- 225° < θ < 270°（第三象限，向左下）

**偏离曲线 (Deviating)**：
- 其他方向

**物理意义**：
- 遵循曲线：IA和IE协同变化
- 偏离曲线：可能有外部扰动（人类活动、土地利用变化等）

### 运动类型分类

结合ΔIA和ΔIE的符号：

| ΔIA | ΔIE | 描述 |
|-----|-----|------|
| > 0 | > 0 | 变干旱 + 蒸发增加 |
| > 0 | < 0 | 变干旱 + 蒸发减少 |
| < 0 | > 0 | 变湿润 + 蒸发增加 |
| < 0 | < 0 | 变湿润 + 蒸发减少 |

---

## 创新：LAI+CO2 PET方法

这是本框架的核心创新，前人未系统实现。

### 为什么需要改进PET？

传统PET方法（如Penman-Monteith）的局限：
1. **忽略植被动态**：LAI季节性、年际变化
2. **忽略CO2效应**：大气CO2浓度上升对气孔导度的影响

### 改进思路

基于Penman-Monteith方程，动态调整表面阻抗：

```
ET = (Δ * Rn + ρ * cp * (es - ea) / ra) / (Δ + γ * (1 + rs/ra))

核心改进：rs = f(LAI, CO2)
```

### LAI影响机制

**物理过程**：
- LAI增加 → 蒸腾叶面积增大 → 总阻抗降低 → PET增加

**实现方法**（指数衰减）：
```
rs_LAI = rs_min * exp(-k * (LAI - LAI_ref))

参数：
- rs_min: 参考最小表面阻抗 (70 s/m)
- LAI_ref: 参考LAI (3.0)
- k: 衰减系数 (0.5)
```

### CO2影响机制

**物理过程**：
- CO2浓度升高 → 气孔部分关闭（减少水分损失）
- 气孔导度降低 → 表面阻抗增加 → PET降低

这是**CO2施肥效应**的水分侧体现。

**实现方法**（对数线性）：
```
rs_CO2_factor = 1 + k_CO2 * ln(CO2 / CO2_ref)

参数：
- CO2_ref: 参考浓度 (380 ppm)
- k_CO2: 敏感性系数 (0.15-0.25，文献范围)
```

### 综合表面阻抗

```
rs_final = rs_LAI * rs_CO2_factor
```

### 预期效果

1. **LAI趋势**：
   - 全球绿化 → LAI增加 → PET增加

2. **CO2趋势**：
   - CO2上升 → PET降低

3. **净效应**：
   - 取决于LAI和CO2的相对变化
   - 在植被增加区域（如中国），两者部分抵消
   - 在植被退化区域，CO2效应占优

---

## 驱动因子归因

### 目的

量化各种驱动因子对Budyko偏差的贡献。

### 因变量：Budyko偏差

```
deviation = IE_obs - IE_theory

其中 IE_obs = (P - Q) / P  （基于径流Q！）
```

### 自变量：驱动因子

**气候因子**：
- 温度趋势
- 降水模式变化

**植被因子**：
- LAI趋势
- NDVI变化
- 森林覆盖变化

**人类活动**：
- 灌溉面积
- 水库蓄水
- 城市化

**其他**：
- 积雪变化
- 土地利用

### 方法1：相关分析

```python
for driver in drivers:
    correlation, p_value = pearson(deviation, driver)
```

识别关键驱动因子。

### 方法2：多元线性回归

```
deviation = β0 + β1*X1 + β2*X2 + ... + ε

重要性 = |βi| / Σ|βi|
```

### 方法3：随机森林（推荐）

```python
model = RandomForest()
model.fit(X_drivers, y_deviation)

importance = model.feature_importances_
```

优点：
- 捕捉非线性关系
- 处理多重共线性
- 特征重要性排序

### 解释

例如：
```
灌溉重要性 = 35%
森林损失 = 25%
LAI趋势 = 20%
水库 = 15%
其他 = 5%
```

结论：该流域的Budyko偏差主要由灌溉驱动，其次是森林损失。

---

## 总结

本方法论文档阐述了：

1. **Budyko理论**：水分-能量平衡的基本框架
2. **径流Q的核心地位**：是连接理论与现实的唯一桥梁
3. **水量平衡**：EA = P - Q，Q是关键
4. **偏差分析**：Ibrahim方法系统化偏差量化
5. **轨迹分析**：Jaramillo方法识别变化模式
6. **创新PET**：LAI+CO2动态调整，更准确反映植被-气候互动
7. **归因分析**：量化各驱动因子对偏差的贡献

**核心要点**：
- 径流Q是Budyko分析的基石
- LAI和CO2显著影响PET
- 流域偏差受多种因素驱动
- 系统化分析需要整合多种方法

---

## 参考文献

1. Budyko, M. I. (1974). Climate and Life. Academic Press.
2. Fu, B. P. (1981). On the calculation of the evaporation from land surface. Scientia Atmospherica Sinica, 5(1), 23-31.
3. Ibrahim, B., et al. (2023). On the Need to Update the Water-Energy Balance Framework for Predicting Catchment Runoff. Water Resources Research, 59(1).
4. Jaramillo, F., et al. (2022). Fewer Basins Will Follow Their Budyko Curves Under Global Warming. Water Resources Research, 58(3).
5. Ball, J. T., et al. (1987). A model predicting stomatal conductance and its contribution to the control of photosynthesis under different environmental conditions. Progress in Photosynthesis Research, 4, 221-224.
6. Medlyn, B. E., et al. (2001). Stomatal conductance of forest species after long-term exposure to elevated CO2 concentration: a synthesis. New Phytologist, 149(2), 247-264.
7. Yang, Y., et al. (2019). Evapotranspiration on a greening Earth. Nature Reviews Earth & Environment, 1, 27-37.
