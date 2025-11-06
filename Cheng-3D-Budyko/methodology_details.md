# 附录:核心方法数学详解 
# Appendix: Detailed Mathematical Methodology

本文档提供论文中所有核心方法的详细数学推导和说明。  
This document provides detailed mathematical derivations and explanations for all core methods in the paper.

---

## 目录 (Table of Contents)
1. [Budyko框架与径流估算](#1-budyko框架与径流估算)
2. [BFC曲线与基流估算](#2-bfc曲线与基流估算)
3. [数字滤波基流分离](#3-数字滤波基流分离)
4. [Boosted Regression Trees (BRT)](#4-boosted-regression-trees-brt)
5. [SHAP可解释性分析](#5-shap可解释性分析)
6. [评估指标](#6-评估指标)

---

## 1. Budyko框架与径流估算
## 1. Budyko Framework and Runoff Estimation

### 1.1 基本原理 (Basic Principles)

Budyko (1961) 框架基于长期水量平衡和能量平衡约束:  
The Budyko (1961) framework is based on long-term water and energy balance constraints:

**水量平衡 (Water Balance):**
$$P = E_a + Q + \Delta S$$

在长期尺度上,土壤水分储量变化可忽略 ($\Delta S \approx 0$),因此:  
Over long-term scales, changes in soil water storage are negligible ($\Delta S \approx 0$), thus:

$$P = E_a + Q \quad \text{(方程 1 / Equation 1)}$$

其中 / Where:
- $P$ = 降水 (mm/yr) / Precipitation (mm/yr)
- $E_a$ = 实际蒸散发 (mm/yr) / Actual evapotranspiration (mm/yr)
- $Q$ = 径流 (mm/yr) / Runoff (mm/yr)
- $\Delta S$ = 土壤水分储量变化 / Change in soil water storage

**能量约束 (Energy Constraint):**
$$E_a \leq E_p$$

其中 $E_p$ 是潜在蒸散发 (potential evapotranspiration)。  
Where $E_p$ is potential evapotranspiration.

### 1.2 Budyko曲线方程 (Budyko Curve Equation)

Fu (1981) 和 Zhang et al. (2004) 提出的参数化形式:  
Parametric form proposed by Fu (1981) and Zhang et al. (2004):

$$\frac{E_a}{P} = 1 + \frac{E_p}{P} - \left[1 + \left(\frac{E_p}{P}\right)^\alpha\right]^{1/\alpha} \quad \text{(方程 2 / Equation 2)}$$

其中 / Where:
- $\alpha$ = Budyko参数,反映流域特性 (植被、土壤、地形等) / Budyko parameter reflecting catchment properties (vegetation, soil, topography, etc.)
- $\alpha$ 取值范围: $1 \leq \alpha < \infty$
- 当 $\alpha \to 1$: 蒸散发接近能量限制 / Evaporation approaches energy limit
- 当 $\alpha \to \infty$: 蒸散发接近水分限制 / Evaporation approaches water limit

### 1.3 径流系数方程 (Runoff Coefficient Equation)

结合方程1和方程2,得到径流系数 (RFC = Q/P):  
Combining Equations 1 and 2, we obtain the runoff coefficient (RFC = Q/P):

$$\frac{Q}{P} = 1 - \frac{E_a}{P} = -\frac{E_p}{P} + \left[1 + \left(\frac{E_p}{P}\right)^\alpha\right]^{1/\alpha} \quad \text{(方程 3 / Equation 3)}$$

**物理意义 (Physical Interpretation):**
- 湿润区 ($E_p/P < 1$): 径流主要受能量限制,Q/P较高
- 干旱区 ($E_p/P > 1$): 径流主要受水分限制,Q/P较低
- Humid regions ($E_p/P < 1$): Runoff mainly energy-limited, high Q/P
- Arid regions ($E_p/P > 1$): Runoff mainly water-limited, low Q/P

### 1.4 参数α的物理意义 (Physical Meaning of Parameter α)

参数 $\alpha$ 综合反映了以下流域特征:  
Parameter $\alpha$ comprehensively reflects the following catchment characteristics:

1. **植被特征 (Vegetation)**: 
   - 植被覆盖度 (Vegetation cover): NDVI, LAI
   - 根系深度 (Rooting depth): RD
   - 水分利用效率 (Water use efficiency): WUE

2. **土壤特征 (Soil)**:
   - 土壤厚度 (Soil thickness): STHI
   - 孔隙度 (Porosity): SPO
   - 持水能力 (Water holding capacity)

3. **地形特征 (Topography)**:
   - 海拔 (Elevation): ELEV
   - 坡度 (Slope): SLO
   - 地形湿度指数 (Topographic wetness index): CTI

4. **气候特征 (Climate)**:
   - 温度 (Temperature): TC
   - 季节性 (Seasonality): SAI
   - 雪的影响 (Snow influence): SWE

5. **人类活动 (Human activities)**:
   - 人类影响指数 (Human footprint): HFP

---

## 2. BFC曲线与基流估算
## 2. BFC Curve and Baseflow Estimation

### 2.1 基流分割概念 (Baseflow Partitioning Concept)

总径流可分解为:  
Total runoff can be decomposed into:

$$Q = Q_b + Q_q$$

其中 / Where:
- $Q_b$ = 基流 (Baseflow): 来自地下水和延迟源 (groundwater and delayed sources)
- $Q_q$ = 快流 (Quickflow): 来自快速径流过程 (fast runoff processes)

### 2.2 BFC曲线推导 (BFC Curve Derivation)

Cheng et al. (2021) 扩展Budyko框架以估算基流。引入**潜在基流** ($Q_{b,p}$) 概念:  
Cheng et al. (2021) extended the Budyko framework to estimate baseflow. Introducing the concept of **potential baseflow** ($Q_{b,p}$):

**定义 (Definition):**  
$Q_{b,p}$ 表示在水分不受限制情况下的基流上限,类似于 $E_p$ 对蒸散发的概念。  
$Q_{b,p}$ represents the upper limit of baseflow when water is not limiting, analogous to $E_p$ for evapotranspiration.

**修正的水量平衡 (Modified Water Balance):**

考虑基流分割后的水量平衡:  
Considering water balance after baseflow partitioning:

$$P = E_a + Q_b + Q_q$$

假设 $Q_q$ 首先从 $P$ 中分离,剩余水量用于 $E_a$ 和 $Q_b$:  
Assuming $Q_q$ is separated first from $P$, remaining water is partitioned into $E_a$ and $Q_b$:

$$P - Q_q = E_a + Q_b$$

将其类比于Budyko框架:  
Analogous to the Budyko framework:

$$P - Q_q = E_a + Q_b \approx P \cdot \left[1 + \frac{E_p + Q_{b,p}}{P} - \left(1 + \left(\frac{E_p + Q_{b,p}}{P}\right)^\alpha\right)^{1/\alpha}\right]$$

### 2.3 BFC曲线最终方程 (Final BFC Curve Equation)

通过代数推导,得到基流系数 (BFC = Q_b/P):  
Through algebraic derivation, we obtain the baseflow coefficient (BFC = Q_b/P):

$$\frac{Q_b}{P} = \frac{Q_{b,p}}{P} + \left[1 + \left(\frac{E_p}{P}\right)^\alpha\right]^{1/\alpha} - \left[1 + \left(\frac{E_p}{P} + \frac{Q_{b,p}}{P}\right)^\alpha\right]^{1/\alpha} \quad \text{(方程 4 / Equation 4)}$$

**关键特性 (Key Properties):**

1. **物理约束 (Physical Constraints)**:
   - $0 \leq Q_b/P \leq Q/P \leq 1$
   - 当 $Q_{b,p} \to 0$: $Q_b \to 0$ (无基流 / No baseflow)
   - 当 $Q_{b,p} \to \infty$: $Q_b \to Q$ (所有径流为基流 / All runoff is baseflow)

2. **参数依赖性 (Parameter Dependence)**:
   - $Q_b$ 增加随着 $Q_{b,p}$ 增加 (正相关 / Positive correlation)
   - $Q_b$ 增加随着 $E_p/P$ 减少 (负相关 / Negative correlation)
   - 参数 $\alpha$ 对 $Q_b$ 的影响较小 (Small influence of $\alpha$ on $Q_b$)

### 2.4 快流估算 (Quickflow Estimation)

$$Q_q = Q - Q_b$$

或用系数形式 / Or in coefficient form:

$$\text{QFC} = \text{RFC} - \text{BFC}$$

### 2.5 基流指数 (Baseflow Index)

$$\text{BFI} = \frac{Q_b}{Q} = \frac{\text{BFC}}{\text{RFC}}$$

**物理意义 (Physical Interpretation):**
- BFI ≈ 1: 基流主导的流域 (Baseflow-dominated catchment)
- BFI ≈ 0: 快流主导的流域 (Quickflow-dominated catchment)

---

## 3. 数字滤波基流分离
## 3. Digital Filter Baseflow Separation

### 3.1 Lyne-Hollick (LH) 方法 (Lyne-Hollick Method)

**递归滤波方程 (Recursive Filter Equation):**

前向滤波 / Forward pass:
$$q_t = f_1 \cdot q_{t-1} + \frac{1 + f_1}{2}(Q_t - Q_{t-1})$$

其中 / Where:
- $q_t$ = 时刻 $t$ 的快流 (quickflow at time $t$)
- $Q_t$ = 时刻 $t$ 的总径流 (total runoff at time $t$)
- $f_1$ = 滤波参数 (filter parameter), 通常取 0.925
- $1 + f_1$ = 递归常数 (recession constant)

**约束条件 (Constraints):**
$$0 \leq q_t \leq Q_t$$

基流计算 / Baseflow calculation:
$$Q_{b,t} = Q_t - q_t$$

### 3.2 三次滤波过程 (Three-Pass Filtering)

为减少相位失真,采用三次滤波:  
To reduce phase distortion, three-pass filtering is applied:

1. **前向滤波 (Forward pass)**: 从 $t=1$ 到 $t=n$
2. **后向滤波 (Backward pass)**: 从 $t=n$ 到 $t=1$
3. **再次前向滤波 (Forward pass again)**: 从 $t=1$ 到 $t=n$

### 3.3 长期平均基流 (Long-term Mean Baseflow)

$$\overline{Q_b} = \frac{1}{n}\sum_{t=1}^{n} Q_{b,t}$$

其中 $n$ 是观测时间长度(天数)。  
Where $n$ is the length of observation period (days).

---

## 4. Boosted Regression Trees (BRT)
## 4. Boosted Regression Trees

### 4.1 基本原理 (Basic Principles)

BRT结合了两种技术:  
BRT combines two techniques:

1. **回归树 (Regression Trees)**: 递归二分法构建决策树
2. **提升算法 (Boosting)**: 集成多个弱学习器提高性能

### 4.2 回归树构建 (Regression Tree Construction)

**递归分割 (Recursive Partitioning):**

对于数据集 $\{(x_i, y_i)\}_{i=1}^{N}$,寻找最佳分割点:  
For dataset $\{(x_i, y_i)\}_{i=1}^{N}$, find optimal split:

$$\min_{j,s} \left[\min_{c_1}\sum_{x_i \in R_1(j,s)}(y_i - c_1)^2 + \min_{c_2}\sum_{x_i \in R_2(j,s)}(y_i - c_2)^2\right]$$

其中 / Where:
- $j$ = 分割变量 (splitting variable)
- $s$ = 分割点 (split point)
- $R_1(j,s) = \{x|x_j \leq s\}$ 和 $R_2(j,s) = \{x|x_j > s\}$ 是分割区域

### 4.3 提升算法 (Boosting Algorithm)

**梯度提升 (Gradient Boosting):**

初始化 / Initialize:
$$F_0(x) = \bar{y}$$

对于 $m = 1$ 到 $M$ (树的数量 / number of trees):

1. 计算残差 / Compute residuals:
$$r_{im} = y_i - F_{m-1}(x_i)$$

2. 拟合新的回归树 / Fit new regression tree:
$$h_m(x) = \text{Tree}(r_m, x)$$

3. 更新模型 / Update model:
$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$

其中 $\nu$ 是学习率 (learning rate)。  
Where $\nu$ is the learning rate.

最终预测 / Final prediction:
$$\hat{y} = F_M(x) = F_0(x) + \nu\sum_{m=1}^{M}h_m(x)$$

### 4.4 超参数 (Hyperparameters)

本研究使用的超参数设置:  
Hyperparameter settings used in this study:

1. **树复杂度 (Tree Complexity, tc)**: 12
   - 控制每棵树的最大交互深度 / Controls maximum interaction depth of each tree

2. **学习率 (Learning Rate, lr)**: 0.01
   - 控制每棵树的贡献 / Controls contribution of each tree
   - 较小的值需要更多树,但通常性能更好 / Smaller values need more trees but generally perform better

3. **袋装分数 (Bag Fraction, bf)**: 0.50
   - 每次迭代随机选择的数据比例 / Proportion of data randomly selected for each iteration
   - 增加模型的随机性和泛化能力 / Increases model randomness and generalization

4. **树的数量 (Number of Trees)**: 自适应选择 / Adaptively selected
   - 通过交叉验证确定 / Determined via cross-validation

### 4.5 损失函数 (Loss Function)

对于回归问题,使用均方误差:  
For regression problems, mean squared error is used:

$$L(y, F(x)) = \frac{1}{N}\sum_{i=1}^{N}(y_i - F(x_i))^2$$

---

## 5. SHAP可解释性分析
## 5. SHAP (SHapley Additive exPlanations)

### 5.1 Shapley值理论 (Shapley Value Theory)

基于博弈论的Shapley值:  
Shapley values based on game theory:

$$\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!(|N|-|S|-1)!}{|N|!}[f_{S\cup\{j\}}(x_{S\cup\{j\}}) - f_S(x_S)]$$

其中 / Where:
- $\phi_j$ = 特征 $j$ 的Shapley值 (Shapley value for feature $j$)
- $N$ = 所有特征的集合 (set of all features)
- $S$ = 特征子集 (subset of features)
- $f_S(x_S)$ = 使用特征子集 $S$ 的预测 (prediction using feature subset $S$)

### 5.2 SHAP的优势 (Advantages of SHAP)

1. **加法性 (Additivity)**:
$$f(x) = \phi_0 + \sum_{j=1}^{p}\phi_j$$

2. **局部准确性 (Local Accuracy)**:  
   对单个预测的准确解释 / Accurate explanation for individual predictions

3. **一致性 (Consistency)**:  
   如果模型改变使特征贡献增加,Shapley值不减 / If model change increases feature contribution, Shapley value does not decrease

4. **处理特征相关性 (Handling Feature Correlation)**:  
   通过条件期望考虑特征间相关性 / Considers feature correlations via conditional expectation

### 5.3 主导驱动因素识别 (Identifying Dominant Drivers)

对于每个流域,计算13个特征的SHAP值:  
For each catchment, compute SHAP values for 13 features:

$$|\phi_j| = \text{absolute SHAP value for feature } j$$

主导驱动因素定义为:  
Dominant driver defined as:

$$j^* = \arg\max_{j \in \{1,...,13\}} |\phi_j|$$

---

## 6. 评估指标
## 6. Evaluation Metrics

### 6.1 决定系数 (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

其中 / Where:
- $y_i$ = 观测值 (observed value)
- $\hat{y}_i$ = 预测值 (predicted value)
- $\bar{y}$ = 观测值平均 (mean of observed values)
- $R^2 = 1$: 完美预测 / Perfect prediction
- $R^2 = 0$: 预测不比平均值好 / Prediction no better than mean

### 6.2 均方根误差 (Root Mean Square Error)

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**单位 (Unit)**: 与原始数据相同 (mm/yr for this study)

**解释 (Interpretation)**: 值越小表示预测越准确 / Smaller values indicate better prediction

### 6.3 Nash-Sutcliffe效率系数 (Nash-Sutcliffe Efficiency)

$$\text{NSE} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

**范围 (Range)**: $(-\infty, 1]$
- NSE = 1: 完美模拟 / Perfect simulation
- NSE = 0: 模型预测等同于观测平均值 / Model predictions equal to observed mean
- NSE < 0: 模型不如使用平均值 / Model worse than using mean

### 6.4 Kling-Gupta效率系数 (Kling-Gupta Efficiency)

$$\text{KGE} = 1 - \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2}$$

其中 / Where:
- $r$ = 相关系数 (correlation coefficient)
- $\alpha = \frac{\sigma_{\hat{y}}}{\sigma_y}$ = 变异性比 (variability ratio)
- $\beta = \frac{\bar{\hat{y}}}{\bar{y}}$ = 偏差比 (bias ratio)

**优势 (Advantages)**:  
同时考虑相关性、变异性和偏差 / Simultaneously considers correlation, variability, and bias

---

## 参考文献 (References)

1. Budyko, M. I. (1961). The heat balance of the Earth's surface. *Soviet Geography*, 2(4), 3-13.

2. Fu, B. P. (1981). On the calculation of the evaporation from land surface. *Chinese Journal of Atmospheric Sciences*, 5(1), 23-31.

3. Zhang, L., Hickel, K., Dawes, W. R., Chiew, F. H. S., Western, A. W., & Briggs, P. R. (2004). A rational function approach for estimating mean annual evapotranspiration. *Water Resources Research*, 40(2), W02502.

4. Cheng, S., Cheng, L., Liu, P., Qin, S., Zhang, L., Xu, C. Y., et al. (2021). An analytical baseflow coefficient curve for depicting the spatial variability of mean annual catchment baseflow. *Water Resources Research*, 57(8), e2020WR029529.

5. Lyne, V., & Hollick, M. (1979). Stochastic time-variable rainfall-runoff modelling. *Institute of Engineers Australia National Conference*.

6. Elith, J., Leathwick, J. R., & Hastie, T. (2008). A working guide to boosted regression trees. *Journal of Animal Ecology*, 77(4), 802-813.

7. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

---

**文档版本 (Document Version)**: 1.0  
**最后更新 (Last Updated)**: 2025-01-01  
**维护者 (Maintainer)**: [Your Name]
