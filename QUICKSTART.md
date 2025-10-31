# 快速开始指南

10分钟快速上手Budyko-Analysis框架！

---

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/Budyko-Analysis.git
cd Budyko-Analysis

# 安装依赖
pip install -r requirements.txt
```

---

## 最简示例（5行代码）

```python
import numpy as np
from src.budyko.water_balance import WaterBalanceCalculator
from src.budyko.curves import BudykoCurves
from src.models.pet_lai_co2 import PETWithLAICO2

# 1. 准备数据（P, Q, 气象）
P = np.array([800, 850, 900, 950, 1000])   # 降水
Q = np.array([200, 220, 240, 260, 280])     # 径流 - 核心！
T = np.array([15, 16, 14, 15, 16])          # 温度
LAI = np.array([3.5, 3.8, 3.2, 3.6, 3.9])   # 叶面积指数
CO2 = np.array([380, 390, 400, 410, 420])   # CO2浓度

# 2. 计算PET（创新方法）
pet_calc = PETWithLAICO2()
PET = pet_calc.calculate(
    temperature=T,
    humidity=np.full(5, 60),
    wind_speed=np.full(5, 2),
    radiation=np.full(5, 200),
    lai=LAI,
    co2=CO2
)

# 3. 水量平衡计算（基于径流Q）
wb_calc = WaterBalanceCalculator()
results = wb_calc.calculate_budyko_indices(P, Q, PET)

# 4. Budyko曲线拟合
budyko = BudykoCurves()
omega, stats = budyko.fit_omega(results.aridity_index, results.evaporation_index)

# 5. 结果
print(f"干旱指数 IA: {np.mean(results.aridity_index):.2f}")
print(f"蒸发指数 IE: {np.mean(results.evaporation_index):.2f}")
print(f"流域参数 ω: {omega:.2f}")
```

---

## 完整示例

运行完整工作流示例：

```bash
cd examples
python complete_workflow_example.py
```

这将：
- 生成模拟流域数据
- 计算多种PET方法并对比
- 进行水量平衡计算
- 拟合Budyko曲线
- 分析轨迹运动
- 驱动因子归因
- 生成可视化和报告

输出位置：`../outputs/complete_workflow/`

---

## 测试

运行测试确保安装正确：

```bash
cd tests
python test_complete_workflow.py
```

或使用pytest：

```bash
pytest tests/ -v
```

---

## 核心概念速览

### 1. 径流Q是核心

```
实际蒸发 EA = P - Q
蒸发指数 IE = EA / P = (P - Q) / P

没有Q，无法进行Budyko分析！
```

### 2. Budyko空间

```
X轴：干旱指数 IA = PET / P
Y轴：蒸发指数 IE = EA / P

Budyko曲线：IE = f(IA, ω)
```

### 3. 创新PET方法

```
传统方法：忽略LAI和CO2变化
创新方法：动态考虑LAI和CO2对气孔导度的影响

PET_new = f(T, RH, u, Rn, LAI, CO2)
```

---

## 数据准备

### 必需数据

1. **径流Q** - 最重要！
   - 格式：CSV或NetCDF
   - 单位：mm/day 或 mm/month
   - 来源：水文站观测、Caravan数据集

2. **降水P**
   - 来源：CMFD、ERA5等

3. **气象数据（计算PET）**
   - 温度、湿度、风速、辐射

### 可选数据（创新分析）

4. **LAI数据**
   - 来源：MODIS MOD15A2H

5. **CO2浓度**
   - 来源：Mauna Loa观测

6. **GRACE TWS**（储量变化）

7. **驱动因子**（归因分析）
   - 土地利用、灌溉、水库等

---

## 常见问题

### Q1: 运行示例时出错怎么办？

检查：
1. Python版本 >= 3.8
2. 依赖是否全部安装：`pip install -r requirements.txt`
3. 路径是否正确（使用绝对路径）

### Q2: 如何使用自己的数据？

替换`complete_workflow_example.py`中的`generate_synthetic_data()`函数为你的数据加载函数。

### Q3: PET计算很慢？

如果数据量大，考虑：
1. 使用更简单的PET方法（如Hargreaves）
2. 先聚合到月尺度或年尺度
3. 并行处理（参见`src/utils/parallel_processing.py`）

### Q4: 如何解释结果？

参见`docs/METHODOLOGY.md`详细方法论说明。

---

## 下一步

1. **阅读文档**：
   - `README.md` - 项目概览
   - `docs/METHODOLOGY.md` - 方法论详解

2. **查看示例**：
   - `examples/` - 各种示例脚本

3. **运行测试**：
   - `tests/` - 单元测试和集成测试

4. **使用真实数据**：
   - 替换为你的流域数据
   - 调整参数和配置

5. **贡献**：
   - 发现bug？提Issue
   - 有改进？提Pull Request

---

## 联系

- Issues: [GitHub Issues](https://github.com/yourusername/Budyko-Analysis/issues)
- Email: your.email@example.com

祝研究顺利！
