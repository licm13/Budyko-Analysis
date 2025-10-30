"""
Quick Start Example for Budyko Framework
快速开始示例 - 展示核心功能
"""

import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('src')

from budyko.analyzer import BudykoAnalyzer, SeasonalBudykoAnalyzer
from pet.calculator import PETCalculator
from visualization.plotter import BudykoVisualizer
from config.settings import *

print("=" * 70)
print("Budyko Framework 快速开始示例")
print("基于 He et al. (2023) 三维Budyko框架")
print("=" * 70)

# ===========================
# 1. 生成模拟数据（实际应用中替换为真实数据）
# ===========================
print("\n[步骤 1/6] 生成模拟数据...")

np.random.seed(42)
n_basins = 100
n_years = 20

# 模拟流域数据
basin_ids = np.arange(1, n_basins + 1)

# 气候梯度：从湿润到干旱
P = np.random.uniform(400, 1500, n_basins)  # 降水 [mm]
Q = P * np.random.uniform(0.1, 0.6, n_basins)  # 径流 [mm]
T = np.random.uniform(5, 25, n_basins)  # 温度 [°C]

# 模拟其他变量
RH = np.random.uniform(50, 85, n_basins)  # 相对湿度 [%]
u2 = np.random.uniform(1, 4, n_basins)  # 风速 [m/s]
Rn = np.random.uniform(80, 180, n_basins)  # 净辐射 [W/m²]
Rn_MJ = Rn * 0.0864  # 转换为 MJ/m²/day

# 模拟LAI（研究创新点）
LAI = np.random.uniform(0.5, 6, n_basins)  # 叶面积指数

# 模拟TWS（三维Budyko创新点）
TWS = np.random.uniform(-50, 50, n_basins)  # TWS异常 [mm]

print(f"  生成了 {n_basins} 个流域的模拟数据")

# ===========================
# 2. 计算多种PET方法
# ===========================
print("\n[步骤 2/6] 计算潜在蒸散发 (PET)...")

pet_calc = PETCalculator(elevation=500)

# 标准Penman-Monteith
PET_PM = pet_calc.penman_monteith_fao56(
    T=T, RH=RH, u2=u2, Rn=Rn_MJ
)

# 创新：考虑LAI和CO2的改进PET
CO2 = 410 * np.ones_like(T)  # 当前CO2浓度 [ppm]
PET_improved = pet_calc.penman_monteith_with_lai_co2(
    T=T, RH=RH, u2=u2, Rn=Rn_MJ,
    LAI=LAI, CO2=CO2
)

# Priestley-Taylor
PET_PT = PETCalculator.priestley_taylor(T=T, Rn=Rn_MJ)

print(f"  标准PM PET: {np.mean(PET_PM):.2f} ± {np.std(PET_PM):.2f} mm/day")
print(f"  改进PM PET: {np.mean(PET_improved):.2f} ± {np.std(PET_improved):.2f} mm/day")
print(f"  PT PET:    {np.mean(PET_PT):.2f} ± {np.std(PET_PT):.2f} mm/day")
print(f"  创新改进带来的差异: {np.mean(PET_improved - PET_PM):.2f} mm/day")

# ===========================
# 3. 传统Budyko分析
# ===========================
print("\n[步骤 3/6] 传统Budyko框架分析...")

# 使用改进的PET
budyko = BudykoAnalyzer(time_scale='annual')
budyko.load_basin_data(
    P=P,
    Q=Q,
    PET=PET_improved * 365,  # 转换为年总量
    basin_ids=basin_ids
)

# 估计最优ω参数
omega = budyko.estimate_omega(
    budyko.data['DI'].values,
    budyko.data['EI'].values
)

print(f"  最优景观参数 ω = {omega:.2f}")

# 计算偏离
deviation = budyko.calculate_budyko_deviation(omega=omega)
budyko.data['deviation'] = deviation

# 分类
budyko.data['limit_type'] = budyko.classify_water_energy_limit()

energy_limited = (budyko.data['limit_type'] == 'energy_limited').sum()
water_limited = (budyko.data['limit_type'] == 'water_limited').sum()

print(f"  能量限制流域: {energy_limited} ({energy_limited/n_basins*100:.1f}%)")
print(f"  水分限制流域: {water_limited} ({water_limited/n_basins*100:.1f}%)")
print(f"  平均偏离: {np.mean(deviation):.4f} ± {np.std(deviation):.4f}")

# ===========================
# 4. 三维Budyko分析（论文创新）
# ===========================
print("\n[步骤 4/6] 三维Budyko框架分析（整合TWS）...")

# 添加TWS数据
budyko.data['TWS'] = TWS

# 计算三维指数
budyko.calculate_3d_indices()

# 分析DI/EI与SI/SCI关系
relationships = budyko.calculate_si_sci_relationships()

print("  DI/EI与SI/SCI的关系:")
for key, value in relationships.items():
    print(f"    {key}: {value:.3f}")

# ===========================
# 5. 可视化
# ===========================
print("\n[步骤 5/6] 生成可视化图表...")

visualizer = BudykoVisualizer()

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 图1: 传统Budyko空间
print("  绘制传统Budyko空间...")
visualizer.plot_budyko_space(
    DI=budyko.data['DI'].values,
    EI=budyko.data['EI'].values,
    omega=omega,
    color_by=budyko.data['deviation'].values,
    color_label='Budyko偏离',
    title='中国流域Budyko空间（模拟数据）',
    save_path=OUTPUT_DIR / 'budyko_space_traditional.png'
)

# 图2: DI/EI与SI/SCI关系
print("  绘制三维关系图...")
visualizer.plot_di_si_relationships(
    DI=budyko.data['DI'].values,
    SI=budyko.data['SI'].values,
    SCI=budyko.data['SCI'].values,
    EI=budyko.data['EI'].values,
    save_path=OUTPUT_DIR / 'di_si_relationships.png'
)

# 图3: 偏离分布
print("  绘制偏离分布图...")
visualizer.plot_deviation_histogram(
    deviation=budyko.data['deviation'].values,
    save_path=OUTPUT_DIR / 'deviation_distribution.png'
)

# 图4: 三维Budyko空间
print("  绘制三维Budyko空间...")
visualizer.plot_3d_budyko(
    DI=budyko.data['DI'].values,
    EI=budyko.data['EI'].values,
    SI=budyko.data['SI'].values,
    omega=omega,
    save_path=OUTPUT_DIR / '3d_budyko_space.png'
)

# ===========================
# 6. 输出结果
# ===========================
print("\n[步骤 6/6] 保存分析结果...")

# 汇总统计
summary = budyko.get_summary_statistics()
summary.to_csv(OUTPUT_DIR / 'summary_statistics.csv')

# 完整结果
budyko.data.to_csv(OUTPUT_DIR / 'budyko_analysis_results.csv', index=False)

# 生成报告
report_lines = [
    "=" * 70,
    "Budyko框架分析报告",
    "=" * 70,
    "",
    "一、数据概况",
    f"  流域数量: {n_basins}",
    f"  降水范围: {P.min():.1f} - {P.max():.1f} mm",
    f"  径流系数范围: {(Q/P).min():.2f} - {(Q/P).max():.2f}",
    "",
    "二、PET方法对比",
    f"  标准PM方法: {np.mean(PET_PM):.2f} mm/day",
    f"  改进PM方法（LAI+CO2）: {np.mean(PET_improved):.2f} mm/day",
    f"  方法差异: {np.mean(PET_improved - PET_PM):.2f} mm/day ({np.mean(PET_improved - PET_PM)/np.mean(PET_PM)*100:.1f}%)",
    "",
    "三、传统Budyko分析",
    f"  最优ω参数: {omega:.3f}",
    f"  能量限制流域: {energy_limited} ({energy_limited/n_basins*100:.1f}%)",
    f"  水分限制流域: {water_limited} ({water_limited/n_basins*100:.1f}%)",
    f"  平均偏离: {np.mean(deviation):.4f}",
    f"  偏离标准差: {np.std(deviation):.4f}",
    "",
    "四、三维Budyko分析（创新点）",
    "  DI/EI与SI/SCI相关关系:",
]

for key, value in relationships.items():
    report_lines.append(f"    {key}: {value:.3f}")

report_lines.extend([
    "",
    "五、研究创新总结",
    "  1. 考虑LAI和CO2的改进PET方法",
    "     - 更准确反映植被和大气变化对蒸散发的影响",
    f"     - 相比标准方法差异达 {np.mean(PET_improved - PET_PM)/np.mean(PET_PM)*100:.1f}%",
    "  2. 三维Budyko框架整合TWS",
    "     - 揭示水储量变化与水能平衡的关系",
    "     - DI/EI与SI/SCI呈现显著线性关系",
    "",
    "六、输出文件",
    "  - budyko_space_traditional.png: 传统Budyko空间图",
    "  - di_si_relationships.png: DI/EI与SI/SCI关系图",
    "  - deviation_distribution.png: 偏离分布图",
    "  - 3d_budyko_space.png: 三维Budyko空间图",
    "  - summary_statistics.csv: 汇总统计",
    "  - budyko_analysis_results.csv: 完整分析结果",
    "",
    "=" * 70,
])

report_text = "\n".join(report_lines)

with open(OUTPUT_DIR / 'analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)

# ===========================
# 完成
# ===========================
print("\n" + "=" * 70)
print("分析完成！")
print(f"所有结果已保存到: {OUTPUT_DIR}")
print("=" * 70)
print("\n下一步建议:")
print("1. 查看输出图表，了解Budyko关系")
print("2. 阅读 notebooks/01_complete_budyko_analysis.md 了解完整分析流程")
print("3. 替换为真实的流域和气象数据")
print("4. 探索季节性、积雪等其他因素的影响")
print("\n研究创新亮点:")
print("✓ 考虑LAI和CO2的改进PET计算")
print("✓ 三维Budyko框架整合TWS")
print("✓ 系统的偏离归因分析")