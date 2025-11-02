#!/usr/bin/env python3
# examples/complete_workflow_example.py
"""
完整的Budyko分析工作流示例

本示例演示了从数据加载到结果输出的完整流程，
重点展示径流数据(Q)在整个分析中的核心作用。

工作流程：
1. 数据准备（P, Q, 气象数据, LAI, CO2）
2. PET计算（传统方法 vs LAI+CO2创新方法）
3. 水量平衡计算（基于径流Q）
4. Budyko曲线拟合
5. 偏差计算和分析
6. 轨迹分析
7. 驱动因子归因
8. 可视化和报告

注意：本示例使用模拟数据。使用真实数据时，请替换数据加载部分。
"""

from pathlib import Path
import sys

# Ensure <repo_root>/src is importable regardless of CWD
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 导入核心模块
from budyko.water_balance import WaterBalanceCalculator, RunoffAnalyzer
from budyko.curves import BudykoCurves
from budyko.deviation import DeviationAnalysis, TemporalStability
from budyko.trajectory_jaramillo import TrajectoryAnalyzer, MovementStatistics
from models.pet_lai_co2 import PETWithLAICO2, PETComparator
from models.pet_models import PETModelFactory
from analysis.deviation_attribution import DeviationAttributor

# 设置
sns.set_style('whitegrid')

# 导入中文字体配置
try:
    from utils.plotting_config import setup_chinese_fonts
    setup_chinese_fonts()
except ImportError:
    # 备用配置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['figure.dpi'] = 100

# 创建输出目录
output_dir = Path('../outputs/complete_workflow')
output_dir.mkdir(parents=True, exist_ok=True)


def generate_synthetic_data(n_basins: int = 50, n_years: int = 30):
    """
    生成模拟流域数据

    在实际应用中，请替换为真实数据加载函数
    """
    print("生成模拟流域数据...")
    np.random.seed(42)

    data = {}

    for basin_id in range(n_basins):
        # 基础气候特征
        mean_P = np.random.uniform(600, 1200)  # 平均降水
        mean_T = np.random.uniform(10, 20)     # 平均温度
        aridity_level = np.random.uniform(0.5, 2.5)  # 干旱程度

        # 年度数据
        years = np.arange(1990, 1990 + n_years)

        # 降水（含年际变率）
        P = mean_P + 150 * np.random.randn(n_years)
        P = np.maximum(P, 300)  # 确保合理范围

        # 气温（含增温趋势）
        T = mean_T + 0.03 * np.arange(n_years) + 1.5 * np.random.randn(n_years)

        # LAI（季节性 + 趋势）
        LAI = 3.0 + 0.02 * np.arange(n_years) + 0.5 * np.random.randn(n_years)
        LAI = np.clip(LAI, 0.5, 8)

        # CO2浓度（逐年增加）
        CO2 = 360 + 2 * np.arange(n_years)

        # 其他气象变量
        RH = 65 + 10 * np.random.randn(n_years)
        u2 = 2 + 0.5 * np.random.randn(n_years)
        Rn = 200 + 30 * np.random.randn(n_years)

        # PET（简化）
        PET = mean_P * aridity_level + 100 * np.random.randn(n_years)
        PET = np.maximum(PET, 400)

        # 径流Q（核心！）
        # RC = f(aridity, 土地利用, 人类活动)
        base_RC = 0.4 / (1 + aridity_level)  # 干旱度越高，RC越低

        # 添加趋势和扰动
        RC_trend = -0.002 * np.arange(n_years)  # 径流系数下降趋势
        RC = base_RC + RC_trend + 0.05 * np.random.randn(n_years)
        RC = np.clip(RC, 0.05, 0.8)

        Q = RC * P

        # 驱动因子（用于归因分析）
        irrigation = np.maximum(0, 0.1 + 0.01 * np.arange(n_years) + 0.05 * np.random.randn(n_years))
        forest_cover = 0.6 - 0.005 * np.arange(n_years) + 0.05 * np.random.randn(n_years)

        data[f'basin_{basin_id:03d}'] = {
            'basin_id': f'basin_{basin_id:03d}',
            'years': years,
            'P': P,
            'Q': Q,
            'T': T,
            'RH': RH,
            'u2': u2,
            'Rn': Rn,
            'LAI': LAI,
            'CO2': CO2,
            'irrigation': irrigation,
            'forest_cover': forest_cover,
            'latitude': 25 + 15 * np.random.rand(),
            'elevation': 100 + 2000 * np.random.rand()
        }

    print(f"  生成了 {n_basins} 个流域，每个 {n_years} 年数据")
    return data


def step1_pet_calculation(basin_data: dict):
    """
    步骤1：计算PET（对比传统方法和创新方法）
    """
    print("\n" + "="*60)
    print("步骤1：PET计算（传统方法 vs LAI+CO2方法）")
    print("="*60)

    pet_results = {}

    for basin_id, data in list(basin_data.items())[:3]:  # 演示前3个流域
        print(f"\n处理流域: {basin_id}")

        # 传统Penman-Monteith
        pet_traditional = PETModelFactory.create('penman_monteith')
        PET_traditional = pet_traditional.calculate(
            temp_avg=data['T'],
            temp_max=data['T'] + 5,
            temp_min=data['T'] - 5,
            rh_mean=data['RH'],
            wind_speed=data['u2'],
            solar_radiation=data['Rn'] * 0.0864,  # W/m² to MJ/m²/day
            latitude=data['latitude'],
            elevation=data['elevation'],
            day_of_year=np.arange(1, len(data['T']) + 1) * 12  # 简化
        )

        # 创新LAI+CO2方法
        pet_lai_co2 = PETWithLAICO2(
            elevation=data['elevation'],
            latitude=data['latitude']
        )
        PET_lai_co2 = pet_lai_co2.calculate(
            temperature=data['T'],
            humidity=data['RH'],
            wind_speed=data['u2'],
            radiation=data['Rn'],
            lai=data['LAI'],
            co2=data['CO2']
        )

        # 对比
        diff = PET_lai_co2 - PET_traditional
        diff_pct = 100 * diff / PET_traditional

        print(f"  传统PET: {np.mean(PET_traditional):.1f} ± {np.std(PET_traditional):.1f} mm/yr")
        print(f"  LAI+CO2 PET: {np.mean(PET_lai_co2):.1f} ± {np.std(PET_lai_co2):.1f} mm/yr")
        print(f"  差异: {np.mean(diff):.1f} mm/yr ({np.mean(diff_pct):.1f}%)")

        pet_results[basin_id] = {
            'traditional': PET_traditional,
            'lai_co2': PET_lai_co2
        }

    return pet_results


def step2_water_balance(basin_data: dict, pet_results: dict):
    """
    步骤2：水量平衡计算（基于径流Q）
    """
    print("\n" + "="*60)
    print("步骤2：水量平衡计算（径流Q是核心！）")
    print("="*60)

    wb_calc = WaterBalanceCalculator()
    wb_results = {}

    for basin_id in pet_results.keys():
        data = basin_data[basin_id]
        print(f"\n流域: {basin_id}")

        # 径流诊断
        runoff_analyzer = RunoffAnalyzer()
        diagnosis = runoff_analyzer.diagnose_runoff_data(
            Q=data['Q'],
            P=data['P']
        )

        print(f"\n  径流数据诊断:")
        print(f"    平均径流: {diagnosis['basic_stats']['mean_runoff']:.1f} mm/yr")
        print(f"    径流系数: {diagnosis['runoff_coefficient']['mean']:.3f}")
        print(f"    数据质量: 缺失率 {diagnosis['data_quality']['pct_missing']:.1f}%")

        # 水量平衡计算（使用LAI+CO2 PET）
        results = wb_calc.calculate_budyko_indices(
            P=data['P'],
            Q=data['Q'],  # 径流 - 核心数据！
            PET=pet_results[basin_id]['lai_co2']
        )

        print(f"\n  水量平衡结果:")
        print(f"    实际蒸发EA = P - Q = {np.mean(results.actual_evaporation):.1f} mm/yr")
        print(f"    干旱指数IA = PET/P = {np.mean(results.aridity_index):.2f}")
        print(f"    蒸发指数IE = EA/P = (P-Q)/P = {np.mean(results.evaporation_index):.2f}")

        wb_results[basin_id] = results

    return wb_results


def step3_budyko_fitting(wb_results: dict):
    """
    步骤3：Budyko曲线拟合
    """
    print("\n" + "="*60)
    print("步骤3：Budyko曲线拟合")
    print("="*60)

    budyko = BudykoCurves()
    fitting_results = {}

    for basin_id, results in wb_results.items():
        print(f"\n流域: {basin_id}")

        # 拟合ω参数
        omega, fit_stats = budyko.fit_omega(
            ia_values=results.aridity_index,
            ie_values=results.evaporation_index
        )

        print(f"  最优ω参数: {omega:.3f}")
        print(f"  拟合R²: {fit_stats['r2']:.3f}")
        print(f"  RMSE: {fit_stats['rmse']:.4f}")

        # 计算理论IE和偏差
        ie_theory = budyko.tixeront_fu(results.aridity_index, omega)
        deviation = results.evaporation_index - ie_theory

        print(f"  平均偏差: {np.mean(deviation):.4f}")
        print(f"  偏差范围: [{np.min(deviation):.4f}, {np.max(deviation):.4f}]")

        fitting_results[basin_id] = {
            'omega': omega,
            'fit_stats': fit_stats,
            'ie_theory': ie_theory,
            'deviation': deviation
        }

    return fitting_results


def step4_trajectory_analysis(basin_data: dict, wb_results: dict):
    """
    步骤4：Budyko空间轨迹分析（Jaramillo方法）
    """
    print("\n" + "="*60)
    print("步骤4：Budyko空间轨迹分析")
    print("="*60)

    trajectory_analyzer = TrajectoryAnalyzer()
    movements = []

    for basin_id in list(wb_results.keys())[:5]:  # 演示前5个流域
        results = wb_results[basin_id]

        # 将30年分为两个时期
        mid_point = len(results.aridity_index) // 2

        period_1 = {
            'IA': np.mean(results.aridity_index[:mid_point]),
            'IE': np.mean(results.evaporation_index[:mid_point]),
            'name': 'Period_1990-2004'
        }

        period_2 = {
            'IA': np.mean(results.aridity_index[mid_point:]),
            'IE': np.mean(results.evaporation_index[mid_point:]),
            'name': 'Period_2005-2019'
        }

        # 计算运动
        movement = trajectory_analyzer.calculate_movement(
            catchment_id=basin_id,
            period_1=period_1,
            period_2=period_2
        )

        print(f"\n流域 {basin_id}:")
        print(f"  起点: IA={movement.ia_t1:.3f}, IE={movement.ie_t1:.3f}")
        print(f"  终点: IA={movement.ia_t2:.3f}, IE={movement.ie_t2:.3f}")
        print(f"  运动强度: {movement.intensity:.4f}")
        print(f"  方向角: {movement.direction_angle:.1f}°")
        print(f"  遵循曲线: {movement.follows_curve}")
        print(f"  运动类型: {movement.movement_type}")

        movements.append(movement)

    # 统计
    n_following = sum(1 for m in movements if m.follows_curve)
    pct_following = 100 * n_following / len(movements)
    print(f"\n总结：{pct_following:.1f}% 的流域遵循Budyko曲线运动")

    return movements


def step5_attribution_analysis(basin_data: dict, fitting_results: dict):
    """
    步骤5：偏差归因分析
    """
    print("\n" + "="*60)
    print("步骤5：偏差归因分析（径流Q揭示的偏差 → 寻找病因）")
    print("="*60)

    # 收集所有流域的偏差和驱动因子
    all_deviations = []
    all_irrigation = []
    all_forest_cover = []
    all_lai_trend = []

    for basin_id in fitting_results.keys():
        deviation = fitting_results[basin_id]['deviation']
        data = basin_data[basin_id]

        # 计算趋势
        years = np.arange(len(deviation))
        lai_trend = np.polyfit(years, data['LAI'], 1)[0]

        all_deviations.append(np.mean(deviation))
        all_irrigation.append(np.mean(data['irrigation']))
        all_forest_cover.append(np.mean(data['forest_cover']))
        all_lai_trend.append(lai_trend)

    # 归因分析
    attributor = DeviationAttributor()
    attributor.set_deviation(np.array(all_deviations))
    attributor.add_driver('irrigation', np.array(all_irrigation))
    attributor.add_driver('forest_cover', np.array(all_forest_cover))
    attributor.add_driver('lai_trend', np.array(all_lai_trend))

    # 相关分析
    print("\n相关性分析:")
    corr_results = attributor.correlate_drivers()
    print(corr_results.to_string(index=False))

    # 随机森林归因
    print("\n随机森林归因:")
    rf_results = attributor.random_forest_attribution()
    print(f"  解释方差: {rf_results.explained_variance:.2%}")
    print("\n  驱动因子重要性排序:")
    for driver, importance in sorted(rf_results.driver_importance.items(),
                                    key=lambda x: x[1], reverse=True):
        print(f"    {driver:20s}: {importance:.2%}")

    return rf_results


def step6_visualization(basin_data: dict, wb_results: dict,
                        fitting_results: dict, movements: list):
    """
    步骤6：综合可视化
    """
    print("\n" + "="*60)
    print("步骤6：生成可视化图表")
    print("="*60)

    # 图1：Budyko空间散点图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：所有流域的Budyko空间
    ax1 = axes[0]
    for basin_id, results in wb_results.items():
        omega = fitting_results[basin_id]['omega']
        deviation = fitting_results[basin_id]['deviation']

        scatter = ax1.scatter(
            results.aridity_index,
            results.evaporation_index,
            c=deviation,
            cmap='RdBu_r',
            s=20,
            alpha=0.6,
            vmin=-0.1,
            vmax=0.1
        )

    # Budyko理论曲线
    ia_range = np.linspace(0.1, 5, 100)
    for omega in [1.5, 2.0, 2.5, 3.0]:
        ie_curve = BudykoCurves().tixeront_fu(ia_range, omega)
        ax1.plot(ia_range, ie_curve, 'k--', alpha=0.3, linewidth=1)

    # 边界
    ax1.plot([0, 5], [0, 1], 'k-', alpha=0.5, linewidth=1.5, label='能量限制')
    ax1.plot([1, 5], [1, 1], 'k-', alpha=0.5, linewidth=1.5, label='水分限制')

    ax1.set_xlabel('干旱指数 IA (PET/P)', fontsize=12)
    ax1.set_ylabel('蒸发指数 IE (EA/P) = (P-Q)/P', fontsize=12)
    ax1.set_title('Budyko空间（径流Q决定Y轴）', fontsize=13)
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax1, label='Budyko偏差')

    # 右图：轨迹分析
    ax2 = axes[1]
    for movement in movements:
        color = 'blue' if movement.follows_curve else 'red'
        ax2.arrow(
            movement.ia_t1, movement.ie_t1,
            movement.delta_ia, movement.delta_ie,
            head_width=0.02, head_length=0.03,
            fc=color, ec=color, alpha=0.6
        )
        ax2.plot(movement.ia_t1, movement.ie_t1, 'ko', markersize=5)

    # Budyko曲线
    for omega in [2.0, 2.5]:
        ie_curve = BudykoCurves().tixeront_fu(ia_range, omega)
        ax2.plot(ia_range, ie_curve, 'k--', alpha=0.3, linewidth=1)

    ax2.set_xlabel('干旱指数 IA', fontsize=12)
    ax2.set_ylabel('蒸发指数 IE', fontsize=12)
    ax2.set_title('Budyko空间轨迹（Jaramillo方法）', fontsize=13)
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='遵循曲线'),
        Line2D([0], [0], color='red', lw=2, label='偏离曲线')
    ]
    ax2.legend(handles=legend_elements)

    plt.tight_layout()
    # Save to figures folder with script-based filename
    figures_dir = Path(__file__).resolve().parent / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    script_stem = Path(__file__).stem
    fig_path = figures_dir / f"{script_stem}__budyko_space_analysis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  已保存: {fig_path}")

    plt.close()


def generate_report(basin_data: dict, pet_results: dict, wb_results: dict,
                   fitting_results: dict, movements: list, attribution_results):
    """
    步骤7：生成分析报告
    """
    print("\n" + "="*60)
    print("步骤7：生成分析报告")
    print("="*60)

    report_lines = []
    report_lines.append("="*70)
    report_lines.append("Budyko框架分析完整报告")
    report_lines.append("="*70)
    report_lines.append("")

    # 1. 数据概况
    report_lines.append("1. 数据概况")
    report_lines.append("-" * 70)
    report_lines.append(f"流域数量: {len(basin_data)}")
    report_lines.append(f"时间跨度: {basin_data[list(basin_data.keys())[0]]['years'][0]} - "
                       f"{basin_data[list(basin_data.keys())[0]]['years'][-1]}")
    report_lines.append("")

    # 2. PET方法对比
    report_lines.append("2. PET方法对比")
    report_lines.append("-" * 70)
    report_lines.append("传统方法 vs LAI+CO2创新方法:")

    all_diff = []
    for basin_id, pet in pet_results.items():
        diff = np.mean(pet['lai_co2'] - pet['traditional'])
        all_diff.append(diff)

    report_lines.append(f"  平均差异: {np.mean(all_diff):.1f} mm/yr")
    report_lines.append(f"  差异范围: [{np.min(all_diff):.1f}, {np.max(all_diff):.1f}] mm/yr")
    report_lines.append("  结论: LAI和CO2对PET有显著影响")
    report_lines.append("")

    # 3. 水量平衡
    report_lines.append("3. 水量平衡（基于径流Q）")
    report_lines.append("-" * 70)

    all_ia = []
    all_ie = []
    all_rc = []

    for results in wb_results.values():
        all_ia.extend(results.aridity_index)
        all_ie.extend(results.evaporation_index)
        all_rc.extend(results.runoff_coefficient)

    report_lines.append(f"  干旱指数IA (PET/P): {np.mean(all_ia):.2f} ± {np.std(all_ia):.2f}")
    report_lines.append(f"  蒸发指数IE (EA/P = (P-Q)/P): {np.mean(all_ie):.2f} ± {np.std(all_ie):.2f}")
    report_lines.append(f"  径流系数RC (Q/P): {np.mean(all_rc):.2f} ± {np.std(all_rc):.2f}")
    report_lines.append("")
    report_lines.append("  **径流Q是决定IE的核心数据！**")
    report_lines.append("")

    # 4. Budyko拟合
    report_lines.append("4. Budyko曲线拟合")
    report_lines.append("-" * 70)

    all_omega = [fit['omega'] for fit in fitting_results.values()]
    all_deviation = []
    for fit in fitting_results.values():
        all_deviation.extend(fit['deviation'])

    report_lines.append(f"  流域参数ω: {np.mean(all_omega):.2f} ± {np.std(all_omega):.2f}")
    report_lines.append(f"  范围: [{np.min(all_omega):.2f}, {np.max(all_omega):.2f}]")
    report_lines.append(f"  平均偏差: {np.mean(all_deviation):.4f}")
    report_lines.append(f"  偏差标准差: {np.std(all_deviation):.4f}")
    report_lines.append("")

    # 5. 轨迹分析
    report_lines.append("5. Budyko空间轨迹分析")
    report_lines.append("-" * 70)

    n_following = sum(1 for m in movements if m.follows_curve)
    pct_following = 100 * n_following / len(movements)

    report_lines.append(f"  分析流域数: {len(movements)}")
    report_lines.append(f"  遵循Budyko曲线: {n_following} ({pct_following:.1f}%)")
    report_lines.append(f"  偏离Budyko曲线: {len(movements) - n_following} ({100 - pct_following:.1f}%)")
    report_lines.append("")

    # 6. 归因分析
    report_lines.append("6. 偏差归因分析")
    report_lines.append("-" * 70)
    report_lines.append(f"  解释方差: {attribution_results.explained_variance:.2%}")
    report_lines.append("\n  驱动因子重要性排序:")

    for driver, importance in sorted(attribution_results.driver_importance.items(),
                                    key=lambda x: x[1], reverse=True):
        report_lines.append(f"    {driver:20s}: {importance:.2%}")

    report_lines.append("")

    # 7. 核心结论
    report_lines.append("="*70)
    report_lines.append("核心结论")
    report_lines.append("="*70)
    report_lines.append("")
    report_lines.append("1. 径流数据Q是Budyko分析的基石")
    report_lines.append("   - 通过Q计算实际蒸发: EA = P - Q")
    report_lines.append("   - 通过Q确定蒸发指数: IE = (P - Q) / P")
    report_lines.append("   - 没有Q，无法进行Budyko分析")
    report_lines.append("")
    report_lines.append("2. LAI和CO2对PET有显著影响")
    report_lines.append(f"   - 平均差异: {np.mean(all_diff):.1f} mm/yr")
    report_lines.append("   - 建议使用考虑LAI和CO2的PET方法")
    report_lines.append("")
    report_lines.append("3. 部分流域显著偏离Budyko曲线")
    report_lines.append(f"   - {100 - pct_following:.1f}% 的流域不遵循曲线运动")
    report_lines.append("   - 需要进一步分析偏离原因")
    report_lines.append("")
    report_lines.append("4. 驱动因子归因结果")
    top_driver = max(attribution_results.driver_importance.items(), key=lambda x: x[1])
    report_lines.append(f"   - 最重要驱动因子: {top_driver[0]} ({top_driver[1]:.2%})")
    report_lines.append("")

    # 保存报告
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\n报告已保存: {report_path}")
    print("\n" + "\n".join(report_lines))


def main():
    """
    主函数：执行完整工作流
    """
    print("\n" + "="*70)
    print("Budyko分析完整工作流")
    print("="*70)
    print("\n本示例演示径流数据(Q)在Budyko分析中的核心作用")
    print("从数据加载到最终报告的完整流程\n")

    # 步骤0：生成/加载数据
    print("步骤0：准备数据...")
    basin_data = generate_synthetic_data(n_basins=20, n_years=30)

    # 步骤1：PET计算
    pet_results = step1_pet_calculation(basin_data)

    # 步骤2：水量平衡
    wb_results = step2_water_balance(basin_data, pet_results)

    # 步骤3：Budyko拟合
    fitting_results = step3_budyko_fitting(wb_results)

    # 步骤4：轨迹分析
    movements = step4_trajectory_analysis(basin_data, wb_results)

    # 步骤5：归因分析
    attribution_results = step5_attribution_analysis(basin_data, fitting_results)

    # 步骤6：可视化
    step6_visualization(basin_data, wb_results, fitting_results, movements)

    # 步骤7：报告
    generate_report(basin_data, pet_results, wb_results, fitting_results,
                   movements, attribution_results)

    print("\n" + "="*70)
    print("完整工作流执行完毕！")
    print(f"结果已保存到: {output_dir.absolute()}")
    print("="*70)


if __name__ == '__main__':
    main()
