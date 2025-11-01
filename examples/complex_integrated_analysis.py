#!/usr/bin/env python3
# examples/complex_integrated_analysis.py
"""
复杂算例：整合多种Budyko分析方法

本示例演示了如何在一个（模拟的）流域上应用一个完整的、复杂的研究工作流，
整合了您研究思路中的所有核心点。

工作流程：
1.  **数据模拟**: 生成一个60年的流域时间序列 (1961-2020)。
    - P, T, 气象数据
    - LAI (模拟绿化趋势)
    - CO2 (模拟上升趋势)
    - TWS (模拟储量变化)
    - Q (模拟径流，并使其**偏离**理论值，模拟人类活动影响)
2.  **PET计算对比**:
    - 计算标准 Penman-Monteith (FAO56)
    - 计算创新的 **LAI+CO2 PET**
3.  **水量平衡计算 (核心)**:
    - 基于 **Q_obs** 计算 **EA = P - Q**
    - 基于 **TWS** 计算 **ΔS**
    - 计算 2D Budyko 指数 (IA, IE)
    - 计算 3D Budyko 指数 (EA_ext = P-Q-ΔS, SCI)
4.  **Ibrahim偏差分析 (2025)**:
    - 划分3个20年时段 (T1, T2, T3)
    - 拟合 T1 的 ω1
    - 计算 T2 相对 T1 曲线的偏差 (ε_T2)
    - 计算 T3 相对 T2 曲线的偏差 (ε_T3)
    - 评估时间稳定性 (Stable, Variable, Shift)
5.  **Jaramillo轨迹分析 (2022)**:
    - 比较 T1 和 T3 的均值点
    - 计算运动轨迹 (方向, 强度)
    - 判断是否“遵循曲线” (Follows Curve)
6.  **He et al. 3D框架分析 (2023)**:
    - 绘制 DI vs IE_ext (扩展Budyko空间)
    - 绘制 DI vs SCI (检验线性关系)
7.  **偏差归因分析**:
    - 将模拟的“人类活动”（导致Q偏离的因素）作为驱动因子X
    - 将Ibrahim偏差 (ε) 作为因变量Y
    - 使用随机森林“发现”导致偏离的原因
8.  **积雪影响分析** (可选步骤):
    - 演示如何使用 SnowImpactAnalyzer
9.  **综合可视化**:
    - 生成多面板图表总结所有发现
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple

# 确保可以导入 <repo_root>/src 下的包
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 使用包名导入（而不是 src. 前缀）
from budyko.water_balance import WaterBalanceCalculator, WaterBalanceResults
from budyko.curves import BudykoCurves
from budyko.deviation import DeviationAnalysis, TemporalStability, MarginalDistribution
from budyko.trajectory_jaramillo import TrajectoryAnalyzer
from models.pet_lai_co2 import PETWithLAICO2
from models.pet_models import PenmanMonteith
from analysis.deviation_attribution import DeviationAttributor
from analysis.snow_analyzer import SnowImpactAnalyzer
from visualization.budyko_plots import BudykoVisualizer
from visualization.direction_rose import plot_direction_rose

# --- 1. 数据模拟 ---
def generate_complex_data(n_years: int = 60, start_year: int = 1961) -> pd.DataFrame:
    """生成一个复杂的、有趋势的模拟流域数据集"""
    print(f"[1. 数据模拟] 正在生成 {n_years} 年的模拟数据 (1961-2020)...")
    
    np.random.seed(42)
    years = np.arange(start_year, start_year + n_years)
    time_idx = np.arange(n_years)
    
    # 1. 气候变量
    P = 900 + 50 * np.sin(2 * np.pi * time_idx / 30) + 150 * np.random.randn(n_years)
    P = np.maximum(P, 400)
    T_avg = 15 + 0.03 * time_idx + 0.5 * np.random.randn(n_years) # 增温趋势
    T_max = T_avg + 6
    T_min = T_avg - 6
    RH = 60 - 0.1 * time_idx + 5 * np.random.randn(n_years) # 变干趋势
    u2 = 2.0 + 0.3 * np.random.randn(n_years)
    Rn_MJ = 7.0 + 0.5 * np.sin(2 * np.pi * time_idx / 20) + 0.5 * np.random.randn(n_years) # MJ/m²/day
    Rn_W = Rn_MJ / 0.0864 # W/m²
    
    # 2. 创新驱动因子 (LAI, CO2)
    LAI = 2.5 + 0.02 * time_idx + 0.3 * np.random.randn(n_years) # 绿化趋势
    LAI = np.clip(LAI, 1.0, 6.0)
    CO2 = 320 + 1.5 * time_idx + 2 * np.random.randn(n_years) # CO2上升趋势
    
    # 3. 储量 (TWS)
    TWS = 50 * np.sin(2 * np.pi * time_idx / 15) - 0.5 * time_idx # 轻微下降趋势
    delta_S = np.diff(TWS, prepend=TWS[0])
    
    # 4. 模拟“人类活动” (用于归因)
    # 假设灌溉取水逐年增加
    irrigation_withdrawal = 10 + 1.2 * time_idx + 5 * np.random.randn(n_years)
    irrigation_withdrawal = np.maximum(irrigation_withdrawal, 0)

    # 5. 模拟径流 (Q)
    # 5a. 首先，计算“自然”径流 (Q_nat)
    # 使用LAI+CO2 PET计算IA
    pet_calc = PETWithLAICO2(elevation=500, latitude=40)
    # 注意顺序：temperature, humidity, wind_speed, radiation, lai, co2
    PET_lai_co2 = pet_calc.calculate(T_avg, RH, u2, Rn_W, LAI, CO2)
    PET_lai_co2_annual = PET_lai_co2 * 365.25 # 假设是日均值
    
    IA_nat = PET_lai_co2_annual / P
    # 假设该流域的“自然”omega = 2.5
    omega_nat = 2.5
    IE_nat = BudykoCurves.tixeront_fu(IA_nat, omega_nat)
    EA_nat = IE_nat * P
    Q_nat = P - EA_nat # 自然径流
    
    # 5b. 计算“观测”径流 (Q_obs)
    # 观测径流 = 自然径流 - 灌溉取水 (并考虑储量变化)
    Q_obs = P - EA_nat - delta_S - irrigation_withdrawal
    Q_obs = np.maximum(Q_obs, Q_nat * 0.1) # 确保径流不为负
    
    # 创建DataFrame
    data = {
        'year': years,
        'P': P,
        'Q_obs': Q_obs,
        'Q_nat': Q_nat,
        'T_avg': T_avg,
        'T_max': T_max,
        'T_min': T_min,
        'RH': RH,
        'u2': u2,
        'Rn_MJ': Rn_MJ,
        'Rn_W': Rn_W,
        'LAI': LAI,
        'CO2': CO2,
        'TWS': TWS,
        'delta_S': delta_S,
        'irrigation_withdrawal': irrigation_withdrawal,
        'PET_lai_co2': PET_lai_co2_annual
    }
    df = pd.DataFrame(data)
    print(f"  模拟数据生成完毕。 径流 Q_obs 已被人为降低 (模拟灌溉)。")
    return df

# --- 2. PET计算对比 ---
def step2_pet_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """计算并对比两种PET方法"""
    print("\n[2. PET计算对比]")
    
    # 1. 创新的 LAI+CO2 PET (已在模拟中计算)
    pet_lai_co2 = df['PET_lai_co2'].values
    
    # 2. 标准 Penman-Monteith (FAO56)
    pet_standard_calc = PenmanMonteith()
    # 模拟数据输入 (注意单位转换)
    pet_standard = pet_standard_calc.calculate(
        temp_avg=df['T_avg'].to_numpy(),
        temp_max=df['T_max'].to_numpy(),
        temp_min=df['T_min'].to_numpy(),
        rh_mean=df['RH'].to_numpy(),
        wind_speed=df['u2'].to_numpy(),
        solar_radiation=df['Rn_MJ'].to_numpy(), # PM需要 MJ/m²/day
        latitude=40.0,
        elevation=500.0,
        day_of_year=np.tile(np.arange(1, 366), len(df)//365 + 1)[:len(df)] # 简化
    )
    # 从日均转为年总量
    pet_standard_annual = pd.Series(pet_standard).rolling(window=365, min_periods=365).sum().to_numpy()
    mean_ps = float(np.nanmean(pet_standard_annual))
    if not np.isfinite(mean_ps):
        mean_ps = float(np.nanmean(pet_standard)) if np.isfinite(np.nanmean(pet_standard)) else 0.0
    pet_standard_annual = np.where(np.isnan(pet_standard_annual), mean_ps, pet_standard_annual)
    
    # (在我们的模拟中, PET_lai_co2已经是年值, 为公平比较, 我们重新日算)
    pet_calc_lai_co2 = PETWithLAICO2(elevation=500, latitude=40)
    pet_lai_co2_daily = pet_calc_lai_co2.calculate(
        temperature=df['T_avg'].values,  # 假设是日均
        humidity=df['RH'].values,
        wind_speed=df['u2'].values,
        radiation=df['Rn_W'].values,     # W/m²
        lai=df['LAI'].values,
        co2=df['CO2'].values
    )
    pet_lai_co2_annual = pd.Series(pet_lai_co2_daily).rolling(window=365, min_periods=365).sum().to_numpy()
    mean_plc = float(np.nanmean(pet_lai_co2_annual))
    if not np.isfinite(mean_plc):
        mean_plc = float(np.nanmean(pet_lai_co2_daily)) if np.isfinite(np.nanmean(pet_lai_co2_daily)) else 0.0
    pet_lai_co2_annual = np.where(np.isnan(pet_lai_co2_annual), mean_plc, pet_lai_co2_annual)

    df['PET_standard'] = pet_standard_annual
    df['PET_lai_co2'] = pet_lai_co2_annual # 更新为日转年
    
    print(f"  标准 PET (PM-FAO56): {np.mean(pet_standard_annual):.1f} mm/yr")
    print(f"  创新 PET (LAI+CO2):  {np.mean(pet_lai_co2_annual):.1f} mm/yr")
    print(f"  差异 (创新 - 标准):   {np.mean(pet_lai_co2_annual - pet_standard_annual):.1f} mm/yr")
    print("  (差异原因: CO2上升抑制蒸发 > LAI上升促进蒸发)")
    
    return df

# --- 3. 水量平衡计算 ---
def step3_water_balance(df: pd.DataFrame) -> Tuple[pd.DataFrame, WaterBalanceResults]:
    """计算2D和3D Budyko指数 (基于观测径流Q_obs)"""
    print("\n[3. 水量平衡计算 (基于 Q_obs)]")
    
    wb_calc = WaterBalanceCalculator()
    
    # 使用创新的PET (PET_lai_co2) 作为能量项
    # 使用观测的径流 (Q_obs) 作为水量项
    wb_results = wb_calc.calculate_budyko_indices(
        P=df['P'].to_numpy(),
        Q=df['Q_obs'].to_numpy(),
        PET=df['PET_lai_co2'].to_numpy(),
        delta_S=df['delta_S'].to_numpy(), # 提供ΔS用于3D分析
        TWS=df['TWS'].to_numpy()          # 提供TWS用于3D分析
    )
    
    print(f"  2D (P-Q):     IA={np.nanmean(wb_results.aridity_index):.3f}, IE={np.nanmean(wb_results.evaporation_index):.3f}")
    ie_ext_mean = float(np.nanmean(wb_results.evaporation_index_extended)) if wb_results.evaporation_index_extended is not None else float('nan')
    sci_mean = float(np.nanmean(wb_results.storage_change_index)) if wb_results.storage_change_index is not None else float('nan')
    print(f"  3D (P-Q-ΔS):  IE_ext={ie_ext_mean:.3f}, SCI={sci_mean:.3f}")
    
    # 将结果添加回df
    df['IA'] = wb_results.aridity_index
    df['IE'] = wb_results.evaporation_index
    df['EA'] = wb_results.actual_evaporation
    df['IE_ext'] = wb_results.evaporation_index_extended
    df['SCI'] = wb_results.storage_change_index
    
    return df, wb_results

# --- 4. Ibrahim 偏差分析 ---
def step4_ibrahim_analysis(df: pd.DataFrame) -> Dict:
    """执行Ibrahim (2025) 偏差分析"""
    print("\n[4. Ibrahim 偏差分析]")
    
    deviation_analyzer = DeviationAnalysis(period_length=20)
    stability_analyzer = TemporalStability()
    marginal_analyzer = MarginalDistribution()
    
    periods_data = {}
    deviation_dists = {}
    
    # T1 (1961-1980), T2 (1981-2000), T3 (2001-2020)
    period_defs = [('T1', 1961, 1980), ('T2', 1981, 2000), ('T3', 2001, 2020)]
    
    # Step 1: 拟合各时段ω
    for name, start, end in period_defs:
        period_df = df[(df['year'] >= start) & (df['year'] <= end)]
        omega, stats = BudykoCurves.fit_omega(period_df['IA'].to_numpy(), period_df['IE'].to_numpy())
        periods_data[name] = {
            'omega': omega,
            'ia_annual': period_df['IA'].to_numpy(),
            'ie_annual': period_df['IE'].to_numpy(),
            'ia_mean': period_df['IA'].mean(),
            'ie_mean': period_df['IE'].mean(),
            'start_year': start,
            'end_year': end
        }
        print(f"  {name} ({start}-{end}): ω={omega:.3f}, IE_mean={periods_data[name]['ie_mean']:.3f}")
        
    # Step 2: 计算偏差分布
    # 偏差 T2 (基于 T1的ω)
    dist_T2 = deviation_analyzer.calculate_deviations(
        ia_i=periods_data['T1']['ia_annual'],
        ie_obs_i=periods_data['T1']['ie_annual'],
        omega_i=periods_data['T1']['omega'],
        ia_i_plus_1=periods_data['T2']['ia_annual'],
        ie_obs_i_plus_1=periods_data['T2']['ie_annual'],
        period_pair='T1-T2'
    )
    wilcoxon_T2 = deviation_analyzer.wilcoxon_test(dist_T2)
    deviation_dists['T1-T2'] = {'distribution': dist_T2, 'wilcoxon_test': wilcoxon_T2}
    print(f"  偏差 ε(T2|ω1): Median={dist_T2.median:.4f} (p={wilcoxon_T2['p_value']:.3f})")

    # 偏差 T3 (基于 T2的ω)
    dist_T3 = deviation_analyzer.calculate_deviations(
        ia_i=periods_data['T2']['ia_annual'],
        ie_obs_i=periods_data['T2']['ie_annual'],
        omega_i=periods_data['T2']['omega'],
        ia_i_plus_1=periods_data['T3']['ia_annual'],
        ie_obs_i_plus_1=periods_data['T3']['ie_annual'],
        period_pair='T2-T3'
    )
    wilcoxon_T3 = deviation_analyzer.wilcoxon_test(dist_T3)
    deviation_dists['T2-T3'] = {'distribution': dist_T3, 'wilcoxon_test': wilcoxon_T3}
    print(f"  偏差 ε(T3|ω2): Median={dist_T3.median:.4f} (p={wilcoxon_T3['p_value']:.3f})")
    
    # Step 4: 评估稳定性
    stability = stability_analyzer.assess_stability(
        distributions=[dist_T2, dist_T3],
        ie_means=[periods_data['T2']['ie_mean'], periods_data['T3']['ie_mean']]
    )
    print(f"  时间稳定性: {stability['category']}")
    
    # Step 5: 聚合边际分布
    marginal = marginal_analyzer.aggregate_distributions(
        [dist_T2, dist_T3], stability['category']
    )
    print(f"  边际分布: Median={marginal['median']:.4f}, IQR={marginal['iqr']:.4f}")
    
    return {'periods': periods_data, 'stability': stability, 'marginal': marginal, 'deviations': deviation_dists}

# --- 5. Jaramillo 轨迹分析 ---
def step5_jaramillo_analysis(ibrahim_results: Dict) -> Dict:
    """执行Jaramillo (2022) 轨迹分析"""
    print("\n[5. Jaramillo 轨迹分析 (T1 vs T3)]")
    
    trajectory_analyzer = TrajectoryAnalyzer()
    
    p1 = ibrahim_results['periods']['T1']
    p3 = ibrahim_results['periods']['T3']

    movement = trajectory_analyzer.calculate_movement(
        catchment_id='complex_basin',
        period_1={'IA': p1['ia_mean'], 'IE': p1['ie_mean'], 'name': 'T1'},
        period_2={'IA': p3['ia_mean'], 'IE': p3['ie_mean'], 'name': 'T3'}
    )
    
    print(f"  起点 (T1): IA={movement.ia_t1:.3f}, IE={movement.ie_t1:.3f}")
    print(f"  终点 (T3): IA={movement.ia_t2:.3f}, IE={movement.ie_t2:.3f}")
    print(f"  运动向量: (ΔIA={movement.delta_ia:.3f}, ΔIE={movement.delta_ie:.3f})")
    print(f"  运动强度: {movement.intensity:.4f}")
    print(f"  方向角: {movement.direction_angle:.1f}°")
    print(f"  是否遵循曲线: {movement.follows_curve}")
    print(f"  (分析: 我们的模拟Q被人为降低, IE上升, ΔIE>0。这导致偏离曲线)")
    
    return {'movement': movement}

# --- 6. 偏差归因分析 ---
def step6_attribution_analysis(df: pd.DataFrame, ibrahim_results: Dict) -> Dict:
    """执行偏差归因分析"""
    print("\n[6. 偏差归因 (发现模拟的灌溉影响)]")
    
    attributor = DeviationAttributor()
    
    # 因变量 Y: Ibrahim偏差 (年度)
    # 我们用 T3 时段的偏差 (相对于 T2的ω)
    y_deviation = ibrahim_results['deviations']['T2-T3']['distribution'].annual_deviations
    
    # 自变量 X: T3 时段的驱动因子
    df_T3 = df[df['year'].isin(range(2001, 2021))]
    
    attributor.set_deviation(y_deviation)
    attributor.add_driver('irrigation', df_T3['irrigation_withdrawal'].to_numpy())
    attributor.add_driver('lai', df_T3['LAI'].to_numpy())
    attributor.add_driver('co2', df_T3['CO2'].to_numpy())
    attributor.add_driver('T_avg', df_T3['T_avg'].to_numpy())
    
    # 运行随机森林归因
    rf_results = attributor.random_forest_attribution()
    
    print(f"  归因模型 R²: {rf_results.explained_variance:.2%}")
    print("  驱动因子重要性:")
    imp = rf_results.driver_importance
    for driver, importance in sorted(imp.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {driver:15s}: {importance:.2%}")
        
    print(f"  (分析: 'irrigation' 应该是最重要的, 因为它是我们模拟偏差的来源)")
    return rf_results

# --- 7. 积雪影响分析 (演示) ---
def step7_snow_analysis_demo(df: pd.DataFrame):
    """演示积雪分析模块"""
    print("\n[7. 积雪影响分析 (演示)]")
    
    snow_analyzer = SnowImpactAnalyzer(T_threshold=1.0, melt_factor=2.0)
    
    # 模拟日度数据 (简化)
    daily_P = np.repeat(df['P'].to_numpy(), 365) / 365
    daily_T = np.repeat(df['T_avg'].to_numpy(), 365) # 粗略
    
    snow_results = snow_analyzer.calculate_snowfall_and_snowmelt(daily_P, daily_T)
    
    # 聚合到年度
    snow_df = pd.DataFrame(snow_results)
    snow_df['year'] = np.repeat(df['year'].to_numpy(), 365)
    annual_snow = snow_df.groupby('year').sum()
    
    snow_ratio = annual_snow['snowfall'].sum() / (annual_snow['snowfall'].sum() + annual_snow['rainfall'].sum())
    print(f"  模拟流域总降雪比 (Snow Ratio): {snow_ratio:.2%}")
    
    if snow_ratio > 0.1:
        print("  流域受积雪影响，可进一步分析修正后的Budyko指数。")
    else:
        print("  流域不受积雪主导，跳过积雪修正。")
        
    return snow_ratio

# --- 8. 综合可视化 ---
def step8_visualization(df: pd.DataFrame, ibrahim: Dict, jaramillo: Dict, attribution):
    """生成综合图表"""
    print("\n[8. 综合可视化]")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)
    
    # --- 图 A: 核心Budyko空间 (P-Q) ---
    ax_a = fig.add_subplot(gs[0, 0])
    sns.scatterplot(data=df, x='IA', y='IE', hue='year', palette='viridis', s=20, ax=ax_a)
    ia_range = np.linspace(0, 5, 200)
    for om in [2.0, 2.5, 3.0]:
        ax_a.plot(ia_range, BudykoCurves.tixeront_fu(ia_range, om), linestyle='--', label=f"ω={om}", alpha=0.6)
    ax_a.set_title("A: Budyko 空间 (IE = (P-Q)/P)", fontweight='bold')
    ax_a.set_xlabel("干旱指数 IA = PET/P")
    ax_a.set_ylabel("蒸发指数 IE = EA/P")
    ax_a.legend().remove()
    
    # --- 图 B: Ibrahim 偏差分布 ---
    ax_b = fig.add_subplot(gs[0, 1])
    dev_T2 = ibrahim['deviations']['T1-T2']['distribution'].annual_deviations
    dev_T3 = ibrahim['deviations']['T2-T3']['distribution'].annual_deviations
    sns.histplot(dev_T2, kde=True, ax=ax_b, color='blue', label=f"ε(T2|ω1) (Med={np.median(dev_T2):.3f})")
    sns.histplot(dev_T3, kde=True, ax=ax_b, color='red', label=f"ε(T3|ω2) (Med={np.median(dev_T3):.3f})")
    ax_b.axvline(0, color='k', linestyle='--')
    ax_b.set_title(f"B: Ibrahim 偏差分布 (稳定性: {ibrahim['stability']['category']})", fontweight='bold')
    ax_b.set_xlabel("偏差 ε_IEω")
    ax_b.legend()
    
    # --- 图 C: Jaramillo 轨迹 ---
    ax_c = fig.add_subplot(gs[0, 2])
    mov = jaramillo['movement']
    ax_c.arrow(mov.ia_t1, mov.ie_t1, mov.delta_ia, mov.delta_ie, 
             head_width=0.01, length_includes_head=True, fc='r', ec='r')
    ax_c.plot(mov.ia_t1, mov.ie_t1, 'bo', label=f"T1 ({ibrahim['periods']['T1']['start_year']}-{ibrahim['periods']['T1']['end_year']})")
    ax_c.plot(mov.ia_t2, mov.ie_t2, 'rs', label=f"T3 ({ibrahim['periods']['T3']['start_year']}-{ibrahim['periods']['T3']['end_year']})")
    ia_range = np.linspace(0, 5, 200)
    for om in [p['omega'] for p in ibrahim['periods'].values()]:
        ax_c.plot(ia_range, BudykoCurves.tixeront_fu(ia_range, om), linestyle=':', color='gray', alpha=0.7)
    ax_c.set_title(f"C: Jaramillo 轨迹 (偏离: {not mov.follows_curve})", fontweight='bold')
    ax_c.set_xlabel("IA (时段均值)")
    ax_c.set_ylabel("IE (时段均值)")
    ax_c.legend()

    # --- 图 D: He et al. 3D 框架 (P-Q-ΔS) ---
    ax_d = fig.add_subplot(gs[1, 0])
    sns.scatterplot(data=df, x='IA', y='IE_ext', hue='SCI', palette='coolwarm', s=20, ax=ax_d)
    ax_d.set_title("D: 3D Budyko 空间 (IE_ext = (P-Q-ΔS)/P)", fontweight='bold')
    ax_d.set_xlabel("干旱指数 IA = PET/P")
    ax_d.set_ylabel("扩展蒸发指数 IE_ext")
    
    # --- 图 E: He et al. 3D 关系 (DI vs SCI) ---
    ax_e = fig.add_subplot(gs[1, 1])
    sns.regplot(data=df, x='IA', y='SCI', ax=ax_e, scatter_kws={'s': 10, 'alpha': 0.5}, line_kws={'color': 'red'})
    ax_e.set_title("E: 3D 关系 (DI vs SCI)", fontweight='bold')
    ax_e.set_xlabel("干旱指数 IA")
    ax_e.set_ylabel("储量变化指数 SCI = ΔS/P")
    
    # --- 图 F: 偏差归因 ---
    ax_f = fig.add_subplot(gs[1, 2])
    imp = attribution.driver_importance
    drivers = sorted(imp.keys(), key=lambda k: imp[k], reverse=True)
    importances = [imp[k] for k in drivers]
    sns.barplot(x=importances, y=drivers, ax=ax_f, orient='h')
    ax_f.set_title("F: 偏差归因 (随机森林)", fontweight='bold')
    ax_f.set_xlabel("特征重要性")
    
    # --- 图 G: PET 对比 ---
    ax_g = fig.add_subplot(gs[2, 0])
    sns.kdeplot(x=df['PET_standard'], ax=ax_g, label='标准 PM-FAO56', color='gray')
    sns.kdeplot(x=df['PET_lai_co2'], ax=ax_g, label='创新 PM (LAI+CO2)', color='green')
    ax_g.set_title("G: PET 方法对比", fontweight='bold')
    ax_g.set_xlabel("PET (mm/yr)")
    ax_g.legend()
    
    # --- 图 H: 驱动因子趋势 ---
    ax_h = fig.add_subplot(gs[2, 1])
    ax_h2 = ax_h.twinx()
    sns.regplot(data=df, x='year', y='LAI', ax=ax_h, color='g', label='LAI', scatter_kws={'s': 5})
    sns.regplot(data=df, x='year', y='CO2', ax=ax_h2, color='k', label='CO2', scatter_kws={'s': 5})
    ax_h.set_ylabel("LAI", color='g')
    ax_h2.set_ylabel("CO2 (ppm)", color='k')
    ax_h.set_title("H: 植被与CO2趋势", fontweight='bold')
    
    # --- 图 I: 水量平衡趋势 ---
    ax_i = fig.add_subplot(gs[2, 2])
    sns.regplot(data=df, x='year', y='P', ax=ax_i, label='P (降水)', scatter_kws={'s': 5})
    sns.regplot(data=df, x='year', y='Q_obs', ax=ax_i, label='Q_obs (观测径流)', scatter_kws={'s': 5})
    sns.regplot(data=df, x='year', y='EA', ax=ax_i, label='EA (P-Q_obs)', scatter_kws={'s': 5})
    sns.regplot(data=df, x='year', y='irrigation_withdrawal', ax=ax_i, label='模拟灌溉', scatter_kws={'s': 5}, line_kws={'linestyle': ':', 'color': 'red'})
    ax_i.set_title("I: 水量平衡趋势 (Q下降, EA上升)", fontweight='bold')
    ax_i.set_ylabel("水量 (mm/yr)")
    ax_i.legend()

    plt.tight_layout(pad=1.5)
    # 统一输出到脚本同级目录下的 figures 文件夹，文件名包含脚本名
    figures_dir = Path(__file__).resolve().parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    script_stem = Path(__file__).stem
    save_path = figures_dir / f"{script_stem}__summary.png"
    plt.savefig(save_path, dpi=300)
    print(f"  综合图表已保存到: {save_path}")
    plt.close()

# --- 主函数 ---
def main():
    """执行完整算例"""
    
    # 1. 模拟数据
    df = generate_complex_data()
    
    # 2. PET对比
    df = step2_pet_comparison(df)
    
    # 3. 水量平衡
    df, wb_results = step3_water_balance(df)
    
    # 4. Ibrahim 分析
    ibrahim_results = step4_ibrahim_analysis(df)
    
    # 5. Jaramillo 分析
    jaramillo_results = step5_jaramillo_analysis(ibrahim_results)
    
    # 6. 归因分析
    attribution_results = step6_attribution_analysis(df, ibrahim_results)
    
    # 7. 积雪分析 (演示)
    _ = step7_snow_analysis_demo(df)
    
    # 8. 可视化
    step8_visualization(df, ibrahim_results, jaramillo_results, attribution_results)
    
    print("\n[--- 复杂算例执行完毕 ---]")

if __name__ == "__main__":
    main()