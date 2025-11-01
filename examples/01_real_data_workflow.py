# examples/01_real_data_workflow.py
"""
完整的真实世界Budyko分析工作流程

**目标**：
展示如何使用本代码库处理真实的、多源的科学数据，
完成从数据加载到综合分析的全流程Budyko分析。

**数据来源**（模拟）：
1. 站点径流数据（CSV） - Q，水量平衡的锚点
2. 格点气象数据（NetCDF） - P, T, RH, 风速, 辐射
3. 遥感LAI数据（NetCDF） - MODIS LAI
4. 遥感TWS数据（NetCDF） - GRACE TWS
5. CO2浓度数据 - Mauna Loa观测
6. 人为驱动因子（CSV） - 灌溉、土地利用等

**分析范式**：
1. Ibrahim (2025) 偏差分析 - 时间稳定性
2. Jaramillo (2022) 轨迹分析 - 空间运动方向
3. He (2023) 3D框架 - ΔS的作用

**核心创新**：
对比标准PET（Penman-Monteith）与创新PET（考虑LAI和CO2）的差异
"""

import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# 假设当前在examples/目录，需要将src/加入路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# ========================
# 导入本代码库的模块
# ========================
from data_processing.basin_processor import BasinDataProcessor
from data_processing.grace_lai_processor import GRACEDataLoader, LAIDataLoader, CO2DataLoader

from models.pet_models import PenmanMonteithPET
from models.pet_lai_co2 import PETWithLAICO2, PETComparator

from budyko.water_balance import WaterBalanceCalculator
from budyko.curves import BudykoCurves
from budyko.deviation import DeviationAnalysis, TemporalStability, MarginalDistribution
from budyko.trajectory_jaramillo import TrajectoryAnalyzer, MovementStatistics

from analysis.deviation_attribution import DeviationAttributor

from visualization.budyko_plots import BudykoVisualizer, create_comprehensive_figure


def main():
    """主工作流程"""

    print("="*80)
    print("完整Budyko分析工作流程")
    print("="*80)
    print()

    # =========================================================================
    # 第一部分：数据加载与预处理
    # =========================================================================
    print("【第一部分】数据加载与预处理")
    print("-"*80)

    # --- 1.1 定义数据路径（实际使用时替换为真实路径）---
    DATA_DIR = Path('data')  # 假设的数据目录

    runoff_file = DATA_DIR / 'runoff' / 'station_Q.csv'
    basin_shapefile = DATA_DIR / 'shapefiles' / 'my_basin.shp'
    cmfd_dir = DATA_DIR / 'met' / 'CMFD'
    lai_dir = DATA_DIR / 'sat' / 'MODIS_LAI'
    grace_dir = DATA_DIR / 'grace' / 'TWS'
    driver_file = DATA_DIR / 'human' / 'drivers.csv'

    print(f"数据目录: {DATA_DIR}")
    print(f"注意：本示例使用模拟数据进行演示\n")

    # --- 1.2 生成模拟数据（演示用）---
    print("生成模拟数据...")
    years = np.arange(1980, 2021)  # 1980-2020，共41年
    n_years = len(years)
    basin_id = 'Basin_001'

    # 模拟年度数据
    np.random.seed(42)

    # 降水P：500-1500mm，有年际变率
    P_mean = 800
    P = P_mean + 200 * np.sin(2 * np.pi * years / 30) + np.random.normal(0, 100, n_years)
    P = np.maximum(P, 300)  # 确保非负

    # 径流Q：基于P，径流系数0.3-0.5
    RC_base = 0.4
    Q = P * (RC_base + 0.1 * np.sin(2 * np.pi * years / 20) + np.random.normal(0, 0.05, n_years))
    Q = np.clip(Q, 50, P * 0.9)  # 确保 Q < P

    # 温度T：10-20°C，有升温趋势
    T = 15 + 0.03 * (years - 1980) + 3 * np.sin(2 * np.pi * (np.arange(n_years) % 12) / 12) + np.random.normal(0, 1, n_years)

    # 相对湿度RH：60-80%
    RH = 70 + np.random.normal(0, 5, n_years)
    RH = np.clip(RH, 40, 95)

    # 风速u2：1-3 m/s
    u2 = 2 + np.random.normal(0, 0.5, n_years)
    u2 = np.maximum(u2, 0.5)

    # 辐射Rn：100-200 W/m²
    Rn = 150 + 30 * np.sin(2 * np.pi * (np.arange(n_years) % 12) / 12) + np.random.normal(0, 10, n_years)
    Rn = np.maximum(Rn, 50)

    # LAI：1-5 m²/m²，有增长趋势（绿化）
    LAI = 2.5 + 0.5 * (years - 1980) / 40 + 0.5 * np.sin(2 * np.pi * (np.arange(n_years) % 12) / 12) + np.random.normal(0, 0.3, n_years)
    LAI = np.clip(LAI, 0.5, 6)

    # CO2：从340ppm（1980）增长到415ppm（2020）
    CO2 = 340 + 1.9 * (years - 1980) + np.random.normal(0, 2, n_years)

    # TWS：陆地水储量变化（GRACE, 2002-2020）
    # 早期数据用插值填充
    TWS = np.zeros(n_years)
    TWS[years >= 2002] = 50 * np.sin(2 * np.pi * (years[years >= 2002] - 2002) / 10) + np.random.normal(0, 20, sum(years >= 2002))
    TWS[years < 2002] = np.nan  # GRACE之前无数据

    # ΔS：储量变化
    delta_S = np.diff(TWS, prepend=np.nan)

    # 人为驱动因子：灌溉面积比例（模拟）
    irrigation_ratio = 0.1 + 0.2 * (years - 1980) / 40 + np.random.normal(0, 0.02, n_years)
    irrigation_ratio = np.clip(irrigation_ratio, 0, 0.5)

    # 构建DataFrame
    data = pd.DataFrame({
        'year': years,
        'basin_id': basin_id,
        'P': P,
        'Q': Q,
        'T': T,
        'RH': RH,
        'u2': u2,
        'Rn': Rn,
        'LAI': LAI,
        'CO2': CO2,
        'TWS': TWS,
        'delta_S': delta_S,
        'irrigation_ratio': irrigation_ratio
    })

    print(f"✓ 模拟数据生成完成：{n_years}年，流域ID={basin_id}\n")

    # =========================================================================
    # 第二部分：PET计算与对比
    # =========================================================================
    print("\n【第二部分】PET计算与对比")
    print("-"*80)

    # --- 2.1 标准PET（Penman-Monteith）---
    print("计算标准PET（Penman-Monteith FAO-56）...")
    pm_model = PenmanMonteithPET()

    # 简化：用年均值代表，实际应用应使用日值聚合
    PET_standard = []
    for idx, row in data.iterrows():
        pet = pm_model.calculate(
            temp_avg=row['T'],
            temp_max=row['T'] + 5,  # 简化
            temp_min=row['T'] - 5,
            rh_mean=row['RH'],
            wind_speed=row['u2'],
            solar_radiation=row['Rn'] * 0.0864,  # W/m² -> MJ/m²/day
            latitude=35.0,  # 假设纬度
            elevation=500.0,  # 假设海拔
            day_of_year=180  # 假设年中
        )
        PET_standard.append(pet * 365)  # mm/day -> mm/year

    data['PET_standard'] = PET_standard
    print(f"✓ 标准PET计算完成")
    print(f"  平均值: {np.mean(PET_standard):.1f} mm/year")

    # --- 2.2 创新PET（考虑LAI和CO2）---
    print("\n计算创新PET（考虑LAI与CO2动态影响）...")
    pet_lai_co2 = PETWithLAICO2(elevation=500.0, latitude=35.0)

    PET_innovation = []
    for idx, row in data.iterrows():
        pet = pet_lai_co2.calculate(
            temperature=row['T'],
            humidity=row['RH'],
            wind_speed=row['u2'],
            radiation=row['Rn'],
            lai=row['LAI'],
            co2=row['CO2']
        )
        PET_innovation.append(pet * 365)  # mm/day -> mm/year

    data['PET_innovation'] = PET_innovation
    print(f"✓ 创新PET计算完成")
    print(f"  平均值: {np.mean(PET_innovation):.1f} mm/year")

    # --- 2.3 PET对比 ---
    print("\nPET方法对比:")
    comparison = PETComparator.compare(data['PET_standard'], data['PET_innovation'])
    print(f"  标准PET均值: {comparison['mean_A']:.1f} mm/year")
    print(f"  创新PET均值: {comparison['mean_B']:.1f} mm/year")
    print(f"  平均差异: {comparison['mean_diff']:.1f} mm/year")
    print(f"  相关系数: {comparison['corr']:.3f}")
    print(f"  RMSE: {comparison['rmse']:.1f} mm/year")

    # =========================================================================
    # 第三部分：水量平衡与Budyko指数计算
    # =========================================================================
    print("\n【第三部分】水量平衡与Budyko指数计算")
    print("-"*80)

    # --- 3.1 使用创新PET计算Budyko指数 ---
    print("计算Budyko指数（基于观测径流Q和创新PET）...")

    wb_calc = WaterBalanceCalculator(
        allow_negative_ea=False,
        min_precipitation=200.0,
        max_runoff_ratio=0.95
    )

    # 2D指数（无ΔS）
    results_2d = wb_calc.calculate_budyko_indices(
        P=data['P'].values,
        Q=data['Q'].values,
        PET=data['PET_innovation'].values
    )

    data['EA'] = results_2d.actual_evaporation
    data['IA'] = results_2d.aridity_index
    data['IE'] = results_2d.evaporation_index
    data['RC'] = results_2d.runoff_coefficient

    print(f"✓ 2D Budyko指数计算完成")
    print(f"  IA（干旱指数）范围: {data['IA'].min():.2f} - {data['IA'].max():.2f}")
    print(f"  IE（蒸发指数）范围: {data['IE'].min():.2f} - {data['IE'].max():.2f}")
    print(f"  平均径流系数RC: {data['RC'].mean():.2f}")

    # 3D指数（考虑ΔS，仅GRACE时期）
    mask_grace = data['year'] >= 2002
    if mask_grace.sum() > 0:
        print("\n计算3D Budyko指数（考虑ΔS，2002-2020）...")

        results_3d = wb_calc.calculate_budyko_indices(
            P=data.loc[mask_grace, 'P'].values,
            Q=data.loc[mask_grace, 'Q'].values,
            PET=data.loc[mask_grace, 'PET_innovation'].values,
            delta_S=data.loc[mask_grace, 'delta_S'].values,
            TWS=data.loc[mask_grace, 'TWS'].values
        )

        data.loc[mask_grace, 'EA_ext'] = results_3d.actual_evaporation_extended
        data.loc[mask_grace, 'SCI'] = results_3d.storage_change_index

        print(f"✓ 3D指数计算完成")
        print(f"  SCI（储量变化指数）范围: {data.loc[mask_grace, 'SCI'].min():.3f} - {data.loc[mask_grace, 'SCI'].max():.3f}")

    # =========================================================================
    # 第四部分：Ibrahim (2025) 偏差分析
    # =========================================================================
    print("\n【第四部分】Ibrahim (2025) 偏差分析")
    print("-"*80)

    # --- 4.1 划分时段（3个20年时段）---
    print("划分时段...")
    period_length = 14  # 由于只有41年，用14年时段

    periods = []
    for i, start_year in enumerate([1980, 1994, 2007]):
        end_year = start_year + period_length - 1
        if end_year > 2020:
            end_year = 2020

        mask = (data['year'] >= start_year) & (data['year'] <= end_year)
        period_data = data[mask].copy()

        # 计算该时段的ω参数
        omega, fit_result = BudykoCurves.fit_omega(
            period_data['IA'].values,
            period_data['IE'].values
        )

        periods.append({
            'name': f'T{i+1}',
            'start_year': start_year,
            'end_year': end_year,
            'data': period_data,
            'omega': omega,
            'IA': period_data['IA'].values,
            'IE': period_data['IE'].values,
            'IA_mean': period_data['IA'].mean(),
            'IE_mean': period_data['IE'].mean(),
        })

        print(f"  时段{i+1} ({start_year}-{end_year}): ω={omega:.3f}, R²={fit_result['r2']:.3f}, n={len(period_data)}")

    # --- 4.2 计算时段间偏差分布 ---
    print("\n计算偏差分布...")
    dev_analyzer = DeviationAnalysis(period_length=period_length)

    deviation_results = {}
    for i in range(len(periods) - 1):
        p_i = periods[i]
        p_i_plus_1 = periods[i+1]

        pair_name = f'Δ{i+1}-{i+2}'

        dist = dev_analyzer.calculate_deviations(
            ia_i=p_i['IA'],
            ie_obs_i=p_i['IE'],
            omega_i=p_i['omega'],
            ia_i_plus_1=p_i_plus_1['IA'],
            ie_obs_i_plus_1=p_i_plus_1['IE'],
            period_pair=pair_name
        )

        # Wilcoxon检验
        wilcoxon_result = dev_analyzer.wilcoxon_test(dist)

        deviation_results[pair_name] = {
            'distribution': dist,
            'wilcoxon_test': wilcoxon_result
        }

        sig = "***" if wilcoxon_result['p_value'] < 0.001 else \
              "**" if wilcoxon_result['p_value'] < 0.01 else \
              "*" if wilcoxon_result['p_value'] < 0.05 else "ns"

        print(f"  {pair_name}: 中位数={dist.median:.4f}, p={wilcoxon_result['p_value']:.4f} {sig}")

    # --- 4.3 时间稳定性评估 ---
    print("\n评估时间稳定性...")
    stability_analyzer = TemporalStability()

    all_distributions = [v['distribution'] for v in deviation_results.values()]
    ie_means = [p['IE_mean'] for p in periods]

    stability_result = stability_analyzer.assess_stability(
        distributions=all_distributions,
        ie_means=ie_means
    )

    print(f"  稳定性类别: {stability_result['category']}")
    print(f"  序列符号: {stability_result['sequence_notation']}")
    print(f"  系统性偏移: {'是' if stability_result['systematic_shift'] else '否'}")

    # --- 4.4 边际分布聚合 ---
    print("\n聚合边际分布...")
    marginal_analyzer = MarginalDistribution()

    marginal = marginal_analyzer.aggregate_distributions(
        distributions=all_distributions,
        stability_category=stability_result['category']
    )

    print(f"  边际中位数: {marginal['median']:.4f}")
    print(f"  边际IQR: {marginal['iqr']:.4f}")
    print(f"  预测能力: {marginal['predictive_power']}")

    # =========================================================================
    # 第五部分：Jaramillo (2022) 轨迹分析
    # =========================================================================
    print("\n【第五部分】Jaramillo (2022) 轨迹分析")
    print("-"*80)

    print("计算Budyko空间运动轨迹...")
    traj_analyzer = TrajectoryAnalyzer()

    # 计算T1 -> T3的运动
    movement = traj_analyzer.calculate_movement(
        catchment_id=basin_id,
        period_1={'IA': periods[0]['IA_mean'], 'IE': periods[0]['IE_mean'], 'name': 'T1'},
        period_2={'IA': periods[-1]['IA_mean'], 'IE': periods[-1]['IE_mean'], 'name': 'T3'},
        reference_omega=periods[0]['omega']
    )

    print(f"  起点（T1）: IA={movement.ia_t1:.3f}, IE={movement.ie_t1:.3f}")
    print(f"  终点（T3）: IA={movement.ia_t2:.3f}, IE={movement.ie_t2:.3f}")
    print(f"  运动强度: {movement.intensity:.4f}")
    print(f"  方向角: {movement.direction_angle:.1f}°")
    print(f"  遵循曲线: {'是' if movement.follows_curve else '否'}")
    print(f"  运动类型: {movement.movement_type}")

    # =========================================================================
    # 第六部分：偏差归因分析
    # =========================================================================
    print("\n【第六部分】偏差归因分析")
    print("-"*80)

    print("将Ibrahim偏差作为Y，人为驱动因子作为X，运行随机森林归因...")

    # 合并所有年度偏差
    all_years_with_deviation = []
    for pair_name, dev_result in deviation_results.items():
        dist = dev_result['distribution']
        period_idx = int(pair_name.split('-')[1]) - 1  # 'Δ1-2' -> period 1 (T2)

        if period_idx < len(periods):
            period = periods[period_idx]
            for year_idx, epsilon in enumerate(dist.annual_deviations):
                year = period['data'].iloc[year_idx]['year']

                # 提取该年的驱动因子
                year_data = data[data['year'] == year]
                if len(year_data) > 0:
                    all_years_with_deviation.append({
                        'year': year,
                        'epsilon_IE': epsilon,
                        'irrigation_ratio': year_data.iloc[0]['irrigation_ratio'],
                        'LAI': year_data.iloc[0]['LAI'],
                        'CO2': year_data.iloc[0]['CO2'],
                        'IA': year_data.iloc[0]['IA']
                    })

    attribution_data = pd.DataFrame(all_years_with_deviation)

    if len(attribution_data) > 10:
        attributor = DeviationAttributor()

        # 准备数据
        driver_names = ['irrigation_ratio', 'LAI', 'CO2', 'IA']
        X = attribution_data[driver_names].values
        y = attribution_data['epsilon_IE'].values

        # 训练模型
        attributor.fit(X, y, driver_names)

        # 特征重要性
        importance = attributor.get_feature_importance()

        print("  特征重要性:")
        for driver, imp in importance.items():
            print(f"    {driver}: {imp:.3f}")

        # 预测
        y_pred = attributor.predict(X)

        # 评估
        metrics = attributor.evaluate(y, y_pred)
        print(f"  模型R²: {metrics['r2']:.3f}")
        print(f"  模型RMSE: {metrics['rmse']:.4f}")
    else:
        print("  数据点不足，跳过归因分析")

    # =========================================================================
    # 第七部分：综合可视化
    # =========================================================================
    print("\n【第七部分】综合可视化")
    print("-"*80)

    print("生成多面板综合图表...")

    fig = plt.figure(figsize=(20, 12))

    # --- 面板1：Budyko空间轨迹 ---
    ax1 = fig.add_subplot(2, 3, 1)
    period_results = {p['name']: p for p in periods}
    BudykoVisualizer.plot_catchment_trajectory(
        ax1,
        period_results,
        omega_reference=periods[0]['omega']
    )
    ax1.set_title('Panel A: Budyko空间流域轨迹', fontsize=13, fontweight='bold', loc='left')

    # --- 面板2：PET对比 ---
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(data['year'], data['PET_standard'], 'b-', label='标准PET (PM)', linewidth=2)
    ax2.plot(data['year'], data['PET_innovation'], 'r-', label='创新PET (LAI+CO2)', linewidth=2)
    ax2.set_xlabel('年份', fontsize=11)
    ax2.set_ylabel('PET [mm/year]', fontsize=11)
    ax2.set_title('Panel B: PET方法对比', fontsize=13, fontweight='bold', loc='left')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # --- 面板3：LAI与CO2趋势 ---
    ax3 = fig.add_subplot(2, 3, 3)
    ax3_twin = ax3.twinx()

    line1 = ax3.plot(data['year'], data['LAI'], 'g-', linewidth=2, label='LAI')
    ax3.set_xlabel('年份', fontsize=11)
    ax3.set_ylabel('LAI [m²/m²]', fontsize=11, color='g')
    ax3.tick_params(axis='y', labelcolor='g')

    line2 = ax3_twin.plot(data['year'], data['CO2'], 'orange', linewidth=2, label='CO2')
    ax3_twin.set_ylabel('CO2 [ppm]', fontsize=11, color='orange')
    ax3_twin.tick_params(axis='y', labelcolor='orange')

    ax3.set_title('Panel C: LAI与CO2时间演变', fontsize=13, fontweight='bold', loc='left')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, fontsize=10, loc='upper left')
    ax3.grid(alpha=0.3)

    # --- 面板4：偏差分布 ---
    ax4 = fig.add_subplot(2, 3, 4)
    for i, (pair_name, dev_result) in enumerate(deviation_results.items()):
        dist = dev_result['distribution']
        ax4.hist(dist.annual_deviations, bins=10, alpha=0.5,
                label=f'{pair_name} (中位数={dist.median:.3f})', edgecolor='black')
    ax4.axvline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    ax4.set_xlabel('年度偏差 ε$_{IE,ω}$ [-]', fontsize=11)
    ax4.set_ylabel('频数', fontsize=11)
    ax4.set_title('Panel D: 偏差分布（Ibrahim方法）', fontsize=13, fontweight='bold', loc='left')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # --- 面板5：运动方向玫瑰图（简化版）---
    ax5 = fig.add_subplot(2, 3, 5, projection='polar')
    theta = np.deg2rad(movement.direction_angle)
    r = movement.intensity

    ax5.arrow(0, 0, theta, r, width=0.1, head_width=0.3, head_length=0.05,
             fc='red', ec='black', linewidth=2, length_includes_head=True)
    ax5.set_theta_zero_location('N')
    ax5.set_theta_direction(-1)
    ax5.set_title('Panel E: 运动方向（Jaramillo）', fontsize=13, fontweight='bold', pad=20)

    # 标注遵循/偏离曲线区域
    for angle_min, angle_max in traj_analyzer.FOLLOW_ANGLE_RANGES:
        ax5.fill_between(np.deg2rad(np.linspace(angle_min, angle_max, 50)),
                        0, r * 2, alpha=0.2, color='green',
                        label='遵循曲线' if angle_min == 45 else '')
    ax5.legend(fontsize=9, loc='upper right')

    # --- 面板6：DI vs SCI（3D框架）---
    if mask_grace.sum() > 0:
        ax6 = fig.add_subplot(2, 3, 6)
        grace_data = data[mask_grace]
        scatter = ax6.scatter(grace_data['IA'], grace_data['SCI'],
                            c=grace_data['IE'], cmap='viridis',
                            s=80, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax6.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax6.set_xlabel('干旱指数 IA [-]', fontsize=11)
        ax6.set_ylabel('储量变化指数 SCI [-]', fontsize=11)
        ax6.set_title('Panel F: DI-SCI关系（He 3D框架）', fontsize=13, fontweight='bold', loc='left')
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('IE [-]', fontsize=10)
        ax6.grid(alpha=0.3)

    plt.tight_layout()

    # 保存图表到脚本所在目录的 figures 文件夹，文件名带脚本名
    figures_dir = Path(__file__).resolve().parent / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    script_stem = Path(__file__).stem
    output_path = figures_dir / f"{script_stem}__comprehensive_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 综合图表已保存: {output_path}")

    # =========================================================================
    # 第八部分：结果总结
    # =========================================================================
    print("\n【第八部分】分析总结")
    print("="*80)

    print("\n1. 数据概况：")
    print(f"   - 分析时段: {years[0]} - {years[-1]} ({n_years}年)")
    print(f"   - 流域: {basin_id}")
    print(f"   - 平均降水: {data['P'].mean():.1f} mm/year")
    print(f"   - 平均径流: {data['Q'].mean():.1f} mm/year")
    print(f"   - 平均径流系数: {data['RC'].mean():.2f}")

    print("\n2. PET对比结果：")
    print(f"   - 标准PET vs 创新PET平均差异: {comparison['mean_diff']:.1f} mm/year")
    print(f"   - 两者相关系数: {comparison['corr']:.3f}")
    print(f"   → LAI和CO2对PET有{'显著' if abs(comparison['mean_diff']) > 50 else '一定'}影响")

    print("\n3. Budyko指数：")
    print(f"   - 干旱指数IA: {data['IA'].mean():.2f} ± {data['IA'].std():.2f}")
    print(f"   - 蒸发指数IE: {data['IE'].mean():.2f} ± {data['IE'].std():.2f}")
    print(f"   - 流域类型: {'水分限制' if data['IA'].mean() > 1 else '能量限制'}")

    print("\n4. Ibrahim偏差分析：")
    print(f"   - 时间稳定性: {stability_result['category']}")
    print(f"   - 边际偏差中位数: {marginal['median']:.4f}")
    print(f"   - 偏差IQR: {marginal['iqr']:.4f}")
    print(f"   - 预测能力: {marginal['predictive_power']}")
    print(f"   → 流域{'较稳定' if stability_result['category'] == 'Stable' else '有明显变化'}")

    print("\n5. Jaramillo轨迹分析：")
    print(f"   - 运动强度: {movement.intensity:.4f}")
    print(f"   - 方向角: {movement.direction_angle:.1f}°")
    print(f"   - 遵循Budyko曲线: {'是' if movement.follows_curve else '否'}")
    print(f"   - 运动类型: {movement.movement_type}")
    print(f"   → 流域{'沿着' if movement.follows_curve else '偏离'}Budyko曲线运动")

    if mask_grace.sum() > 0:
        print("\n6. He 3D框架：")
        sci_mean = data.loc[mask_grace, 'SCI'].mean()
        print(f"   - 平均SCI: {sci_mean:.4f}")
        print(f"   - TWS变化: {'储量增加' if sci_mean > 0 else '储量减少'}")
        print(f"   → 储量变化{'不可忽略' if abs(sci_mean) > 0.1 else '较小'}")

    print("\n7. 主要结论：")
    print("   ✓ 本代码库成功整合了三种前沿Budyko分析方法")
    print("   ✓ 创新PET模型能捕捉LAI和CO2对水文循环的影响")
    print("   ✓ 观测径流Q作为锚点，确保了分析的可靠性")
    print("   ✓ 综合分析揭示了流域水文变化的时空特征")

    print("\n"+"="*80)
    print("完整工作流程演示结束！")
    print("="*80)

    return data, periods, deviation_results, movement


if __name__ == '__main__':
    # 运行主流程
    data, periods, deviation_results, movement = main()

    print("\n提示：本示例使用模拟数据。实际应用时，请替换为真实数据路径。")
    print("数据可以通过以下方式获取：")
    print("  - 径流Q: 水文站观测数据")
    print("  - 气象数据: CMFD (http://data.tpdc.ac.cn)")
    print("  - GRACE TWS: JPL/CSR (https://grace.jpl.nasa.gov)")
    print("  - MODIS LAI: NASA LPDAAC (https://lpdaac.usgs.gov)")
    print("  - CO2: NOAA/ESRL (https://gml.noaa.gov/ccgg/trends/)")
