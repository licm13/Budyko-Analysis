#!/usr/bin/env python3
"""
主脚本：复杂Budyko分析工作流

该脚本整合了以下三种前沿分析范式：
1. Ibrahim (2025) 偏差分析
2. Jaramillo (2022) 轨迹分析
3. He (2023) 3D框架分析

工作流程：
- 生成/加载多源时间序列数据（径流、气象、GRACE TWS、MODIS LAI、CO2、人为驱动因子）
- 使用创新的（LAI+CO2）方法和标准方法计算PET并对比
- 计算2D（P-Q）和3D（P-Q-ΔS）的Budyko指数
- 执行偏差分析、轨迹分析和3D框架分析
- 使用机器学习进行偏差归因
- 生成综合可视化图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, Dict
import os

# 导入项目模块
from src.models.pet_models import PenmanMonteithPET
from src.models.pet_lai_co2 import PETWithLAICO2, PETComparator
from src.budyko.water_balance import WaterBalanceCalculator
from src.budyko.curves import BudykoCurves
from src.budyko.deviation import DeviationAnalysis, TemporalStability
from src.budyko.trajectory_jaramillo import TrajectoryAnalyzer
from src.analysis.deviation_attribution import DeviationAttributor

warnings.filterwarnings('ignore')


def generate_complex_data(seed: int = 42) -> pd.DataFrame:
    """
    生成60年（1961-2020）的复杂流域时间序列数据

    包含：
    - 气象数据：降水P, 温度T_avg, 相对湿度RH, 风速u2, 净辐射Rn_W
    - 陆面数据：LAI（有绿化趋势）, CO2（上升趋势）
    - 水文数据：TWS（2002年后）, irrigation_withdrawal（逐年增加）
    - 观测径流：Q_obs = P - EA_natural - ΔS - irrigation_withdrawal

    Parameters
    ----------
    seed : int
        随机种子

    Returns
    -------
    pd.DataFrame
        包含所有变量的年度数据框
    """
    np.random.seed(seed)
    n_years = 60
    years = np.arange(1961, 2021)

    # ===== 1. 气象数据 =====
    # 降水 (mm/yr)：多年平均800mm，带年际波动
    P_mean = 800
    P_trend = 5  # 轻微上升趋势
    P = P_mean + P_trend * np.arange(n_years) / n_years * 50 + \
        np.random.normal(0, 100, n_years)
    P = np.maximum(P, 300)  # 保证最小值

    # 温度 (°C)：平均15°C，有增温趋势
    T_base = 15
    T_warming = 0.03  # 每年升温0.03°C
    T_avg = T_base + T_warming * np.arange(n_years) + \
            np.random.normal(0, 1.5, n_years)

    # 相对湿度 (%)
    RH = 65 + np.random.normal(0, 8, n_years)
    RH = np.clip(RH, 30, 95)

    # 风速 (m/s)
    u2 = 2.0 + np.random.normal(0, 0.3, n_years)
    u2 = np.maximum(u2, 0.5)

    # 净辐射 (W/m²)
    Rn_W = 150 + 10 * np.sin(2 * np.pi * np.arange(n_years) / 30) + \
           np.random.normal(0, 15, n_years)
    Rn_W = np.maximum(Rn_W, 80)

    # ===== 2. 陆面数据 =====
    # LAI (m²/m²)：绿化趋势
    LAI_base = 2.5
    LAI_greening = 0.015  # 每年增加0.015
    LAI = LAI_base + LAI_greening * np.arange(n_years) + \
          np.random.normal(0, 0.2, n_years)
    LAI = np.maximum(LAI, 1.0)

    # CO2 (ppm)：从320ppm上升到410ppm
    CO2_start = 320
    CO2_end = 410
    CO2 = CO2_start + (CO2_end - CO2_start) * np.arange(n_years) / (n_years - 1) + \
          np.random.normal(0, 2, n_years)

    # ===== 3. 人为驱动因子 =====
    # 灌溉取水量 (mm/yr)：逐年增加
    irrigation_base = 10
    irrigation_growth = 0.5  # 每年增加0.5mm
    irrigation_withdrawal = irrigation_base + irrigation_growth * np.arange(n_years) + \
                            np.random.normal(0, 3, n_years)
    irrigation_withdrawal = np.maximum(irrigation_withdrawal, 0)

    # ===== 4. 水文数据 =====
    # 先计算自然状态下的蒸散发EA_natural
    # 这里简化假设：EA_natural ≈ 0.65 * P（自然条件下的蒸散发比例）
    EA_natural = 0.65 * P + np.random.normal(0, 30, n_years)

    # TWS变化 (mm)：仅2002年后有数据（GRACE时代）
    TWS = np.full(n_years, np.nan)
    # 2002-2020年：轻微下降趋势
    grace_years = years >= 2002
    n_grace = grace_years.sum()
    TWS_trend = -2  # 每年下降2mm
    TWS[grace_years] = TWS_trend * np.arange(n_grace) + \
                       np.random.normal(0, 15, n_grace)

    # ΔS = TWS的一阶差分
    delta_S = np.zeros(n_years)  # 初始化为0而不是NaN
    # 对于有TWS数据的年份，计算差分
    tws_indices = np.where(grace_years)[0]
    if len(tws_indices) > 1:
        delta_S[tws_indices[1:]] = np.diff(TWS[tws_indices])
    # 2002年之前，假设ΔS接近0（长期稳定）
    pre_grace = years < 2002
    delta_S[pre_grace] = np.random.normal(0, 5, pre_grace.sum())

    # ===== 5. 观测径流 Q_obs =====
    # 关键：Q_obs = P - EA_natural - ΔS - irrigation_withdrawal
    # 这样Q_obs明确受到灌溉的影响而偏离自然状态
    Q_obs = P - EA_natural - delta_S - irrigation_withdrawal
    Q_obs = np.maximum(Q_obs, 10)  # 保证最小径流

    # ===== 6. 组装DataFrame =====
    df = pd.DataFrame({
        'year': years,
        'P': P,
        'T_avg': T_avg,
        'RH': RH,
        'u2': u2,
        'Rn_W': Rn_W,
        'LAI': LAI,
        'CO2': CO2,
        'TWS': TWS,
        'irrigation_withdrawal': irrigation_withdrawal,
        'Q_obs': Q_obs,
        'delta_S': delta_S  # 添加delta_S用于后续分析
    })

    return df


def calculate_pet_methods(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用标准方法和创新方法计算PET

    Parameters
    ----------
    df : pd.DataFrame
        包含气象和陆面数据的数据框

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (PET_standard, PET_innovation)
    """
    print("\n" + "="*60)
    print("步骤1: PET计算")
    print("="*60)

    # 1. 标准Penman-Monteith方法
    print("\n[1/2] 使用标准Penman-Monteith方法...")
    pm_model = PenmanMonteithPET()

    # 简化假设：Tmax = T_avg + 5, Tmin = T_avg - 5
    Tmax = df['T_avg'].values + 5
    Tmin = df['T_avg'].values - 5

    # 将W/m²转换为MJ/m²/day
    Rs_MJ = df['Rn_W'].values * 0.0864  # W/m² → MJ/m²/day

    latitude = 35.0  # 假设纬度
    elevation = 500  # 假设海拔 (m)

    # 计算PET (mm/day)
    PET_standard_daily = pm_model.calculate(
        temp_avg=df['T_avg'].values,
        temp_max=Tmax,
        temp_min=Tmin,
        rh_mean=df['RH'].values,
        wind_speed=df['u2'].values,
        solar_radiation=Rs_MJ,
        latitude=latitude,
        elevation=elevation,
        day_of_year=182  # 简化:使用年中日期
    )

    # 转换为年总量 (mm/yr)
    PET_standard = PET_standard_daily * 365

    print(f"   ✓ PET_standard 范围: {PET_standard.min():.1f} - {PET_standard.max():.1f} mm/yr")

    # 2. 创新LAI+CO2方法
    print("\n[2/2] 使用LAI+CO2创新方法...")
    lai_co2_model = PETWithLAICO2()

    # 直接传入数组，避免循环 (返回 mm/day)
    PET_innovation_daily = lai_co2_model.calculate(
        temperature=df['T_avg'].values,
        humidity=df['RH'].values,
        wind_speed=df['u2'].values,
        radiation=df['Rn_W'].values,
        lai=df['LAI'].values,
        co2=df['CO2'].values
    )

    # 转换为年总量 (mm/yr)
    PET_innovation = PET_innovation_daily * 365

    print(f"   ✓ PET_innovation 范围: {PET_innovation.min():.1f} - {PET_innovation.max():.1f} mm/yr")

    # 3. 对比两种方法
    print("\n[对比] 两种PET方法的差异:")
    comparator = PETComparator()
    comparison = comparator.compare(PET_standard, PET_innovation)
    print(f"   RMSE: {comparison['rmse']:.2f} mm/yr")
    print(f"   相关系数: {comparison['corr']:.3f}")
    print(f"   平均差异: {comparison['mean_diff']:.2f} mm/yr")
    print(f"   PET_standard 均值: {comparison['mean_A']:.2f} mm/yr")
    print(f"   PET_innovation 均值: {comparison['mean_B']:.2f} mm/yr")

    return PET_standard, PET_innovation


def calculate_budyko_indices(df: pd.DataFrame, PET: np.ndarray) -> pd.DataFrame:
    """
    计算2D和3D Budyko指数

    Parameters
    ----------
    df : pd.DataFrame
        原始数据框
    PET : np.ndarray
        潜在蒸散发

    Returns
    -------
    pd.DataFrame
        添加了Budyko指数的数据框
    """
    print("\n" + "="*60)
    print("步骤2: 水量平衡与Budyko指数计算")
    print("="*60)

    calc = WaterBalanceCalculator()

    # 2D指数
    print("\n[1/2] 计算2D Budyko指数 (P-Q)...")
    EA_2D = df['P'].values - df['Q_obs'].values
    IA = PET / df['P'].values
    IE = EA_2D / df['P'].values

    print(f"   ✓ 干旱指数 IA 范围: {np.nanmin(IA):.2f} - {np.nanmax(IA):.2f}")
    print(f"   ✓ 蒸散发指数 IE 范围: {np.nanmin(IE):.2f} - {np.nanmax(IE):.2f}")

    # 3D指数
    print("\n[2/2] 计算3D Budyko指数 (P-Q-ΔS)...")
    EA_ext = df['P'].values - df['Q_obs'].values - df['delta_S'].values
    SCI = df['delta_S'].values / df['P'].values
    IE_ext = EA_ext / df['P'].values

    # 只在GRACE时代后统计
    grace_mask = df['year'].values >= 2002
    print(f"   ✓ 储水变化指数 SCI 范围 (2002+): {np.nanmin(SCI[grace_mask]):.3f} - {np.nanmax(SCI[grace_mask]):.3f}")
    print(f"   ✓ 扩展蒸散发指数 IE_ext 范围: {np.nanmin(IE_ext):.2f} - {np.nanmax(IE_ext):.2f}")

    # 添加到DataFrame
    df_result = df.copy()
    df_result['PET'] = PET
    df_result['EA_2D'] = EA_2D
    df_result['EA_ext'] = EA_ext
    df_result['IA'] = IA
    df_result['IE'] = IE
    df_result['SCI'] = SCI
    df_result['IE_ext'] = IE_ext

    return df_result


def perform_deviation_analysis(df: pd.DataFrame) -> Tuple[Dict, np.ndarray]:
    """
    执行Ibrahim (2025) 偏差分析

    Parameters
    ----------
    df : pd.DataFrame
        包含Budyko指数的数据框

    Returns
    -------
    Tuple[Dict, np.ndarray]
        (deviation_results字典, T3的年度偏差序列)
    """
    print("\n" + "="*60)
    print("步骤3: Ibrahim (2025) 偏差分析")
    print("="*60)

    # 划分三个时段
    T1_mask = (df['year'] >= 1961) & (df['year'] <= 1980)
    T2_mask = (df['year'] >= 1981) & (df['year'] <= 2000)
    T3_mask = (df['year'] >= 2001) & (df['year'] <= 2020)

    T1_data = df[T1_mask]
    T2_data = df[T2_mask]
    T3_data = df[T3_mask]

    print(f"\n时段划分:")
    print(f"   T1 (1961-1980): {len(T1_data)} 年")
    print(f"   T2 (1981-2000): {len(T2_data)} 年")
    print(f"   T3 (2001-2020): {len(T3_data)} 年")

    # 拟合ω参数
    print("\n[1/3] 拟合Budyko参数ω...")
    curve = BudykoCurves()

    omega_1, result_1 = curve.fit_omega(T1_data['IA'].values, T1_data['IE'].values)
    omega_2, result_2 = curve.fit_omega(T2_data['IA'].values, T2_data['IE'].values)

    print(f"   ✓ ω₁ (T1基准): {omega_1:.3f}")
    print(f"   ✓ ω₂ (T2基准): {omega_2:.3f}")

    # 计算偏差分布
    print("\n[2/3] 计算偏差分布...")
    dev_analyzer = DeviationAnalysis()

    # 过滤NaN值
    def filter_valid(ia, ie):
        """过滤掉包含NaN的数据点"""
        valid_mask = np.isfinite(ia) & np.isfinite(ie)
        return ia[valid_mask], ie[valid_mask]

    T1_IA, T1_IE = filter_valid(T1_data['IA'].values, T1_data['IE'].values)
    T2_IA, T2_IE = filter_valid(T2_data['IA'].values, T2_data['IE'].values)
    T3_IA, T3_IE = filter_valid(T3_data['IA'].values, T3_data['IE'].values)

    # T2相对于T1的偏差（使用ω₁作为基准）
    dist_T2 = dev_analyzer.calculate_deviations(
        ia_i=T1_IA,
        ie_obs_i=T1_IE,
        omega_i=omega_1,
        ia_i_plus_1=T2_IA,
        ie_obs_i_plus_1=T2_IE,
        period_pair='T1→T2'
    )

    # T3相对于T2的偏差（使用ω₂作为基准）
    dist_T3 = dev_analyzer.calculate_deviations(
        ia_i=T2_IA,
        ie_obs_i=T2_IE,
        omega_i=omega_2,
        ia_i_plus_1=T3_IA,
        ie_obs_i_plus_1=T3_IE,
        period_pair='T2→T3'
    )

    print(f"\n   T2偏差统计:")
    print(f"      中位数: {dist_T2.median:.4f}")
    print(f"      均值: {dist_T2.mean:.4f}")
    print(f"      标准差: {dist_T2.std:.4f}")
    print(f"      偏度: {dist_T2.skew:.3f}")

    print(f"\n   T3偏差统计:")
    print(f"      中位数: {dist_T3.median:.4f}")
    print(f"      均值: {dist_T3.mean:.4f}")
    print(f"      标准差: {dist_T3.std:.4f}")
    print(f"      偏度: {dist_T3.skew:.3f}")

    # 评估时间稳定性
    print("\n[3/3] 评估时间稳定性...")
    stability_analyzer = TemporalStability()

    ie_means = [
        T1_data['IE'].mean(),
        T2_data['IE'].mean(),
        T3_data['IE'].mean()
    ]

    stability = stability_analyzer.assess_stability(
        distributions=[dist_T2, dist_T3],
        ie_means=ie_means
    )

    print(f"   ✓ 稳定性类别: {stability}")

    # 保存结果
    results = {
        'omega_1': omega_1,
        'omega_2': omega_2,
        'dist_T2': dist_T2,
        'dist_T3': dist_T3,
        'stability': stability,
        'T1_data': T1_data,
        'T2_data': T2_data,
        'T3_data': T3_data
    }

    # 返回T3的年度偏差用于归因分析
    annual_deviations_T3 = dist_T3.annual_deviations

    return results, annual_deviations_T3


def perform_trajectory_analysis(deviation_results: Dict) -> Dict:
    """
    执行Jaramillo (2022) 轨迹分析

    Parameters
    ----------
    deviation_results : Dict
        偏差分析结果

    Returns
    -------
    Dict
        轨迹分析结果
    """
    print("\n" + "="*60)
    print("步骤4: Jaramillo (2022) 轨迹分析")
    print("="*60)

    T1_data = deviation_results['T1_data']
    T3_data = deviation_results['T3_data']
    omega_ref = deviation_results['omega_1']

    # 计算时段均值
    period_1 = {
        'IA': T1_data['IA'].mean(),
        'IE': T1_data['IE'].mean(),
        'name': 'T1 (1961-1980)'
    }

    period_2 = {
        'IA': T3_data['IA'].mean(),
        'IE': T3_data['IE'].mean(),
        'name': 'T3 (2001-2020)'
    }

    print(f"\n时段均值:")
    print(f"   T1: IA={period_1['IA']:.3f}, IE={period_1['IE']:.3f}")
    print(f"   T3: IA={period_2['IA']:.3f}, IE={period_2['IE']:.3f}")

    # 计算运动
    print("\n计算Budyko空间运动...")
    analyzer = TrajectoryAnalyzer()

    movement = analyzer.calculate_movement(
        catchment_id='watershed_001',
        period_1=period_1,
        period_2=period_2,
        reference_omega=omega_ref
    )

    print(f"\n运动分析结果:")
    print(f"   ✓ 运动强度 (Intensity): {movement.intensity:.4f}")
    print(f"   ✓ 方向角 (Direction): {movement.direction_angle:.1f}°")
    print(f"   ✓ 沿曲线运动 (Follows curve): {movement.follows_curve}")
    print(f"   ✓ 运动类型: {movement.movement_type}")
    print(f"   ✓ ΔIA: {movement.delta_ia:.4f}")
    print(f"   ✓ ΔIE: {movement.delta_ie:.4f}")

    return {
        'movement': movement,
        'period_1': period_1,
        'period_2': period_2
    }


def perform_deviation_attribution(df: pd.DataFrame, annual_deviations_T3: np.ndarray) -> Dict:
    """
    使用机器学习进行偏差归因

    Parameters
    ----------
    df : pd.DataFrame
        完整数据框
    annual_deviations_T3 : np.ndarray
        T3时段的年度偏差序列

    Returns
    -------
    Dict
        归因分析结果
    """
    print("\n" + "="*60)
    print("步骤5: 偏差归因分析（机器学习）")
    print("="*60)

    # 提取T3时段（2001-2020）的驱动因子
    T3_mask = (df['year'] >= 2001) & (df['year'] <= 2020)
    T3_df = df[T3_mask].copy()

    print(f"\n[1/3] 准备归因数据...")
    print(f"   时段: 2001-2020 (最多{len(T3_df)} 年)")
    print(f"   因变量: 年度偏差 (来自Ibrahim分析)")
    print(f"   自变量: irrigation_withdrawal, LAI, CO2")

    # 准备自变量
    driver_names = ['irrigation_withdrawal', 'LAI', 'CO2']
    X_full = T3_df[driver_names].values

    # 对齐X和y的长度（y可能因为NaN过滤而变短）
    n_deviations = len(annual_deviations_T3)
    if len(X_full) > n_deviations:
        # 只保留前n_deviations个样本（假设过滤是从前面开始的）
        X = X_full[:n_deviations]
        print(f"   注意: X数据从{len(X_full)}样本调整为{len(X)}样本以匹配y")
    else:
        X = X_full

    y = annual_deviations_T3

    # 数据标准化（帮助模型训练）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n   驱动因子统计:")
    for i, name in enumerate(driver_names):
        print(f"      {name}: {X[:, i].mean():.2f} ± {X[:, i].std():.2f}")

    # 使用DeviationAttributor
    print(f"\n[2/3] 训练随机森林模型...")
    attributor = DeviationAttributor()

    attributor.fit(
        X=X_scaled,
        y=y,
        driver_names=driver_names,
        n_estimators=100,
        max_depth=5
    )

    # 获取特征重要性
    print(f"\n[3/3] 分析特征重要性...")
    importance_dict = attributor.get_feature_importance()

    # 转换为DataFrame并排序
    importance_df = pd.DataFrame([
        {'driver': name, 'importance': imp}
        for name, imp in importance_dict.items()
    ]).sort_values('importance', ascending=False)

    print(f"\n   ✓ 驱动因子重要性排序:")
    for idx, row in importance_df.iterrows():
        print(f"      {idx+1}. {row['driver']}: {row['importance']:.3f}")

    # 模型性能
    y_pred = attributor.predict(X_scaled)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
    rmse = np.sqrt(np.mean((y - y_pred)**2))

    print(f"\n   模型性能:")
    print(f"      R²: {r2:.3f}")
    print(f"      RMSE: {rmse:.4f}")

    # 解释
    top_driver = importance_df.iloc[0]['driver']
    print(f"\n   结论: 最重要的驱动因子是 '{top_driver}'")
    print(f"         这表明流域偏离Budyko曲线主要由该因素驱动")

    return {
        'attributor': attributor,
        'importance_df': importance_df,
        'driver_names': driver_names,
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'r2': r2,
        'rmse': rmse
    }


def create_comprehensive_visualization(
    df: pd.DataFrame,
    deviation_results: Dict,
    trajectory_results: Dict,
    attribution_results: Dict,
    output_path: str = 'budyko_comprehensive_analysis.png'
):
    """
    创建2x2综合可视化

    Parameters
    ----------
    df : pd.DataFrame
        完整数据框
    deviation_results : Dict
        偏差分析结果
    trajectory_results : Dict
        轨迹分析结果
    attribution_results : Dict
        归因分析结果
    output_path : str
        输出文件路径
    """
    print("\n" + "="*60)
    print("步骤6: 生成综合可视化")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ===== 子图(a): Budyko空间 =====
    print("\n[1/4] 绘制Budyko空间...")
    ax = axes[0, 0]

    # 绘制理论曲线
    ia_range = np.linspace(0.1, 5, 200)
    omega_1 = deviation_results['omega_1']
    ie_curve = BudykoCurves.tixeront_fu(ia_range, omega_1)

    ax.plot(ia_range, ie_curve, 'k-', linewidth=2, label=f'Fu曲线 (ω={omega_1:.2f})', zorder=1)

    # 绘制数据点（按时段着色）
    T1_data = deviation_results['T1_data']
    T2_data = deviation_results['T2_data']
    T3_data = deviation_results['T3_data']

    ax.scatter(T1_data['IA'], T1_data['IE'], c='blue', s=50, alpha=0.6, label='T1 (1961-1980)', zorder=2)
    ax.scatter(T2_data['IA'], T2_data['IE'], c='green', s=50, alpha=0.6, label='T2 (1981-2000)', zorder=2)
    ax.scatter(T3_data['IA'], T3_data['IE'], c='red', s=50, alpha=0.6, label='T3 (2001-2020)', zorder=2)

    # 水量平衡限制线
    ax.plot([0, 5], [1, 1], 'k--', linewidth=1, alpha=0.5, label='水量限制')
    ax.plot([0, 5], [0, 1], 'k--', linewidth=1, alpha=0.5, label='能量限制')

    ax.set_xlabel('干旱指数 IA (PET/P)', fontsize=11, fontweight='bold')
    ax.set_ylabel('蒸散发指数 IE (EA/P)', fontsize=11, fontweight='bold')
    ax.set_title('(a) Budyko空间', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.2)

    # ===== 子图(b): 3D框架 (He 2023) =====
    print("[2/4] 绘制3D框架...")
    ax = axes[0, 1]

    # 绘制IA vs SCI，按年份着色
    grace_mask = df['year'] >= 2002
    df_grace = df[grace_mask]

    scatter = ax.scatter(
        df_grace['IA'],
        df_grace['SCI'],
        c=df_grace['year'],
        cmap='viridis',
        s=80,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('年份', fontsize=10)

    # 添加零线
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('干旱指数 IA (PET/P)', fontsize=11, fontweight='bold')
    ax.set_ylabel('储水变化指数 SCI (ΔS/P)', fontsize=11, fontweight='bold')
    ax.set_title('(b) 3D框架 (He 2023)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # ===== 子图(c): Jaramillo轨迹 =====
    print("[3/4] 绘制Jaramillo轨迹...")
    ax = axes[1, 0]

    # 绘制理论曲线
    ax.plot(ia_range, ie_curve, 'k-', linewidth=2, alpha=0.5, zorder=1)

    # 绘制时段均值点
    p1 = trajectory_results['period_1']
    p2 = trajectory_results['period_2']

    ax.scatter(p1['IA'], p1['IE'], c='blue', s=200, marker='o',
               edgecolors='black', linewidths=2, label='T1均值', zorder=3)
    ax.scatter(p2['IA'], p2['IE'], c='red', s=200, marker='o',
               edgecolors='black', linewidths=2, label='T3均值', zorder=3)

    # 绘制运动箭头
    movement = trajectory_results['movement']
    ax.annotate('', xy=(p2['IA'], p2['IE']),
                xytext=(p1['IA'], p1['IE']),
                arrowprops=dict(arrowstyle='->', lw=3, color='darkred'))

    # 添加运动信息
    ax.text(0.05, 0.95,
            f"运动强度: {movement.intensity:.4f}\n方向: {movement.direction_angle:.1f}°\n类型: {movement.movement_type}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('干旱指数 IA (PET/P)', fontsize=11, fontweight='bold')
    ax.set_ylabel('蒸散发指数 IE (EA/P)', fontsize=11, fontweight='bold')
    ax.set_title('(c) Jaramillo轨迹分析', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.2)

    # ===== 子图(d): 偏差归因 =====
    print("[4/4] 绘制偏差归因...")
    ax = axes[1, 1]

    importance_df = attribution_results['importance_df']

    # 条形图
    bars = ax.barh(importance_df['driver'], importance_df['importance'],
                   color=['#e74c3c', '#3498db', '#2ecc71'])

    # 添加数值标签
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        ax.text(row['importance'] + 0.01, i, f"{row['importance']:.3f}",
                va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('特征重要性', fontsize=11, fontweight='bold')
    ax.set_title(f"(d) 偏差归因 (R²={attribution_results['r2']:.3f})",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, importance_df['importance'].max() * 1.2)

    # 整体调整
    plt.tight_layout()

    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存到: {output_path}")

    plt.close()


def main():
    """主函数：执行完整工作流"""
    print("\n" + "="*60)
    print("Budyko分析工作流 - 主脚本")
    print("="*60)
    print("整合三种前沿分析范式:")
    print("  1. Ibrahim (2025) 偏差分析")
    print("  2. Jaramillo (2022) 轨迹分析")
    print("  3. He (2023) 3D框架分析")
    print("="*60)

    # ===== 0. 数据准备 =====
    print("\n" + "="*60)
    print("步骤0: 数据准备")
    print("="*60)
    print("\n生成60年（1961-2020）复杂流域数据...")
    df = generate_complex_data(seed=42)
    print(f"✓ 数据生成完成: {len(df)} 年")
    print(f"✓ 变量: {', '.join(df.columns)}")

    # ===== 1. PET计算 =====
    PET_standard, PET_innovation = calculate_pet_methods(df)

    # 选择PET方法（这里使用创新方法）
    PET_selected = PET_innovation
    print(f"\n✓ 选择PET方法: 创新LAI+CO2方法")

    # ===== 2. Budyko指数计算 =====
    df = calculate_budyko_indices(df, PET_selected)

    # ===== 3. Ibrahim偏差分析 =====
    deviation_results, annual_deviations_T3 = perform_deviation_analysis(df)

    # ===== 4. Jaramillo轨迹分析 =====
    trajectory_results = perform_trajectory_analysis(deviation_results)

    # ===== 5. 偏差归因 =====
    attribution_results = perform_deviation_attribution(df, annual_deviations_T3)

    # ===== 6. 综合可视化 =====
    output_file = 'budyko_comprehensive_analysis.png'
    create_comprehensive_visualization(
        df=df,
        deviation_results=deviation_results,
        trajectory_results=trajectory_results,
        attribution_results=attribution_results,
        output_path=output_file
    )

    # ===== 7. 总结 =====
    print("\n" + "="*60)
    print("工作流完成！")
    print("="*60)
    print(f"\n主要发现:")
    print(f"  1. Budyko参数ω演变: {deviation_results['omega_1']:.3f} → {deviation_results['omega_2']:.3f}")
    print(f"  2. 时间稳定性: {deviation_results['stability']}")
    print(f"  3. Budyko空间运动强度: {trajectory_results['movement'].intensity:.4f}")
    print(f"  4. 主要驱动因子: {attribution_results['importance_df'].iloc[0]['driver']}")
    print(f"  5. 归因模型R²: {attribution_results['r2']:.3f}")
    print(f"\n输出文件:")
    print(f"  - {output_file}")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
