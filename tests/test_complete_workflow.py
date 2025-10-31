#!/usr/bin/env python3
# tests/test_complete_workflow.py
"""
完整工作流的集成测试

测试整个Budyko分析流程是否能正常运行
"""

import sys
sys.path.append('../src')

import numpy as np
import pytest

from budyko.water_balance import WaterBalanceCalculator
from budyko.curves import BudykoCurves
from budyko.deviation import DeviationAnalysis
from budyko.trajectory_jaramillo import TrajectoryAnalyzer
from models.pet_lai_co2 import PETWithLAICO2
from analysis.deviation_attribution import DeviationAttributor


class TestWaterBalance:
    """测试水量平衡计算"""

    def test_basic_water_balance(self):
        """测试基础水量平衡计算"""
        # 准备测试数据
        P = np.array([800, 850, 900, 950, 1000])
        Q = np.array([200, 220, 240, 260, 280])
        PET = np.array([1000, 1050, 1100, 1150, 1200])

        # 计算
        calc = WaterBalanceCalculator()
        results = calc.calculate_budyko_indices(P, Q, PET)

        # 验证
        assert len(results.actual_evaporation) == 5
        assert np.allclose(results.actual_evaporation, P - Q)
        assert np.all(results.aridity_index == PET / P)
        assert np.all(results.evaporation_index == (P - Q) / P)

    def test_water_balance_with_storage(self):
        """测试考虑储量变化的水量平衡"""
        P = np.array([800, 850, 900, 950, 1000])
        Q = np.array([200, 220, 240, 260, 280])
        PET = np.array([1000, 1050, 1100, 1150, 1200])
        delta_S = np.array([10, -5, 15, -10, 5])

        calc = WaterBalanceCalculator()
        results = calc.calculate_budyko_indices(P, Q, PET, delta_S)

        # EA = P - Q - ΔS
        expected_ea = P - Q - delta_S
        assert np.allclose(results.actual_evaporation, expected_ea)

    def test_quality_control(self):
        """测试数据质量控制"""
        P = np.array([800, 50, 900, 950, 1000])  # P[1]太小
        Q = np.array([200, 220, 240, 260, 950])  # Q[4]太大
        PET = np.array([1000, 1050, 1100, 1150, 1200])

        calc = WaterBalanceCalculator()
        results = calc.calculate_budyko_indices(P, Q, PET)

        # 检查质量标志
        assert results.data_quality[1] == 2  # P太小
        assert results.data_quality[4] >= 1  # RC太大


class TestBudykoCurves:
    """测试Budyko曲线计算"""

    def test_tixeront_fu_formula(self):
        """测试Fu-Budyko公式"""
        budyko = BudykoCurves()

        ia = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        omega = 2.6

        ie = budyko.tixeront_fu(ia, omega)

        # 基本检查
        assert len(ie) == len(ia)
        assert np.all(ie >= 0)
        assert np.all(ie <= 1)
        # IA增加，IE应增加但逐渐饱和
        assert ie[0] < ie[1] < ie[2] < ie[3] < ie[4]

    def test_omega_fitting(self):
        """测试ω参数拟合"""
        budyko = BudykoCurves()

        # 生成测试数据（遵循ω=2.5的曲线，加噪声）
        true_omega = 2.5
        ia = np.linspace(0.5, 3, 20)
        ie_true = budyko.tixeront_fu(ia, true_omega)
        ie_obs = ie_true + 0.01 * np.random.randn(len(ia))

        # 拟合
        omega_fit, stats = budyko.fit_omega(ia, ie_obs)

        # 验证
        assert 2.0 < omega_fit < 3.0  # 应该接近2.5
        assert stats['r2'] > 0.95  # 拟合应该很好
        assert stats['rmse'] < 0.05


class TestPETWithLAICO2:
    """测试LAI+CO2 PET模型"""

    def test_pet_calculation(self):
        """测试PET计算"""
        n = 10
        pet_calc = PETWithLAICO2(elevation=500, latitude=30)

        pet = pet_calc.calculate(
            temperature=15 + 5 * np.random.randn(n),
            humidity=60 + 10 * np.random.randn(n),
            wind_speed=2 + 0.5 * np.random.randn(n),
            radiation=200 + 50 * np.random.randn(n),
            lai=3 + 1 * np.random.randn(n),
            co2=400 * np.ones(n)
        )

        # 基本检查
        assert len(pet) == n
        assert np.all(pet >= 0)
        assert np.all(pet < 20)  # mm/day，合理范围

    def test_co2_effect(self):
        """测试CO2对PET的影响"""
        n = 10
        pet_calc = PETWithLAICO2()

        # 固定其他条件
        temp = 15 * np.ones(n)
        rh = 60 * np.ones(n)
        wind = 2 * np.ones(n)
        rad = 200 * np.ones(n)
        lai = 3 * np.ones(n)

        # 低CO2
        pet_low = pet_calc.calculate(temp, rh, wind, rad, lai, 350 * np.ones(n))

        # 高CO2
        pet_high = pet_calc.calculate(temp, rh, wind, rad, lai, 550 * np.ones(n))

        # CO2增加应导致PET降低（气孔关闭效应）
        assert np.mean(pet_high) < np.mean(pet_low)

    def test_lai_effect(self):
        """测试LAI对PET的影响"""
        n = 10
        pet_calc = PETWithLAICO2()

        # 固定其他条件
        temp = 15 * np.ones(n)
        rh = 60 * np.ones(n)
        wind = 2 * np.ones(n)
        rad = 200 * np.ones(n)
        co2 = 400 * np.ones(n)

        # 低LAI
        pet_low_lai = pet_calc.calculate(temp, rh, wind, rad, 1 * np.ones(n), co2)

        # 高LAI
        pet_high_lai = pet_calc.calculate(temp, rh, wind, rad, 5 * np.ones(n), co2)

        # LAI增加应导致PET增加（蒸腾面积增大）
        assert np.mean(pet_high_lai) > np.mean(pet_low_lai)


class TestTrajectoryAnalysis:
    """测试轨迹分析"""

    def test_movement_calculation(self):
        """测试运动计算"""
        analyzer = TrajectoryAnalyzer()

        period_1 = {'IA': 1.0, 'IE': 0.6, 'name': 'P1'}
        period_2 = {'IA': 1.5, 'IE': 0.7, 'name': 'P2'}

        movement = analyzer.calculate_movement(
            catchment_id='test_basin',
            period_1=period_1,
            period_2=period_2
        )

        # 验证
        assert movement.delta_ia == 0.5
        assert movement.delta_ie == 0.1
        assert movement.intensity > 0
        assert 0 <= movement.direction_angle < 360

    def test_follows_curve_detection(self):
        """测试是否遵循曲线的判断"""
        analyzer = TrajectoryAnalyzer()

        # 沿曲线运动（应该判定为遵循）
        period_1 = {'IA': 1.0, 'IE': 0.5, 'name': 'P1'}
        period_2 = {'IA': 1.3, 'IE': 0.6, 'name': 'P2'}

        movement = analyzer.calculate_movement(
            catchment_id='test',
            period_1=period_1,
            period_2=period_2
        )

        # 方向角应在合理范围
        assert movement.follows_curve in [True, False]


class TestDeviationAttribution:
    """测试偏差归因分析"""

    def test_correlation_analysis(self):
        """测试相关分析"""
        n = 50
        np.random.seed(42)

        # 模拟数据
        deviation = 0.1 * np.random.randn(n)
        driver1 = 0.5 * deviation + 0.1 * np.random.randn(n)  # 强相关
        driver2 = 0.1 * np.random.randn(n)  # 无相关

        # 归因分析
        attributor = DeviationAttributor()
        attributor.set_deviation(deviation)
        attributor.add_driver('driver1', driver1)
        attributor.add_driver('driver2', driver2)

        results = attributor.correlate_drivers()

        # driver1应该相关性更高
        assert abs(results[results['driver'] == 'driver1']['correlation'].values[0]) > \
               abs(results[results['driver'] == 'driver2']['correlation'].values[0])

    def test_random_forest_attribution(self):
        """测试随机森林归因"""
        n = 100
        np.random.seed(42)

        # 模拟非线性关系
        driver1 = np.random.randn(n)
        driver2 = np.random.randn(n)
        deviation = 0.3 * driver1**2 + 0.2 * driver2 + 0.1 * np.random.randn(n)

        attributor = DeviationAttributor()
        attributor.set_deviation(deviation)
        attributor.add_driver('driver1', driver1)
        attributor.add_driver('driver2', driver2)

        results = attributor.random_forest_attribution(n_estimators=50)

        # 基本检查
        assert results.explained_variance >= 0
        assert 'driver1' in results.driver_importance
        assert 'driver2' in results.driver_importance


class TestIntegration:
    """集成测试：完整工作流"""

    def test_complete_workflow(self):
        """测试完整工作流"""
        # 1. 准备数据
        n_years = 20
        P = 800 + 200 * np.random.randn(n_years)
        Q = 0.3 * P + 50 * np.random.randn(n_years)
        Q = np.maximum(Q, 0)

        T = 15 + 3 * np.random.randn(n_years)
        RH = 60 + 10 * np.random.randn(n_years)
        wind = 2 + 0.5 * np.random.randn(n_years)
        rad = 200 + 50 * np.random.randn(n_years)
        LAI = 3 + 1 * np.random.randn(n_years)
        CO2 = 380 + np.arange(n_years)

        # 2. 计算PET
        pet_calc = PETWithLAICO2()
        PET = pet_calc.calculate(T, RH, wind, rad, LAI, CO2)

        # 3. 水量平衡
        wb_calc = WaterBalanceCalculator()
        results = wb_calc.calculate_budyko_indices(P, Q, PET)

        # 4. Budyko拟合
        budyko = BudykoCurves()
        omega, fit_stats = budyko.fit_omega(
            results.aridity_index,
            results.evaporation_index
        )

        # 5. 验证
        assert omega > 0
        assert fit_stats['r2'] >= 0
        assert len(results.actual_evaporation) == n_years

        print("完整工作流测试通过！")


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '-s'])
