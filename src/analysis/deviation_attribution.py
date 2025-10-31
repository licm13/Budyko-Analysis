# src/analysis/deviation_attribution.py
"""
Budyko偏差归因分析模块

核心思路：
----------
径流数据(Q)揭示了流域偏离Budyko曲线的"症状"
本模块用于诊断导致偏离的"病因"（驱动因子）

归因方法：
1. 相关分析：识别关键驱动因子
2. 回归分析：量化贡献率
3. 机器学习：非线性关系挖掘
4. 敏感性分析：评估不确定性

潜在驱动因子：
- 气候变化：温度、降水模式变化
- 土地利用：森林砍伐、城市化
- 人类活动：灌溉、水库、引水
- 植被变化：LAI、NDVI趋势
- 积雪：融雪时间、积雪深度变化

References
----------
- Ibrahim et al. (2023). Water-Energy Balance Framework.
- Jaramillo et al. (2022). Basins Following Budyko Curves.
- Yang et al. (2014). Attribution of catchment evapotranspiration trends.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings


@dataclass
class AttributionResults:
    """归因分析结果数据类"""
    method: str
    driver_importance: Dict[str, float]
    explained_variance: float
    predictions: np.ndarray
    residuals: np.ndarray
    feature_contributions: Optional[Dict[str, np.ndarray]]


class DeviationAttributor:
    """
    Budyko偏差归因分析器

    将径流Q揭示的Budyko偏差作为因变量，
    分析各种驱动因子的贡献
    """

    def __init__(self):
        self.drivers = {}
        self.deviation = None
        self.scaler = StandardScaler()

    def set_deviation(self, deviation: np.ndarray):
        """
        设置Budyko偏差（因变量）

        这个偏差是由径流Q计算得到的：
        deviation = IE_observed - IE_theory
        其中 IE_observed = (P - Q) / P

        Parameters
        ----------
        deviation : np.ndarray
            Budyko偏差
        """
        self.deviation = np.asarray(deviation)

    def add_driver(self,
                   name: str,
                   values: np.ndarray,
                   description: str = ""):
        """
        添加驱动因子

        Parameters
        ----------
        name : str
            驱动因子名称
        values : np.ndarray
            驱动因子值
        description : str
            描述
        """
        if self.deviation is not None:
            if len(values) != len(self.deviation):
                raise ValueError(f"驱动因子 '{name}' 长度与偏差不一致")

        self.drivers[name] = {
            'values': np.asarray(values),
            'description': description
        }

    def correlate_drivers(self,
                         method: str = 'pearson') -> pd.DataFrame:
        """
        计算驱动因子与偏差的相关性

        Parameters
        ----------
        method : str
            相关方法: 'pearson', 'spearman', 'kendall'

        Returns
        -------
        pd.DataFrame
            相关性结果
        """
        if self.deviation is None:
            raise ValueError("请先设置偏差数据")

        if len(self.drivers) == 0:
            raise ValueError("请先添加驱动因子")

        results = []

        for name, driver_info in self.drivers.items():
            values = driver_info['values']

            # 移除NaN值
            mask = ~(np.isnan(values) | np.isnan(self.deviation))
            values_clean = values[mask]
            deviation_clean = self.deviation[mask]

            if len(values_clean) < 3:
                warnings.warn(f"驱动因子 '{name}' 有效数据点过少，跳过")
                continue

            # 计算相关系数
            if method == 'pearson':
                corr, pval = stats.pearsonr(values_clean, deviation_clean)
            elif method == 'spearman':
                corr, pval = stats.spearmanr(values_clean, deviation_clean)
            elif method == 'kendall':
                corr, pval = stats.kendalltau(values_clean, deviation_clean)
            else:
                raise ValueError(f"Unknown correlation method: {method}")

            results.append({
                'driver': name,
                'correlation': corr,
                'p_value': pval,
                'significant': pval < 0.05,
                'n_samples': len(values_clean)
            })

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('correlation', key=abs, ascending=False)

        return df_results

    def linear_regression_attribution(self,
                                     drivers: List[str] = None) -> AttributionResults:
        """
        线性回归归因

        Parameters
        ----------
        drivers : List[str], optional
            要使用的驱动因子列表，默认使用所有

        Returns
        -------
        AttributionResults
            归因结果
        """
        if drivers is None:
            drivers = list(self.drivers.keys())

        # 准备数据
        X, y, valid_mask = self._prepare_data(drivers)

        # 训练模型
        model = LinearRegression()
        model.fit(X, y)

        # 预测
        y_pred = model.predict(X)
        residuals = y - y_pred

        # 解释方差
        explained_var = 1 - np.var(residuals) / np.var(y)

        # 驱动因子重要性（标准化系数）
        importance = {}
        for i, driver_name in enumerate(drivers):
            importance[driver_name] = abs(model.coef_[i])

        # 归一化重要性
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}

        # 计算各驱动因子对偏差的贡献
        contributions = {}
        for i, driver_name in enumerate(drivers):
            contributions[driver_name] = model.coef_[i] * X[:, i]

        return AttributionResults(
            method='linear_regression',
            driver_importance=importance,
            explained_variance=explained_var,
            predictions=self._expand_predictions(y_pred, valid_mask),
            residuals=self._expand_predictions(residuals, valid_mask),
            feature_contributions=self._expand_contributions(contributions, valid_mask)
        )

    def random_forest_attribution(self,
                                  drivers: List[str] = None,
                                  n_estimators: int = 100,
                                  max_depth: int = 10) -> AttributionResults:
        """
        随机森林归因（捕捉非线性关系）

        Parameters
        ----------
        drivers : List[str], optional
            要使用的驱动因子列表
        n_estimators : int
            树的数量
        max_depth : int
            最大深度

        Returns
        -------
        AttributionResults
            归因结果
        """
        if drivers is None:
            drivers = list(self.drivers.keys())

        # 准备数据
        X, y, valid_mask = self._prepare_data(drivers)

        # 训练模型
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X, y)

        # 预测
        y_pred = model.predict(X)
        residuals = y - y_pred

        # 解释方差
        explained_var = model.score(X, y)

        # 驱动因子重要性
        importance = {}
        for i, driver_name in enumerate(drivers):
            importance[driver_name] = model.feature_importances_[i]

        # 计算SHAP值（简化版）- 实际使用可考虑shap库
        contributions = {}
        for i, driver_name in enumerate(drivers):
            # 简化：使用特征重要性 * 特征值作为近似贡献
            contributions[driver_name] = model.feature_importances_[i] * X[:, i]

        return AttributionResults(
            method='random_forest',
            driver_importance=importance,
            explained_variance=explained_var,
            predictions=self._expand_predictions(y_pred, valid_mask),
            residuals=self._expand_predictions(residuals, valid_mask),
            feature_contributions=self._expand_contributions(contributions, valid_mask)
        )

    def _prepare_data(self, drivers: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """准备建模数据"""
        # 收集驱动因子数据
        X_list = []
        for driver_name in drivers:
            X_list.append(self.drivers[driver_name]['values'])

        X = np.column_stack(X_list)
        y = self.deviation

        # 移除NaN行
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        # 标准化
        X_scaled = self.scaler.fit_transform(X_clean)

        return X_scaled, y_clean, valid_mask

    def _expand_predictions(self,
                           predictions: np.ndarray,
                           valid_mask: np.ndarray) -> np.ndarray:
        """将预测结果扩展到原始长度"""
        full_predictions = np.full(len(valid_mask), np.nan)
        full_predictions[valid_mask] = predictions
        return full_predictions

    def _expand_contributions(self,
                             contributions: Dict[str, np.ndarray],
                             valid_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """扩展贡献值"""
        expanded = {}
        for name, contrib in contributions.items():
            full_contrib = np.full(len(valid_mask), np.nan)
            full_contrib[valid_mask] = contrib
            expanded[name] = full_contrib
        return expanded


class TemporalAttributor:
    """
    时间演变归因分析器

    分析Budyko偏差随时间变化的驱动因子
    """

    def __init__(self):
        pass

    def analyze_temporal_trends(self,
                                deviation: np.ndarray,
                                drivers: Dict[str, np.ndarray],
                                years: np.ndarray) -> pd.DataFrame:
        """
        分析偏差和驱动因子的时间趋势

        Parameters
        ----------
        deviation : np.ndarray
            Budyko偏差时间序列
        drivers : Dict[str, np.ndarray]
            驱动因子时间序列
        years : np.ndarray
            年份数组

        Returns
        -------
        pd.DataFrame
            趋势分析结果
        """
        results = []

        # 偏差趋势
        deviation_trend = self._calculate_trend(years, deviation)
        results.append({
            'variable': 'Budyko_Deviation',
            'trend': deviation_trend['slope'],
            'p_value': deviation_trend['p_value'],
            'significant': deviation_trend['p_value'] < 0.05
        })

        # 驱动因子趋势
        for name, values in drivers.items():
            trend = self._calculate_trend(years, values)
            results.append({
                'variable': name,
                'trend': trend['slope'],
                'p_value': trend['p_value'],
                'significant': trend['p_value'] < 0.05
            })

        return pd.DataFrame(results)

    def _calculate_trend(self,
                        x: np.ndarray,
                        y: np.ndarray) -> Dict:
        """计算线性趋势"""
        # 移除NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 3:
            return {'slope': np.nan, 'p_value': np.nan}

        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }


# 使用示例
if __name__ == '__main__':
    # 创建示例数据
    n_basins = 100
    np.random.seed(42)

    # 模拟Budyko偏差（由径流Q决定）
    deviation = 0.05 + 0.1 * np.random.randn(n_basins)

    # 模拟驱动因子
    drivers_data = {
        'irrigation': 0.3 * deviation + 0.1 * np.random.randn(n_basins),
        'forest_loss': -0.2 * deviation + 0.08 * np.random.randn(n_basins),
        'lai_trend': 0.15 * deviation + 0.05 * np.random.randn(n_basins),
        'temperature_trend': 0.1 * np.random.randn(n_basins),
        'reservoir': 0.4 * deviation + 0.12 * np.random.randn(n_basins)
    }

    # 归因分析
    attributor = DeviationAttributor()
    attributor.set_deviation(deviation)

    for name, values in drivers_data.items():
        attributor.add_driver(name, values)

    # 相关分析
    print("======= 驱动因子相关性分析 =======\n")
    corr_results = attributor.correlate_drivers(method='pearson')
    print(corr_results.to_string(index=False))

    # 线性回归归因
    print("\n======= 线性回归归因 =======\n")
    lr_results = attributor.linear_regression_attribution()
    print(f"解释方差: {lr_results.explained_variance:.2%}")
    print("\n驱动因子重要性:")
    for driver, importance in sorted(lr_results.driver_importance.items(),
                                    key=lambda x: x[1], reverse=True):
        print(f"  {driver}: {importance:.2%}")

    # 随机森林归因
    print("\n======= 随机森林归因 =======\n")
    rf_results = attributor.random_forest_attribution(n_estimators=100)
    print(f"解释方差 (R²): {rf_results.explained_variance:.2%}")
    print("\n驱动因子重要性:")
    for driver, importance in sorted(rf_results.driver_importance.items(),
                                    key=lambda x: x[1], reverse=True):
        print(f"  {driver}: {importance:.2%}")

    # 对比两种方法
    print("\n======= 方法对比 =======\n")
    print("驱动因子排名对比:")
    print("\n线性回归 vs 随机森林:")
    for driver in drivers_data.keys():
        lr_imp = lr_results.driver_importance.get(driver, 0)
        rf_imp = rf_results.driver_importance.get(driver, 0)
        print(f"  {driver:20s}: LR={lr_imp:.3f}, RF={rf_imp:.3f}")
