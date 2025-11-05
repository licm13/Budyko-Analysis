# src/budyko/deviation.py
"""
Budyko曲线偏差计算与分析
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class DeviationDistribution:
    """偏差分布数据类"""
    period_name: str
    annual_deviations: np.ndarray
    median: float
    mean: float
    std: float
    iqr: float
    skew: float
    fitted_params: Dict


class DeviationAnalysis:
    """偏差分析类"""
    
    def __init__(self, period_length: int = 20):
        """
        Parameters
        ----------
        period_length : int
            时段长度（年），默认20年
        """
        self.period_length = period_length
        self.distributions = {}
    
    def calculate_deviations(self,
                            ia_i: np.ndarray,
                            ie_obs_i: np.ndarray,
                            omega_i: float,
                            ia_i_plus_1: np.ndarray,
                            ie_obs_i_plus_1: np.ndarray,
                            period_pair: str) -> DeviationDistribution:
        """
        计算时段Ti和Ti+1之间的偏差分布（Step 2）
        
        Parameters
        ----------
        ia_i : np.ndarray
            时段i的干旱指数（年度值）
        ie_obs_i : np.ndarray
            时段i的观测蒸发指数
        omega_i : float
            时段i拟合的ω参数
        ia_i_plus_1 : np.ndarray
            时段i+1的干旱指数
        ie_obs_i_plus_1 : np.ndarray
            时段i+1的观测蒸发指数
        period_pair : str
            时段对名称，如'Δ1-2'
            
        Returns
        -------
        DeviationDistribution
            偏差分布对象
        """
        from .curves import BudykoCurves
        
        # 使用时段i的ω计算时段i+1的预期IE
        ie_expected_i_plus_1 = BudykoCurves.tixeront_fu(ia_i_plus_1, omega_i)
        
        # 计算年度偏差 εIEω = IE,obs - IE,expected
        epsilon = ie_obs_i_plus_1 - ie_expected_i_plus_1
        
        # Compute percentiles once (used by both IQR and skew fitting)
        percentiles = np.percentile(epsilon, [25, 50, 75])
        
        # 统计量 - optimized to reduce redundant calculations
        distribution = DeviationDistribution(
            period_name=period_pair,
            annual_deviations=epsilon,
            median=percentiles[1],  # Use precomputed median
            mean=np.mean(epsilon),
            std=np.std(epsilon, ddof=1),
            iqr=percentiles[2] - percentiles[0],  # Use precomputed quartiles
            skew=self._fast_skew(epsilon),  # Use optimized skewness calculation
            fitted_params=self._fit_skew_normal(epsilon)
        )
        
        self.distributions[period_pair] = distribution
        return distribution
    
    @staticmethod
    def _fast_skew(data: np.ndarray) -> float:
        """
        Fast skewness calculation using numpy operations
        
        Skewness = E[(X - μ)³] / σ³
        
        This is faster than scipy.stats.skew() for small datasets
        by avoiding unnecessary overhead and using optimized numpy operations.
        Uses the same default bias as scipy (bias=True, no correction).
        
        Parameters
        ----------
        data : np.ndarray
            Input data
            
        Returns
        -------
        float
            Sample skewness (no bias correction to match scipy.stats.skew default)
        """
        n = len(data)
        if n < 2:
            return 0.0
        
        # Compute mean and std in one pass
        mean = np.mean(data)
        centered = data - mean
        
        # Compute moments
        m2 = np.mean(centered ** 2)
        m3 = np.mean(centered ** 3)
        
        # Sample skewness (no bias correction, matching scipy default)
        if m2 == 0:
            return 0.0
        
        return m3 / (m2 ** 1.5)
    
    @staticmethod
    def _fit_skew_normal(data: np.ndarray) -> Dict:
        """
        拟合偏态正态分布（Step 3）
        
        PDF: f(x) = (2/λ) φ((x-ξ)/λ) Φ(α(x-ξ)/λ)
        
        Returns
        -------
        dict
            {'location': ξ, 'scale': λ, 'shape': α}
        """
        # 使用scipy拟合
        shape, location, scale = stats.skewnorm.fit(data)
        
        return {
            'location': location,  # ξ
            'scale': scale,        # λ
            'shape': shape,        # α
            'mean': location + scale * shape / np.sqrt(1 + shape**2) * np.sqrt(2/np.pi),
            'std': scale * np.sqrt(1 - 2*shape**2 / ((1+shape**2)*np.pi))
        }
    
    def wilcoxon_test(self, 
                     distribution: DeviationDistribution,
                     alpha: float = 0.05) -> Dict:
        """
        Wilcoxon符号秩检验：检验中位数是否显著异于0
        
        H0: median(εIEω) = 0
        
        Returns
        -------
        dict
            {'statistic': float, 'p_value': float, 'significant': bool}
        """
        statistic, p_value = stats.wilcoxon(
            distribution.annual_deviations,
            alternative='two-sided'
        )
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value <= alpha,
            'median': distribution.median
        }


class TemporalStability:
    """时间稳定性分析（Step 4）"""
    
    CATEGORIES = ['Stable', 'Variable', 'Alternating', 'Shift']
    
    def __init__(self):
        self.catchment_categories = {}
    
    def assess_stability(self,
                        distributions: List[DeviationDistribution],
                        ie_means: List[float]) -> Dict:
        """
        评估多个时段的偏差分布时间稳定性
        
        Parameters
        ----------
        distributions : List[DeviationDistribution]
            按时间顺序的偏差分布列表（最多4个）
        ie_means : List[float]
            对应时段的平均IE值
            
        Returns
        -------
        dict
            分类结果及详细信息
        """
        n_periods = len(distributions)
        
        # Sub-step 4.1: Kolmogorov-Smirnov检验连续分布对
        ks_results = self._ks_tests(distributions)
        
        # Sub-step 4.2: 检查系统性趋势
        median_sequence = [d.median for d in distributions]
        has_systematic_shift = self._check_systematic_shift(median_sequence)
        
        # Sub-step 4.3: 检查与IE的关系
        has_ie_dependency = self._check_ie_dependency(median_sequence, ie_means)
        
        # 分类逻辑（按Table 2）
        category = self._classify_stability(
            ks_results, 
            has_systematic_shift, 
            has_ie_dependency,
            n_periods
        )
        
        return {
            'category': category,
            'n_periods': n_periods,
            'ks_results': ks_results,
            'systematic_shift': has_systematic_shift,
            'ie_dependency': has_ie_dependency,
            'median_sequence': median_sequence,
            'sequence_notation': self._get_sequence_notation(ks_results, median_sequence)
        }
    
    def _ks_tests(self, distributions: List[DeviationDistribution]) -> List[Dict]:
        """连续分布对的KS检验"""
        results = []
        for i in range(len(distributions) - 1):
            d1 = distributions[i].annual_deviations
            d2 = distributions[i+1].annual_deviations
            
            statistic, p_value = stats.ks_2samp(d1, d2)
            
            results.append({
                'pair': f"{distributions[i].period_name}-{distributions[i+1].period_name}",
                'statistic': statistic,
                'p_value': p_value,
                'significant_diff': p_value <= 0.05,
                'symbol': '○' if p_value > 0.05 else ('+'  if distributions[i+1].median > distributions[i].median else '−')
            })
        
        return results
    
    def _check_systematic_shift(self, median_sequence: List[float]) -> bool:
        """检查是否存在系统性单向变化"""
        if len(median_sequence) < 3:
            return False
        
        # 计算差分符号
        diffs = np.diff(median_sequence)
        signs = np.sign(diffs)
        
        # 如果3个或更多连续差分同号 → 系统性变化
        positive_count = np.sum(signs > 0)
        negative_count = np.sum(signs < 0)
        
        return (positive_count >= 3) or (negative_count >= 3)
    
    def _check_ie_dependency(self, 
                            median_sequence: List[float],
                            ie_means: List[float]) -> bool:
        """检查中位偏差与IE的依赖关系"""
        if len(median_sequence) < 3:
            return False
        
        # 简单相关性检验
        if len(median_sequence) >= 3:
            corr, p_value = stats.spearmanr(ie_means, median_sequence)
            return abs(corr) > 0.6 and p_value < 0.1
        
        return False
    
    def _classify_stability(self,
                           ks_results: List[Dict],
                           systematic_shift: bool,
                           ie_dependency: bool,
                           n_periods: int) -> str:
        """
        根据Table 2的决策树分类
        """
        # 统计显著差异的比例
        n_significant = sum(1 for r in ks_results if r['significant_diff'])
        significance_ratio = n_significant / len(ks_results) if ks_results else 0
        
        # Stable: 超过一半的分布对无显著差异
        if significance_ratio < 0.5:
            return 'Stable'
        
        # Shift: 有系统性单向变化
        if systematic_shift:
            return 'Shift'
        
        # Alternating: 有与IE的依赖关系
        if ie_dependency:
            return 'Alternating'
        
        # Variable: 其他情况
        return 'Variable'
    
    def _get_sequence_notation(self,
                              ks_results: List[Dict],
                              median_sequence: List[float]) -> str:
        """
        生成序列符号表示，如 '○ ○ ○ ○' 或 '− + + +'
        """
        return ' '.join([r['symbol'] for r in ks_results])


class MarginalDistribution:
    """边际分布聚合（Step 5）"""
    
    def __init__(self):
        self.marginal_params = {}
    
    def aggregate_distributions(self,
                               distributions: List[DeviationDistribution],
                               stability_category: str) -> Dict:
        """
        聚合多个时段的偏差分布为一个边际分布
        
        Parameters
        ----------
        distributions : List[DeviationDistribution]
            所有时段的偏差分布
        stability_category : str
            稳定性分类
            
        Returns
        -------
        dict
            边际分布参数及统计量
        """
        # 合并所有年度偏差
        all_deviations = np.concatenate([
            d.annual_deviations for d in distributions
        ])
        
        # 拟合参数分布
        shape, location, scale = stats.skewnorm.fit(all_deviations)
        
        # 计算统计量
        marginal = {
            'n_total': len(all_deviations),
            'n_periods': len(distributions),
            'median': np.median(all_deviations),
            'mean': np.mean(all_deviations),
            'std': np.std(all_deviations, ddof=1),
            'iqr': np.percentile(all_deviations, 75) - np.percentile(all_deviations, 25),
            'percentile_10': np.percentile(all_deviations, 10),
            'percentile_90': np.percentile(all_deviations, 90),
            'fitted_location': location,
            'fitted_scale': scale,
            'fitted_shape': shape,
            'stability_category': stability_category,
            'predictive_power': self._assess_predictive_power(stability_category)
        }
        
        return marginal
    
    @staticmethod
    def _assess_predictive_power(category: str) -> str:
        """评估预测能力"""
        power_map = {
            'Stable': 'High',
            'Variable': 'Moderate',
            'Alternating': 'Low',
            'Shift': 'Low'
        }
        return power_map.get(category, 'Unknown')