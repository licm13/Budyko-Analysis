# scripts/run_full_analysis.py
"""
完整的Budyko偏差分析流程
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List

from src.budyko.curves import BudykoCurves, PotentialEvaporation
from src.budyko.deviation import DeviationAnalysis, TemporalStability, MarginalDistribution
from src.data_processing.water_balance import WaterBalance
from src.visualization.budyko_plots import BudykoVisualizer


class BudykoDeviationPipeline:
    """Budyko偏差分析完整流程"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        
        # 定义时段
        self.periods = [
            ('T1', 1901, 1920),
            ('T2', 1921, 1940),
            ('T3', 1941, 1960),
            ('T4', 1961, 1980),
            ('T5', 1981, 2000)
        ]
    
    def run_catchment_analysis(self, 
                               catchment_id: str,
                               data: pd.DataFrame) -> Dict:
        """
        对单个流域运行完整的6步分析流程
        
        Parameters
        ----------
        catchment_id : str
            流域ID
        data : pd.DataFrame
            包含P, EP, Q, EA列的日度数据框
            
        Returns
        -------
        dict
            完整分析结果
        """
        print(f"\n{'='*60}")
        print(f"Analyzing catchment: {catchment_id}")
        print(f"{'='*60}")
        
        results = {
            'catchment_id': catchment_id,
            'periods': {},
            'deviations': {},
            'stability': {},
            'marginal': {}
        }
        
        # Step 1: 估计每个时段的IE曲线
        print("\n[Step 1] Estimating catchment-specific curves...")
        period_results = self._step1_estimate_curves(data)
        results['periods'] = period_results
        
        # Step 2: 计算连续时段间的偏差分布
        print("\n[Step 2] Calculating deviation distributions...")
        deviation_dists = self._step2_calculate_deviations(period_results)
        results['deviations'] = deviation_dists
        
        # Step 3: 拟合参数分布（已在Step 2中完成）
        print("\n[Step 3] Fitting parametric distributions... (Done in Step 2)")
        
        # Step 4: 评估时间稳定性
        print("\n[Step 4] Evaluating temporal stability...")
        stability = self._step4_temporal_stability(deviation_dists, period_results)
        results['stability'] = stability
        
        # Step 5: 聚合边际分布
        print("\n[Step 5] Aggregating marginal distribution...")
        marginal = self._step5_marginal_distribution(deviation_dists, stability)
        results['marginal'] = marginal
        
        # Step 6: 敏感性分析（滑动窗口）
        print("\n[Step 6] Sensitivity analysis with moving windows...")
        sensitivity = self._step6_sensitivity_analysis(data)
        results['sensitivity'] = sensitivity
        
        # 生成报告
        self._generate_catchment_report(catchment_id, results)
        
        return results
    
    def _step1_estimate_curves(self, data: pd.DataFrame) -> Dict:
        """Step 1: 估计各时段的ω参数"""
        period_results = {}
        
        for period_name, start_year, end_year in self.periods:
            # 提取时段数据
            mask = (data['year'] >= start_year) & (data['year'] <= end_year)
            period_data = data[mask]
            
            # 计算年度IA和IE
            annual = period_data.groupby('year').agg({
                'P': 'sum',
                'EP': 'sum',
                'EA': 'sum',
                'Q': 'sum'
            })
            
            ia_annual = annual['EP'] / annual['P']
            ie_annual = annual['EA'] / annual['P']
            
            # 拟合ω
            omega, fit_stats = BudykoCurves.fit_omega(
                ia_annual.values, 
                ie_annual.values
            )
            
            period_results[period_name] = {
                'start_year': start_year,
                'end_year': end_year,
                'n_years': len(annual),
                'omega': omega,
                'fit_stats': fit_stats,
                'ia_annual': ia_annual.values,
                'ie_annual': ie_annual.values,
                'ia_mean': ia_annual.mean(),
                'ie_mean': ie_annual.mean()
            }
            
            print(f"  {period_name} ({start_year}-{end_year}): "
                  f"ω={omega:.3f}, RMSE={fit_stats['rmse']:.4f}")
        
        return period_results
    
    def _step2_calculate_deviations(self, period_results: Dict) -> Dict:
        """Step 2: 计算偏差分布"""
        deviation_analysis = DeviationAnalysis()
        deviation_dists = {}
        
        period_names = ['T1', 'T2', 'T3', 'T4', 'T5']
        
        for i in range(len(period_names) - 1):
            ti = period_names[i]
            ti_plus_1 = period_names[i + 1]
            
            if ti not in period_results or ti_plus_1 not in period_results:
                continue
            
            pair_name = f"Δ{i+1}-{i+2}"
            
            dist = deviation_analysis.calculate_deviations(
                ia_i=period_results[ti]['ia_annual'],
                ie_obs_i=period_results[ti]['ie_annual'],
                omega_i=period_results[ti]['omega'],
                ia_i_plus_1=period_results[ti_plus_1]['ia_annual'],
                ie_obs_i_plus_1=period_results[ti_plus_1]['ie_annual'],
                period_pair=pair_name
            )
            
            # Wilcoxon检验
            wilcoxon = deviation_analysis.wilcoxon_test(dist)
            
            deviation_dists[pair_name] = {
                'distribution': dist,
                'wilcoxon_test': wilcoxon
            }
            
            print(f"  {pair_name}: median εIEω={dist.median:.4f}, "
                  f"p={wilcoxon['p_value']:.4f}, "
                  f"significant={'Yes' if wilcoxon['significant'] else 'No'}")
        
        return deviation_dists
    
    def _step4_temporal_stability(self, 
                                  deviation_dists: Dict,
                                  period_results: Dict) -> Dict:
        """Step 4: 时间稳定性评估"""
        stability_analyzer = TemporalStability()
        
        # 提取分布列表
        distributions = [v['distribution'] for v in deviation_dists.values()]
        
        # 提取对应时段的平均IE
        period_names = ['T2', 'T3', 'T4', 'T5']  # 对应Δ1-2, Δ2-3, Δ3-4, Δ4-5
        ie_means = [period_results[p]['ie_mean'] 
                   for p in period_names if p in period_results]
        
        # 评估稳定性
        stability = stability_analyzer.assess_stability(
            distributions[:len(ie_means)],
            ie_means
        )
        
        print(f"  Temporal stability category: {stability['category']}")
        print(f"  Sequence notation: {stability['sequence_notation']}")
        
        return stability
    
    def _step5_marginal_distribution(self,
                                     deviation_dists: Dict,
                                     stability: Dict) -> Dict:
        """Step 5: 边际分布聚合"""
        marginal_analyzer = MarginalDistribution()
        
        distributions = [v['distribution'] for v in deviation_dists.values()]
        
        marginal = marginal_analyzer.aggregate_distributions(
            distributions,
            stability['category']
        )
        
        print(f"  Marginal distribution:")
        print(f"    Median εIEω: {marginal['median']:.4f}")
        print(f"    IQR: {marginal['iqr']:.4f}")
        print(f"    Predictive power: {marginal['predictive_power']}")
        
        return marginal
    
    def _step6_sensitivity_analysis(self, data: pd.DataFrame) -> Dict:
        """Step 6: 滑动窗口敏感性分析"""
        n_windows = 20
        window_results = []
        
        for shift in range(n_windows):
            # 调整时段起始年
            adjusted_periods = [
                (name, start + shift, end + shift)
                for name, start, end in self.periods
            ]
            
            # 运行简化分析（只计算边际分布）
            try:
                # ... 重复Step 1-5但使用调整后的时段
                # 这里省略详细代码，与上面类似
                pass
            except Exception as e:
                continue
        
        # 分析窗口间的变异
        median_range = np.ptp([w['marginal']['median'] for w in window_results])
        
        return {
            'n_windows': len(window_results),
            'median_range': median_range,
            'results': window_results
        }
    
    def _generate_catchment_report(self, catchment_id: str, results: Dict):
        """生成流域分析报告"""
        report_path = Path(self.config['output_dir']) / f"{catchment_id}_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Budyko Deviation Analysis Report\n")
            f.write(f"{'='*60}\n")
            f.write(f"Catchment ID: {catchment_id}\n\n")
            
            # 时段信息
            f.write("Period-specific parameters:\n")
            for period, data in results['periods'].items():
                f.write(f"  {period}: ω={data['omega']:.3f}, "
                       f"IA={data['ia_mean']:.3f}, IE={data['ie_mean']:.3f}\n")
            
            # 稳定性分类
            f.write(f"\nTemporal stability: {results['stability']['category']}\n")
            
            # 边际分布
            marg = results['marginal']
            f.write(f"\nMarginal distribution:\n")
            f.write(f"  Median εIEω: {marg['median']:.4f}\n")
            f.write(f"  IQR: {marg['iqr']:.4f}\n")
            f.write(f"  Predictive power: {marg['predictive_power']}\n")
        
        print(f"\n  Report saved to: {report_path}")


# 主程序入口
if __name__ == "__main__":
    # 配置
    config = {
        'data_dir': './data/processed',
        'output_dir': './results',
        'catchment_list': 'catchments.csv'
    }
    
    # 初始化流程
    pipeline = BudykoDeviationPipeline(config)
    
    # 读取流域列表
    catchments = pd.read_csv(Path(config['data_dir']) / config['catchment_list'])
    
    # 批量分析
    all_results = {}
    for idx, row in catchments.iterrows():
        catchment_id = row['id']
        
        # 加载数据
        data_path = Path(config['data_dir']) / f"{catchment_id}.csv"
        data = pd.read_csv(data_path)
        
        # 运行分析
        result = pipeline.run_catchment_analysis(catchment_id, data)
        all_results[catchment_id] = result
    
    # 全局汇总
    print("\n" + "="*60)
    print("Global Summary")
    print("="*60)
    
    categories = {'Stable': 0, 'Variable': 0, 'Alternating': 0, 'Shift': 0}
    for res in all_results.values():
        cat = res['stability']['category']
        categories[cat] += 1
    
    print(f"Total catchments: {len(all_results)}")
    for cat, count in categories.items():
        pct = 100 * count / len(all_results)
        print(f"  {cat}: {count} ({pct:.1f}%)")