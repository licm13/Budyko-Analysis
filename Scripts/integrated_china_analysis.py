# scripts/integrated_china_analysis.py
"""
中国上万小流域的整合Budyko分析

结合Ibrahim和Jaramillo两种方法
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from budyko.curves import BudykoCurves
from budyko.deviation_ibrahim import DeviationAnalysis, TemporalStability
from budyko.trajectory_jaramillo import TrajectoryAnalyzer, ScenarioComparator
from data_processing.cmip6_processor import CMIP6Processor
from utils.parallel_processing import ParallelBudykoAnalyzer
from visualization.china_maps import ChinaMapVisualizer


class IntegratedChinaAnalysis:
    """
    中国流域整合分析流程
    
    步骤：
    1. 历史期（1981-2010）：Ibrahim方法分析偏差
    2. 未来期（2071-2100）：Jaramillo方法分析轨迹
    3. 情景比较：SSP1-2.6 vs SSP5-8.5
    4. 区域模式识别
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        
        # 定义中国典型区域
        self.china_regions = {
            'Northeast': ['Songhua', 'Liaohe'],
            'North': ['Haihe', 'Yellow River'],
            'Central': ['Yangtze', 'Huaihe'],
            'South': ['Pearl River', 'Southeast Rivers'],
            'Southwest': ['Upper Yangtze', 'Lancang'],
            'Northwest': ['Tarim', 'Heihe', 'Shiyang']
        }
    
    def step1_historical_deviation_analysis(self, catchments_data: pd.DataFrame):
        """
        步骤1：历史期偏差分析（Ibrahim方法）
        
        时段划分：
        - T1: 1981-2000
        - T2: 2001-2020
        """
        print("\n" + "="*60)
        print("STEP 1: Historical Deviation Analysis (Ibrahim et al. 2025)")
        print("="*60)
        
        deviation_analyzer = DeviationAnalysis(period_length=20)
        stability_analyzer = TemporalStability()
        
        # 并行处理
        parallel_analyzer = ParallelBudykoAnalyzer(n_processes=8)
        
        def analyze_catchment(catchment_id, **kwargs):
            """单个流域的历史分析"""
            row = kwargs['data'].loc[kwargs['data']['catchment_id'] == catchment_id].iloc[0]
            
            # 提取两个时期数据
            period_1_data = {
                'ia_annual': row['IA_annual_T1'],  # 假设已有年度数据
                'ie_annual': row['IE_annual_T1'],
                'omega': row['omega_T1']
            }
            
            period_2_data = {
                'ia_annual': row['IA_annual_T2'],
                'ie_annual': row['IE_annual_T2']
            }
            
            # 计算偏差分布
            dist = deviation_analyzer.calculate_deviations(
                ia_i=period_1_data['ia_annual'],
                ie_obs_i=period_1_data['ie_annual'],
                omega_i=period_1_data['omega'],
                ia_i_plus_1=period_2_data['ia_annual'],
                ie_obs_i_plus_1=period_2_data['ie_annual'],
                period_pair='T1-T2'
            )
            
            # Wilcoxon检验
            wilcoxon = deviation_analyzer.wilcoxon_test(dist)
            
            return {
                'median_deviation': dist.median,
                'iqr_deviation': dist.iqr,
                'p_value': wilcoxon['p_value'],
                'significant': wilcoxon['significant']
            }
        
        results = parallel_analyzer.process_catchments(
            catchment_ids=catchments_data['catchment_id'].tolist(),
            analysis_function=analyze_catchment,
            data=catchments_data
        )
        
        # 统计汇总
        n_total = len(results)
        n_not_deviate = (~results['significant']).sum()
        
        print(f"\nHistorical Analysis Results:")
        print(f"  Total catchments: {n_total}")
        print(f"  Not significantly deviating: {n_not_deviate} ({100*n_not_deviate/n_total:.1f}%)")
        print(f"  Median deviation range: [{results['median_deviation'].min():.4f}, {results['median_deviation'].max():.4f}]")
        
        self.results['historical'] = results
        return results
    
    def step2_future_trajectory_analysis(self, 
                                        cmip6_data: Dict[str, pd.DataFrame]):
        """
        步骤2：未来轨迹分析（Jaramillo方法）
        
        Parameters
        ----------
        cmip6_data : Dict[str, pd.DataFrame]
            {scenario: catchment_budyko_data}
        """
        print("\n" + "="*60)
        print("STEP 2: Future Trajectory Analysis (Jaramillo et al. 2022)")
        print("="*60)
        
        scenario_results = {}
        
        for scenario in ['ssp126', 'ssp585']:
            print(f"\n  Analyzing scenario: {scenario}")
            
            data = cmip6_data[scenario]
            
            # 并行轨迹分析
            parallel_analyzer = ParallelBudykoAnalyzer(n_processes=8)
            
            trajectory_results = parallel_analyzer.batch_trajectory_analysis(
                catchments_df=data,
                period_1_cols=('IA_historical', 'IE_historical'),
                period_2_cols=(f'IA_{scenario}', f'IE_{scenario}')
            )
            
            # 统计
            n_following = trajectory_results['follows_curve'].sum()
            n_total = len(trajectory_results)
            
            print(f"    Following curve: {n_following} ({100*n_following/n_total:.1f}%)")
            print(f"    Mean intensity: {trajectory_results['intensity'].mean():.4f}")
            
            scenario_results[scenario] = trajectory_results
        
        # 情景比较
        comparator = ScenarioComparator(scenarios=['ssp126', 'ssp585'])
        comparison = comparator.compare_following_rates(scenario_results)
        
        print(f"\n  Scenario Comparison:")
        print(comparison[['scenario', 'pct_following', 'pct_deviating', 'mean_intensity']])
        
        self.results['future'] = scenario_results
        self.results['scenario_comparison'] = comparison
        
        return scenario_results, comparison
    
    def step3_regional_patterns(self):
        """
        步骤3：区域模式分析
        """
        print("\n" + "="*60)
        print("STEP 3: Regional Pattern Analysis")
        print("="*60)
        
        # 这里添加区域聚类和模式识别代码
        # 略...
    
    def step4_integrated_visualization(self):
        """
        步骤4：整合可视化
        """
        print("\n" + "="*60)
        print("STEP 4: Creating Integrated Visualizations")
        print("="*60)
        
        # 创建综合图表
        fig = plt.figure(figsize=(20, 12))
        
        # 子图1：历史偏差地图
        ax1 = fig.add_subplot(2, 3, 1)
        # ... 绘图代码
        
        # 子图2：未来轨迹方向玫瑰图
        ax2 = fig.add_subplot(2, 3, 2, projection='polar')
        # ... 绘图代码
        
        # 子图3：情景对比
        ax3 = fig.add_subplot(2, 3, 3)
        # ... 绘图代码
        
        # 保存
        output_path = Path(self.config['output_dir']) / 'integrated_china_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved: {output_path}")
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("\n" + "="*70)
        print(" INTEGRATED BUDYKO ANALYSIS FOR 10,000+ CHINESE CATCHMENTS ")
        print("="*70)
        
        # 加载数据
        print("\nLoading data...")
        catchments_data = self._load_china_catchments()
        cmip6_data = self._load_cmip6_data()
        
        # 执行分析
        self.step1_historical_deviation_analysis(catchments_data)
        self.step2_future_trajectory_analysis(cmip6_data)
        self.step3_regional_patterns()
        self.step4_integrated_visualization()
        
        # 保存结果
        self._save_results()
        
        print("\n" + "="*70)
        print(" ANALYSIS COMPLETE ")
        print("="*70)
    
    def _load_china_catchments(self) -> pd.DataFrame:
        """加载中国流域数据"""
        # 实际实现中从数据库或文件加载
        print("  Loading Chinese catchment data...")
        return pd.DataFrame()  # 占位
    
    def _load_cmip6_data(self) -> Dict[str, pd.DataFrame]:
        """加载CMIP6数据"""
        print("  Loading CMIP6 scenario data...")
        return {}  # 占位
    
    def _save_results(self):
        """保存结果"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存各部分结果
        for key, df in self.results.items():
            if isinstance(df, pd.DataFrame):
                output_path = output_dir / f"{key}_results.csv"
                df.to_csv(output_path, index=False)
                print(f"  Saved: {output_path}")


if __name__ == "__main__":
    # 配置
    config = {
        'data_dir': './data/china',
        'output_dir': './results/china_integrated',
        'n_processes': 16,
        'cmip6_models': ['ACCESS-CM2', 'CNRM-CM6-1', 'EC-Earth3'],
        'scenarios': ['ssp126', 'ssp585']
    }
    
    # 运行分析
    analysis = IntegratedChinaAnalysis(config)
    analysis.run_full_analysis()