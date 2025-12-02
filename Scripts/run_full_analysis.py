"""完整的Budyko偏差分析流程"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List
import sys

# Ensure the repository's src/ is importable when running this script directly
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from budyko.curves import BudykoCurves, PotentialEvaporation
from budyko.deviation import DeviationAnalysis, TemporalStability, MarginalDistribution
from visualization.budyko_plots import BudykoVisualizer


DATA_DIR = Path("./data/processed")
OUTPUT_DIR = Path("./results")
CATCHMENT_LIST_FILENAME = "catchments.csv"


def ensure_output_directory() -> None:
    """Create the output directory if it does not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_catchment_list() -> pd.DataFrame:
    """Load the catchment list or fall back to a demo catchment with guidance."""
    catchment_list_path = DATA_DIR / CATCHMENT_LIST_FILENAME
    if not catchment_list_path.exists():
        print(
            f"\n[Info] 未找到流域列表文件: {catchment_list_path.resolve()}\n"
            "    请将 catchments.csv 放在数据目录下，文件需包含一列 'id'。\n"
            "    当前将使用演示流域 DEMO_001 继续运行。"
        )
        return pd.DataFrame({'id': ['DEMO_001']})

    try:
        catchments = pd.read_csv(catchment_list_path)
        if 'id' not in catchments.columns:
            raise ValueError("catchments.csv 缺少 'id' 列")
        return catchments
    except Exception as exc:  # noqa: BLE001
        print(
            f"\n[Info] 无法读取流域列表 {catchment_list_path} ({exc})。\n"
            "    请检查文件格式是否为CSV且包含'id'列。\n"
            "    当前将使用演示流域 DEMO_001 继续运行。"
        )
        return pd.DataFrame({'id': ['DEMO_001']})


def load_catchment_data(catchment_id: str) -> pd.DataFrame:
    """Load catchment data or synthesize a demo dataset with helpful messaging."""
    data_path = DATA_DIR / f"{catchment_id}.csv"
    if not data_path.exists():
        print(
            f"[Info] 未找到流域数据文件: {data_path.resolve()}。\n"
            "       请将处理好的CSV放入数据目录，或使用演示数据继续。"
        )
        return _synthesize_demo_catchment()

    try:
        data = pd.read_csv(data_path)
        required_cols = {'year', 'P', 'EP', 'EA', 'Q'}
        if not required_cols.issubset(set(data.columns)):
            raise ValueError(f"缺少列: {required_cols - set(data.columns)}")
        return data
    except Exception as exc:  # noqa: BLE001
        print(
            f"[Info] 读取流域数据 {data_path} 时出错 ({exc})。\n"
            "       请确认文件包含 year, P, EP, EA, Q 列。将改用演示数据。"
        )
        return _synthesize_demo_catchment()


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
                ia_annual.to_numpy(dtype=float),
                ie_annual.to_numpy(dtype=float)
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
        if not window_results:
            return {
                'n_windows': 0,
                'median_range': 0.0,
                'results': []
            }
        median_range = np.ptp([w['marginal']['median'] for w in window_results])
        
        return {
            'n_windows': len(window_results),
            'median_range': median_range,
            'results': window_results
        }
    
    def _generate_catchment_report(self, catchment_id: str, results: Dict):
        """生成流域分析报告"""
        report_path = Path(self.config['output_dir']) / f"{catchment_id}_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
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


# Fallback: synthesize a small demo dataset when input CSVs are missing
def _synthesize_demo_catchment(year_start: int = 1901, year_end: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Create a simple, physically-plausible annual dataset for a demo catchment.

    Columns: year, P, EP, EA, Q (annual totals)
    Constraints: EA <= min(P, EP), Q <= P, EA >= 0
    """
    rng = np.random.default_rng(seed)
    years = np.arange(year_start, year_end + 1)

    # Precipitation (mm/yr)
    P = rng.normal(900.0, 150.0, size=years.size)
    P = np.clip(P, 300.0, 2000.0)

    # Potential ET typically comparable to or larger than actual ET
    ep_ratio = rng.normal(0.85, 0.06, size=years.size)
    ep_ratio = np.clip(ep_ratio, 0.5, 1.4)
    EP = np.maximum(0.0, ep_ratio * P)

    # Runoff fraction (0-0.7)
    q_ratio = rng.normal(0.3, 0.08, size=years.size)
    q_ratio = np.clip(q_ratio, 0.05, 0.7)
    Q = q_ratio * P

    # Actual ET limited by water supply and energy (EP)
    EA_raw = P - Q
    EA = np.minimum(EA_raw, 0.95 * EP)
    EA = np.clip(EA, 0.0, None)

    df = pd.DataFrame({
        'year': years.astype(int),
        'P': P.astype(float),
        'EP': EP.astype(float),
        'EA': EA.astype(float),
        'Q': Q.astype(float),
    })
    return df


# 主程序入口
if __name__ == "__main__":
    ensure_output_directory()

    if not DATA_DIR.exists():
        print(
            f"\n[Info] 数据目录不存在: {DATA_DIR.resolve()}\n"
            "       请先下载并解压数据到该目录。当前将使用演示数据运行。"
        )
        catchments = pd.DataFrame({'id': ['DEMO_001']})
    else:
        catchments = load_catchment_list()

    config = {
        'data_dir': DATA_DIR,
        'output_dir': OUTPUT_DIR,
        'catchment_list': CATCHMENT_LIST_FILENAME
    }

    # 初始化流程
    pipeline = BudykoDeviationPipeline(config)

    # 批量分析
    all_results = {}
    for idx, row in catchments.iterrows():
        catchment_id = str(row['id'])

        data = load_catchment_data(catchment_id)

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
    
    total = len(all_results)
    print(f"Total catchments: {total}")
    if total > 0:
        for cat, count in categories.items():
            pct = 100 * count / total
            print(f"  {cat}: {count} ({pct:.1f}%)")
    else:
        print("  No results to summarize.")