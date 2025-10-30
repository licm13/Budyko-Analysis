# src/visualization/budyko_plots.py
"""
Budyko空间可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

class BudykoVisualizer:
    """Budyko空间可视化类"""
    
    @staticmethod
    def plot_catchment_trajectory(ax, period_results: Dict, 
                                  omega_reference: float = None):
        """
        绘制流域在Budyko空间的轨迹（类似Fig. 2和Fig. 6）
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            绘图轴
        period_results : Dict
            各时段结果
        omega_reference : float
            参考ω值（如T1的ω），用于绘制预期曲线
        """
        from ..budyko.curves import BudykoCurves
        
        # 绘制Budyko (1974) 非参数曲线
        ia_range = np.linspace(0, 5, 200)
        ie_budyko = BudykoCurves.budyko_1948(ia_range)
        ax.plot(ia_range, ie_budyko, 'k--', lw=1.5, 
               label='Budyko (1974)', alpha=0.6)
        
        # 绘制能量和水量限制线
        ax.plot([0, 5], [0, 1], 'k:', lw=1, alpha=0.4, label='Energy limit')
        ax.plot([0, 5], [1, 1], 'k:', lw=1, alpha=0.4, label='Water limit')
        
        # 颜色映射
        colors = plt.cm.viridis(np.linspace(0, 1, len(period_results)))
        
        # 绘制每个时段
        for (period_name, data), color in zip(period_results.items(), colors):
            # 年度点（浅色小点）
            ax.scatter(data['ia_annual'], data['ie_annual'],
                      c=[color], alpha=0.3, s=20, edgecolors='none')
            
            # 时段平均点（深色大点）
            ax.scatter(data['ia_mean'], data['ie_mean'],
                      c=[color], s=100, edgecolors='k', lw=1.5,
                      marker='o', label=f"{period_name} ({data['start_year']}-{data['end_year']})",
                      zorder=5)
            
            # 绘制该时段的参数曲线
            ie_curve = BudykoCurves.tixeront_fu(ia_range, data['omega'])
            ax.plot(ia_range, ie_curve, color=color, lw=1.5, alpha=0.7)
        
        # 如果提供参考ω，绘制预期轨迹
        if omega_reference is not None:
            ie_expected = BudykoCurves.tixeront_fu(ia_range, omega_reference)
            ax.plot(ia_range, ie_expected, 'r--', lw=2, 
                   label=f'Expected (ω={omega_reference:.2f})', alpha=0.8)
        
        ax.set_xlabel('Aridity Index $I_A = E_P/P$ [-]', fontsize=12)
        ax.set_ylabel('Evaporative Index $I_E = E_A/P$ [-]', fontsize=12)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.set_title('Catchment Movement in Budyko Space', fontsize=13, fontweight='bold')
    
    @staticmethod
    def plot_deviation_distributions(fig, deviation_dists: Dict):
        """
        绘制偏差分布（类似Fig. 6 中间列）
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            图形对象
        deviation_dists : Dict
            偏差分布字典
        """
        n_dists = len(deviation_dists)
        
        for i, (pair_name, data) in enumerate(deviation_dists.items()):
            ax = fig.add_subplot(n_dists, 1, i+1)
            
            dist = data['distribution']
            epsilon = dist.annual_deviations
            
            # 直方图
            ax.hist(epsilon, bins=15, density=True, alpha=0.6, 
                   color='steelblue', edgecolor='k', lw=0.5)
            
            # 拟合曲线
            from scipy import stats
            x = np.linspace(epsilon.min(), epsilon.max(), 200)
            fitted_pdf = stats.skewnorm.pdf(
                x,
                dist.fitted_params['shape'],
                dist.fitted_params['location'],
                dist.fitted_params['scale']
            )
            ax.plot(x, fitted_pdf, 'r-', lw=2, label='Fitted skew-normal')
            
            # 中位数线
            ax.axvline(dist.median, color='darkred', linestyle='--', lw=2,
                      label=f'Median={dist.median:.4f}')
            
            # Wilcoxon检验结果
            wilcoxon = data['wilcoxon_test']
            sig_text = "***" if wilcoxon['p_value'] < 0.001 else \
                      "**" if wilcoxon['p_value'] < 0.01 else \
                      "*" if wilcoxon['p_value'] < 0.05 else "ns"
            
            ax.text(0.02, 0.95, f"$p$={wilcoxon['p_value']:.4f} {sig_text}",
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('Yearly $\\epsilon_{IE\\omega}$ [-]', fontsize=10)
            ax.set_ylabel('Probability density [-]', fontsize=10)
            ax.set_title(f'{pair_name}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        
        fig.tight_layout()
    
    @staticmethod
    def plot_marginal_distribution(ax, marginal: Dict):
        """
        绘制边际分布（类似Fig. 6 右列）
        """
        from scipy import stats
        
        # 生成分布范围
        median = marginal['median']
        std = marginal['std']
        x = np.linspace(median - 4*std, median + 4*std, 200)
        
        # PDF
        pdf = stats.skewnorm.pdf(
            x,
            marginal['fitted_shape'],
            marginal['fitted_location'],
            marginal['fitted_scale']
        )
        
        ax.fill_between(x, pdf, alpha=0.3, color='gray', label='Marginal distribution')
        ax.plot(x, pdf, 'k-', lw=2)
        
        # 中位数
        ax.axvline(median, color='darkred', linestyle='--', lw=2.5,
                  label=f'Median={median:.4f}')
        
        # IQR阴影
        p25 = marginal['median'] - marginal['iqr']/2
        p75 = marginal['median'] + marginal['iqr']/2
        ax.axvspan(p25, p75, alpha=0.2, color='blue', 
                  label=f'IQR={marginal["iqr"]:.3f}')
        
        # 注释
        text = f"Stability: {marginal['stability_category']}\n" \
               f"Power: {marginal['predictive_power']}\n" \
               f"$n$={marginal['n_total']}"
        ax.text(0.98, 0.95, text, transform=ax.transAxes,
               va='top', ha='right', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax.set_xlabel('Yearly $\\epsilon_{IE\\omega}$ [-]', fontsize=12)
        ax.set_ylabel('Probability density [-]', fontsize=12)
        ax.set_title('Aggregated Marginal Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)


def create_comprehensive_figure(catchment_id: str, results: Dict, output_path: str):
    """
    创建类似Fig. 6的综合图（左：Budyko空间，中：分布序列，右：边际分布）
    """
    fig = plt.figure(figsize=(18, 6))
    
    # 左：Budyko轨迹
    ax_budyko = fig.add_subplot(131)
    BudykoVisualizer.plot_catchment_trajectory(
        ax_budyko, 
        results['periods'],
        omega_reference=results['periods']['T1']['omega']
    )
    
    # 中：偏差分布序列
    # （这里简化为子图网格）
    n_dists = len(results['deviations'])
    for i, (pair_name, data) in enumerate(results['deviations'].items()):
        ax_dist = fig.add_subplot(n_dists, 3, 3*(i+1) + 2)
        
        dist = data['distribution']
        ax_dist.hist(dist.annual_deviations, bins=10, density=True,
                    alpha=0.6, edgecolor='k')
        ax_dist.axvline(dist.median, color='r', linestyle='--', lw=2)
        ax_dist.set_title(f'{pair_name}', fontsize=9)
        ax_dist.tick_params(labelsize=8)
    
    # 右：边际分布
    ax_marginal = fig.add_subplot(133)
    BudykoVisualizer.plot_marginal_distribution(ax_marginal, results['marginal'])
    
    fig.suptitle(f'Budyko Deviation Analysis: {catchment_id}', 
                fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved: {output_path}")