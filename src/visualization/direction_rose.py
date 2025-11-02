# src/visualization/direction_rose.py
"""
Budyko轨迹方向玫瑰图（Jaramillo风格）
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.projections import PolarAxes

def plot_direction_rose(movements_df: pd.DataFrame,
                       title: str = "Budyko Trajectory Directions",
                       ax: PolarAxes = None,
                       n_bins: int = 16) -> PolarAxes:
    """
    绘制运动方向玫瑰图
    
    Parameters
    ----------
    movements_df : pd.DataFrame
        包含'direction_angle'和'intensity'的运动数据
    title : str
        图标题
    ax : PolarAxes, optional
        极坐标轴
    n_bins : int
        方向分bin数
        
    Returns
    -------
    PolarAxes
        绘图轴
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
    
    # 定义方向bins
    bins = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    
    # 统计每个方向的频率
    hist, _ = np.histogram(movements_df['direction_angle'], bins=bins)
    frequencies = hist / len(movements_df) * 100  # 转换为百分比
    
    # 计算每个bin的平均强度
    movements_df['direction_bin'] = pd.cut(
        movements_df['direction_angle'],
        bins=bins,
        labels=bin_centers,
        include_lowest=True
    )
    avg_intensity = movements_df.groupby('direction_bin')['intensity'].mean()
    
    # 颜色映射（根据强度）
    colors = plt.cm.viridis(avg_intensity.values / avg_intensity.max())
    
    # 绘制玫瑰图
    theta = np.deg2rad(bin_centers)
    ax.bar(theta, frequencies, width=np.deg2rad(bin_width),
          bottom=0, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # 标记"遵循曲线"的区域
    follow_ranges = [(45, 90), (225, 270)]
    for angle_min, angle_max in follow_ranges:
        theta_fill = np.deg2rad(np.linspace(angle_min, angle_max, 100))
        r_fill = np.full_like(theta_fill, frequencies.max() * 1.1)
        ax.fill_between(theta_fill, 0, r_fill, alpha=0.15, color='green', 
                        label='Following curve' if angle_min == 45 else '')
    
    # 设置方向标签
    ax.set_theta_zero_location('N')  # 0°在上方
    ax.set_theta_direction(1)  # 顺时针
    
    # 方向标签
    direction_labels = ['N\n(ΔIE)', 'NE', 'E\n(ΔIA)', 'SE',
                       'S', 'SW', 'W', 'NW']
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(direction_labels, fontsize=12)
    
    # 径向标签
    ax.set_ylabel('Percentage (%)', fontsize=12, labelpad=30)
    ax.set_ylim(0, frequencies.max() * 1.2)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True, alpha=0.3)
    
    return ax