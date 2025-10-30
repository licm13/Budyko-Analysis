"""
Visualization Module for Budyko Framework
Budyko框架可视化工具
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, Dict

class BudykoVisualizer:
    """Budyko框架可视化器"""
    
    def __init__(self, style='seaborn-v0_8-whitegrid'):
        """初始化绘图风格"""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # 中文支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_budyko_space(self,
                         DI: np.ndarray,
                         EI: np.ndarray,
                         omega: float = 2.6,
                         color_by: Optional[np.ndarray] = None,
                         color_label: str = '',
                         title: str = 'Budyko空间',
                         figsize: Tuple[int, int] = (10, 8),
                         save_path: Optional[str] = None):
        """
        绘制Budyko空间图
        
        Parameters:
        -----------
        DI, EI : array
            干旱指数和蒸发指数
        omega : float
            Budyko参数
        color_by : array, optional
            用于着色的变量
        color_label : str
            颜色变量标签
        title : str
            图标题
        figsize : tuple
            图大小
        save_path : str, optional
            保存路径
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制数据点
        if color_by is not None:
            scatter = ax.scatter(DI, EI, c=color_by, cmap='viridis',
                               s=50, alpha=0.6, edgecolor='black', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_label, fontsize=12)
        else:
            ax.scatter(DI, EI, s=50, alpha=0.6, color='steelblue',
                      edgecolor='black', linewidth=0.5)
        
        # Budyko曲线
        DI_range = np.linspace(0, 5, 200)
        EI_budyko = 1 + DI_range - np.power(1 + np.power(DI_range, omega), 1/omega)
        ax.plot(DI_range, EI_budyko, 'r-', linewidth=2.5, 
                label=f'Budyko曲线 (ω={omega:.2f})', zorder=10)
        
        # 能量和水分限制边界
        ax.plot([0, 10], [0, 1], 'k--', linewidth=1.5, alpha=0.5, 
                label='能量限制')
        ax.plot([1, 10], [1, 1], 'k--', linewidth=1.5, alpha=0.5, 
                label='水分限制')
        ax.axvline(1, color='gray', linestyle=':', alpha=0.3)
        
        # 设置
        ax.set_xlabel('干旱指数 (DI = PET/P)', fontsize=13, fontweight='bold')
        ax.set_ylabel('蒸发指数 (EI = ET/P)', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlim(0, np.min([5, np.percentile(DI[np.isfinite(DI)], 99)]))
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加区域标注
        ax.text(0.3, 0.9, '能量限制区', fontsize=10, style='italic', alpha=0.6)
        ax.text(3, 0.5, '水分限制区', fontsize=10, style='italic', alpha=0.6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_3d_budyko(self,
                       DI: np.ndarray,
                       EI: np.ndarray,
                       SI: np.ndarray,
                       omega: float = 2.6,
                       figsize: Tuple[int, int] = (12, 9),
                       save_path: Optional[str] = None):
        """
        绘制三维Budyko空间
        
        Parameters:
        -----------
        DI, EI, SI : array
            三个维度的指数
        omega : float
            Budyko参数
        figsize : tuple
            图大小
        save_path : str, optional
            保存路径
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 数据点
        scatter = ax.scatter(DI, SI, EI, c=EI, cmap='coolwarm',
                           s=40, alpha=0.6, edgecolor='black', linewidth=0.3)
        
        # 理论Budyko曲面
        di_mesh = np.linspace(0, 5, 50)
        si_mesh = np.linspace(SI.min(), SI.max(), 50)
        DI_mesh, SI_mesh = np.meshgrid(di_mesh, si_mesh)
        
        EI_mesh = 1 + DI_mesh - np.power(1 + np.power(DI_mesh, omega), 1/omega)
        
        ax.plot_surface(DI_mesh, SI_mesh, EI_mesh, alpha=0.2, 
                       cmap='viridis', edgecolor='none')
        
        # 设置
        ax.set_xlabel('干旱指数 (DI)', fontsize=12, fontweight='bold')
        ax.set_ylabel('储量指数 (SI)', fontsize=12, fontweight='bold')
        ax.set_zlabel('蒸发指数 (EI)', fontsize=12, fontweight='bold')
        ax.set_title('三维Budyko框架', fontsize=14, fontweight='bold', pad=20)
        
        # 颜色条
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('蒸发指数 (EI)', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_seasonal_comparison(self,
                                wet_data: Dict,
                                dry_data: Dict,
                                figsize: Tuple[int, int] = (14, 6),
                                save_path: Optional[str] = None):
        """
        绘制季节对比图
        
        Parameters:
        -----------
        wet_data, dry_data : dict
            包含DI, EI, omega的字典
        figsize : tuple
            图大小
        save_path : str, optional
            保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        seasons = [
            ('湿季', wet_data, 'blue', axes[0]),
            ('干季', dry_data, 'red', axes[1])
        ]
        
        for season_name, data, color, ax in seasons:
            # 数据点
            scatter = ax.scatter(data['DI'], data['EI'], 
                               c=data.get('deviation', data['EI']),
                               cmap='RdBu_r', s=50, alpha=0.6,
                               edgecolor='black', linewidth=0.5)
            
            # Budyko曲线
            omega = data.get('omega', 2.6)
            DI_range = np.linspace(0, 5, 200)
            EI_budyko = 1 + DI_range - np.power(1 + np.power(DI_range, omega), 1/omega)
            ax.plot(DI_range, EI_budyko, color=color, linewidth=2.5, 
                   label=f'Budyko (ω={omega:.2f})', zorder=10)
            
            # 边界
            ax.plot([0, 5], [0, 1], 'k--', alpha=0.5)
            ax.plot([1, 5], [1, 1], 'k--', alpha=0.5)
            
            # 设置
            ax.set_xlabel('干旱指数 (DI)', fontsize=12, fontweight='bold')
            ax.set_ylabel('蒸发指数 (EI)', fontsize=12, fontweight='bold')
            ax.set_title(f'{season_name} Budyko空间', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 1.2)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, ax=ax, label='偏离/EI')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    def plot_di_si_relationships(self,
                                DI: np.ndarray,
                                SI: np.ndarray,
                                SCI: np.ndarray,
                                EI: np.ndarray,
                                figsize: Tuple[int, int] = (15, 5),
                                save_path: Optional[str] = None):
        """
        绘制DI/EI与SI/SCI关系图
        
        Parameters:
        -----------
        DI, SI, SCI, EI : array
            各指数
        figsize : tuple
            图大小
        save_path : str, optional
            保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # DI vs SI
        scatter1 = axes[0].scatter(DI, SI, c=EI, cmap='Blues',
                                  s=40, alpha=0.6, edgecolor='black', linewidth=0.3)
        axes[0].set_xlabel('干旱指数 (DI)', fontsize=12)
        axes[0].set_ylabel('储量指数 (SI)', fontsize=12)
        axes[0].set_title('DI vs SI', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='EI')
        
        # 添加趋势线
        z = np.polyfit(DI[np.isfinite(DI) & np.isfinite(SI)], 
                      SI[np.isfinite(DI) & np.isfinite(SI)], 1)
        p = np.poly1d(z)
        axes[0].plot(np.sort(DI[np.isfinite(DI)]), 
                    p(np.sort(DI[np.isfinite(DI)])),
                    "r--", alpha=0.8, linewidth=2, label=f'拟合: y={z[0]:.2f}x+{z[1]:.2f}')
        axes[0].legend(fontsize=9)
        
        # DI vs SCI
        scatter2 = axes[1].scatter(DI, SCI, c=EI, cmap='Greens',
                                  s=40, alpha=0.6, edgecolor='black', linewidth=0.3)
        axes[1].set_xlabel('干旱指数 (DI)', fontsize=12)
        axes[1].set_ylabel('储量变化指数 (SCI)', fontsize=12)
        axes[1].set_title('DI vs SCI', fontsize=13, fontweight='bold')
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='EI')
        
        # EI vs SCI
        scatter3 = axes[2].scatter(EI, SCI, c=DI, cmap='Reds',
                                  s=40, alpha=0.6, edgecolor='black', linewidth=0.3)
        axes[2].set_xlabel('蒸发指数 (EI)', fontsize=12)
        axes[2].set_ylabel('储量变化指数 (SCI)', fontsize=12)
        axes[2].set_title('EI vs SCI', fontsize=13, fontweight='bold')
        axes[2].axhline(0, color='black', linestyle='--', alpha=0.3)
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=axes[2], label='DI')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    def plot_deviation_histogram(self,
                                deviation: np.ndarray,
                                bins: int = 50,
                                figsize: Tuple[int, int] = (10, 6),
                                save_path: Optional[str] = None):
        """
        绘制Budyko偏离直方图
        
        Parameters:
        -----------
        deviation : array
            偏离值
        bins : int
            直方图bins数
        figsize : tuple
            图大小
        save_path : str, optional
            保存路径
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 直方图
        n, bins_edges, patches = ax.hist(deviation[np.isfinite(deviation)], 
                                        bins=bins, edgecolor='black',
                                        alpha=0.7, color='steelblue')
        
        # 统计线
        mean_dev = np.nanmean(deviation)
        median_dev = np.nanmedian(deviation)
        std_dev = np.nanstd(deviation)
        
        ax.axvline(mean_dev, color='red', linestyle='--', linewidth=2,
                  label=f'均值: {mean_dev:.3f}')
        ax.axvline(median_dev, color='green', linestyle='--', linewidth=2,
                  label=f'中位数: {median_dev:.3f}')
        ax.axvline(mean_dev + std_dev, color='orange', linestyle=':', linewidth=1.5,
                  label=f'±1σ: {std_dev:.3f}')
        ax.axvline(mean_dev - std_dev, color='orange', linestyle=':', linewidth=1.5)
        
        # 设置
        ax.set_xlabel('Budyko偏离 (观测EI - 理论EI)', fontsize=12, fontweight='bold')
        ax.set_ylabel('频数', fontsize=12, fontweight='bold')
        ax.set_title('Budyko偏离分布', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加统计文本框
        textstr = f'统计特征:\nn = {len(deviation[np.isfinite(deviation)])}\n'
        textstr += f'均值 = {mean_dev:.3f}\n中位数 = {median_dev:.3f}\n'
        textstr += f'标准差 = {std_dev:.3f}\n偏度 = {pd.Series(deviation).skew():.3f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_spatial_map(self,
                        lons: np.ndarray,
                        lats: np.ndarray,
                        values: np.ndarray,
                        value_label: str = '值',
                        cmap: str = 'RdBu_r',
                        figsize: Tuple[int, int] = (12, 8),
                        save_path: Optional[str] = None):
        """
        绘制空间分布图
        
        Parameters:
        -----------
        lons, lats : array
            经纬度
        values : array
            要绘制的值
        value_label : str
            值的标签
        cmap : str
            颜色映射
        figsize : tuple
            图大小
        save_path : str, optional
            保存路径
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制
        scatter = ax.scatter(lons, lats, c=values, cmap=cmap,
                           s=100, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # 颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(value_label, fontsize=12, fontweight='bold')
        
        # 设置
        ax.set_xlabel('经度 (°E)', fontsize=12, fontweight='bold')
        ax.set_ylabel('纬度 (°N)', fontsize=12, fontweight='bold')
        ax.set_title(f'{value_label}空间分布', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax


if __name__ == "__main__":
    print("可视化模块加载成功")
    print("可用类: BudykoVisualizer")