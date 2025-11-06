"""
Budyko-ML Runoff Partitioning Project
基于Budyko约束机器学习的径流分割项目

This package contains modules for global runoff partitioning 
using Budyko-constrained machine learning.

该包包含使用Budyko约束机器学习进行全球径流分割的模块。
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# 导入常用函数 / Import common functions
from .utils import (
    budyko_curve,
    bfc_curve,
    lyne_hollick_filter,
    calculate_r2,
    calculate_rmse,
    calculate_nse,
    calculate_kge
)

__all__ = [
    'budyko_curve',
    'bfc_curve',
    'lyne_hollick_filter',
    'calculate_r2',
    'calculate_rmse',
    'calculate_nse',
    'calculate_kge'
]
