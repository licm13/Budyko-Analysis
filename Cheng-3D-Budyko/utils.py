"""
辅助函数模块 (Utility Functions Module)
包含项目中使用的通用函数
Contains common functions used throughout the project

Author: [Your Name]
Date: 2025-01-01
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Union, Tuple, Optional
from pathlib import Path


# =============================================================================
# 水文计算函数 (Hydrological Calculation Functions)
# =============================================================================

def budyko_curve(P: Union[float, np.ndarray], 
                 Ep: Union[float, np.ndarray], 
                 alpha: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    计算Budyko曲线的径流系数 (Q/P)
    Calculate runoff coefficient (Q/P) using Budyko curve
    
    基于Fu-Zhang公式 (方程3)
    Based on Fu-Zhang equation (Equation 3)
    
    Parameters
    ----------
    P : float or array-like
        降水 (mm/yr) / Precipitation (mm/yr)
    Ep : float or array-like
        潜在蒸散发 (mm/yr) / Potential evapotranspiration (mm/yr)
    alpha : float or array-like
        Budyko参数 / Budyko parameter
        
    Returns
    -------
    Q_over_P : float or array-like
        径流系数 (无量纲) / Runoff coefficient (dimensionless)
        
    Notes
    -----
    方程 / Equation:
    Q/P = -Ep/P + [1 + (Ep/P)^α]^(1/α)
    
    Examples
    --------
    >>> P, Ep, alpha = 1000, 800, 2.5
    >>> RFC = budyko_curve(P, Ep, alpha)
    >>> print(f"Runoff coefficient: {RFC:.3f}")
    """
    # 计算干旱指数 / Calculate aridity index
    aridity_index = Ep / P
    
    # 应用Budyko公式 / Apply Budyko formula
    term = (1 + aridity_index**alpha)**(1/alpha)
    Q_over_P = -aridity_index + term
    
    # 确保物理约束: 0 <= Q/P <= 1
    # Ensure physical constraints: 0 <= Q/P <= 1
    Q_over_P = np.clip(Q_over_P, 0, 1)
    
    return Q_over_P


def bfc_curve(P: Union[float, np.ndarray], 
              Ep: Union[float, np.ndarray], 
              alpha: Union[float, np.ndarray],
              Qbp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    计算BFC曲线的基流系数 (Qb/P)
    Calculate baseflow coefficient (Qb/P) using BFC curve
    
    基于Cheng et al. (2021)的BFC公式 (方程4)
    Based on Cheng et al. (2021) BFC equation (Equation 4)
    
    Parameters
    ----------
    P : float or array-like
        降水 (mm/yr) / Precipitation (mm/yr)
    Ep : float or array-like
        潜在蒸散发 (mm/yr) / Potential evapotranspiration (mm/yr)
    alpha : float or array-like
        Budyko参数 / Budyko parameter
    Qbp : float or array-like
        潜在基流 (mm/yr) / Potential baseflow (mm/yr)
        
    Returns
    -------
    Qb_over_P : float or array-like
        基流系数 (无量纲) / Baseflow coefficient (dimensionless)
        
    Notes
    -----
    方程 / Equation:
    Qb/P = Qbp/P + [1+(Ep/P)^α]^(1/α) - [1+(Ep/P+Qbp/P)^α]^(1/α)
    
    Examples
    --------
    >>> P, Ep, alpha, Qbp = 1000, 800, 2.5, 300
    >>> BFC = bfc_curve(P, Ep, alpha, Qbp)
    >>> print(f"Baseflow coefficient: {BFC:.3f}")
    """
    # 计算各项比值 / Calculate ratios
    aridity_index = Ep / P
    Qbp_over_P = Qbp / P
    
    # 应用BFC公式 / Apply BFC formula
    term1 = (1 + aridity_index**alpha)**(1/alpha)
    term2 = (1 + (aridity_index + Qbp_over_P)**alpha)**(1/alpha)
    Qb_over_P = Qbp_over_P + term1 - term2
    
    # 确保物理约束: 0 <= Qb/P <= Q/P <= 1
    # Ensure physical constraints: 0 <= Qb/P <= Q/P <= 1
    Q_over_P = budyko_curve(P, Ep, alpha)
    Qb_over_P = np.clip(Qb_over_P, 0, Q_over_P)
    
    return Qb_over_P


def estimate_alpha_from_obs(P: np.ndarray, 
                           Ep: np.ndarray, 
                           Q: np.ndarray,
                           tol: float = 1e-6,
                           max_iter: int = 100) -> np.ndarray:
    """
    从观测数据估计Budyko参数α
    Estimate Budyko parameter α from observed data
    
    使用牛顿迭代法求解方程3
    Uses Newton's method to solve Equation 3
    
    Parameters
    ----------
    P : array-like
        观测降水 (mm/yr) / Observed precipitation (mm/yr)
    Ep : array-like
        观测潜在蒸散发 (mm/yr) / Observed potential ET (mm/yr)
    Q : array-like
        观测径流 (mm/yr) / Observed runoff (mm/yr)
    tol : float, optional
        收敛容差 / Convergence tolerance (default: 1e-6)
    max_iter : int, optional
        最大迭代次数 / Maximum iterations (default: 100)
        
    Returns
    -------
    alpha : array-like
        估计的α参数 / Estimated α parameter
        
    Notes
    -----
    求解目标方程 / Target equation to solve:
    Q/P = -Ep/P + [1 + (Ep/P)^α]^(1/α)
    
    使用牛顿法迭代 / Newton's method iteration:
    α_{n+1} = α_n - f(α_n)/f'(α_n)
    """
    # 计算观测的径流系数 / Calculate observed runoff coefficient
    RFC_obs = Q / P
    aridity_index = Ep / P
    
    # 初始化α (使用经验值2.0)
    # Initialize α (using empirical value 2.0)
    alpha = np.full_like(RFC_obs, 2.0)
    
    # 牛顿迭代 / Newton's method iteration
    for iteration in range(max_iter):
        # 计算当前α下的RFC / Calculate RFC with current α
        RFC_calc = budyko_curve(P, Ep, alpha)
        
        # 计算残差 / Calculate residual
        residual = RFC_calc - RFC_obs
        
        # 检查收敛 / Check convergence
        if np.all(np.abs(residual) < tol):
            break
        
        # 计算数值导数 / Calculate numerical derivative
        delta_alpha = 0.001
        RFC_plus = budyko_curve(P, Ep, alpha + delta_alpha)
        derivative = (RFC_plus - RFC_calc) / delta_alpha
        
        # 更新α (避免除零)
        # Update α (avoid division by zero)
        mask = np.abs(derivative) > 1e-10
        alpha[mask] -= residual[mask] / derivative[mask]
        
        # 确保α在合理范围内 [1, 10]
        # Ensure α is within reasonable range [1, 10]
        alpha = np.clip(alpha, 1.0, 10.0)
    
    return alpha


def estimate_qbp_from_obs(P: np.ndarray, 
                         Ep: np.ndarray, 
                         Qb: np.ndarray,
                         alpha: np.ndarray,
                         tol: float = 1e-6,
                         max_iter: int = 100) -> np.ndarray:
    """
    从观测数据估计潜在基流Qb,p
    Estimate potential baseflow Qb,p from observed data
    
    使用牛顿迭代法求解方程4
    Uses Newton's method to solve Equation 4
    
    Parameters
    ----------
    P : array-like
        观测降水 (mm/yr) / Observed precipitation (mm/yr)
    Ep : array-like
        观测潜在蒸散发 (mm/yr) / Observed potential ET (mm/yr)
    Qb : array-like
        观测基流 (mm/yr) / Observed baseflow (mm/yr)
    alpha : array-like
        已估计的α参数 / Estimated α parameter
    tol : float, optional
        收敛容差 / Convergence tolerance (default: 1e-6)
    max_iter : int, optional
        最大迭代次数 / Maximum iterations (default: 100)
        
    Returns
    -------
    Qbp : array-like
        估计的Qb,p参数 (mm/yr) / Estimated Qb,p parameter (mm/yr)
        
    Notes
    -----
    求解目标方程 / Target equation to solve:
    Qb/P = Qbp/P + [1+(Ep/P)^α]^(1/α) - [1+(Ep/P+Qbp/P)^α]^(1/α)
    """
    # 计算观测的基流系数 / Calculate observed baseflow coefficient
    BFC_obs = Qb / P
    
    # 初始化Qb,p (使用Qb作为初值)
    # Initialize Qb,p (using Qb as initial value)
    Qbp = Qb.copy()
    
    # 牛顿迭代 / Newton's method iteration
    for iteration in range(max_iter):
        # 计算当前Qb,p下的BFC / Calculate BFC with current Qb,p
        BFC_calc = bfc_curve(P, Ep, alpha, Qbp)
        
        # 计算残差 / Calculate residual
        residual = BFC_calc - BFC_obs
        
        # 检查收敛 / Check convergence
        if np.all(np.abs(residual) < tol):
            break
        
        # 计算数值导数 / Calculate numerical derivative
        delta_Qbp = 1.0  # mm/yr
        BFC_plus = bfc_curve(P, Ep, alpha, Qbp + delta_Qbp)
        derivative = (BFC_plus - BFC_calc) / delta_Qbp
        
        # 更新Qb,p (避免除零)
        # Update Qb,p (avoid division by zero)
        mask = np.abs(derivative) > 1e-10
        Qbp[mask] -= residual[mask] / derivative[mask]
        
        # 确保Qb,p > 0
        # Ensure Qb,p > 0
        Qbp = np.maximum(Qbp, 1.0)
    
    return Qbp


# =============================================================================
# 基流分离函数 (Baseflow Separation Functions)
# =============================================================================

def lyne_hollick_filter(Q: np.ndarray, 
                       alpha: float = 0.925,
                       n_passes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用Lyne-Hollick数字滤波法分离基流
    Separate baseflow using Lyne-Hollick digital filter method
    
    Parameters
    ----------
    Q : array-like
        日径流序列 (mm/day) / Daily runoff series (mm/day)
    alpha : float, optional
        滤波参数 (默认: 0.925) / Filter parameter (default: 0.925)
        通常取值范围 0.90-0.95 / Typical range 0.90-0.95
    n_passes : int, optional
        滤波次数 (默认: 3) / Number of passes (default: 3)
        推荐3次 (前向-后向-前向) / Recommended 3 passes (forward-backward-forward)
        
    Returns
    -------
    Qb : array-like
        基流序列 (mm/day) / Baseflow series (mm/day)
    Qq : array-like
        快流序列 (mm/day) / Quickflow series (mm/day)
        
    Notes
    -----
    递归滤波方程 / Recursive filter equation:
    q_t = alpha * q_{t-1} + (1+alpha)/2 * (Q_t - Q_{t-1})
    
    其中 q_t 是快流,基流计算为 Qb_t = Q_t - q_t
    Where q_t is quickflow, baseflow is calculated as Qb_t = Q_t - q_t
    
    References
    ----------
    Lyne, V., & Hollick, M. (1979). Stochastic time-variable rainfall-runoff modelling.
    
    Examples
    --------
    >>> Q_daily = np.random.lognormal(2, 1, 365)  # 模拟日径流
    >>> Qb, Qq = lyne_hollick_filter(Q_daily)
    >>> print(f"Mean baseflow: {Qb.mean():.2f}, Mean quickflow: {Qq.mean():.2f}")
    """
    # 确保输入是numpy数组 / Ensure input is numpy array
    Q = np.asarray(Q)
    n = len(Q)
    
    # 初始化快流序列 / Initialize quickflow series
    q = Q.copy()
    
    # 执行多次滤波 / Perform multiple passes
    for pass_num in range(n_passes):
        if pass_num % 2 == 0:  # 前向滤波 / Forward pass
            for t in range(1, n):
                q[t] = alpha * q[t-1] + (1 + alpha) / 2 * (Q[t] - Q[t-1])
                # 确保约束: 0 <= q_t <= Q_t
                # Ensure constraint: 0 <= q_t <= Q_t
                q[t] = np.clip(q[t], 0, Q[t])
        else:  # 后向滤波 / Backward pass
            for t in range(n-2, -1, -1):
                q[t] = alpha * q[t+1] + (1 + alpha) / 2 * (Q[t] - Q[t+1])
                # 确保约束 / Ensure constraint
                q[t] = np.clip(q[t], 0, Q[t])
    
    # 计算基流 / Calculate baseflow
    Qb = Q - q
    Qq = q
    
    return Qb, Qq


# =============================================================================
# 评估指标函数 (Evaluation Metric Functions)
# =============================================================================

def calculate_r2(obs: np.ndarray, sim: np.ndarray) -> float:
    """
    计算决定系数 R²
    Calculate coefficient of determination R²
    
    Parameters
    ----------
    obs : array-like
        观测值 / Observed values
    sim : array-like
        模拟值 / Simulated values
        
    Returns
    -------
    r2 : float
        R²值 (0-1) / R² value (0-1)
        
    Notes
    -----
    R² = 1 - SS_res / SS_tot
    其中 SS_res = Σ(obs - sim)²
         SS_tot = Σ(obs - mean(obs))²
    """
    # 移除缺失值 / Remove missing values
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]
    
    if len(obs) == 0:
        return np.nan
    
    # 计算残差平方和 / Calculate residual sum of squares
    ss_res = np.sum((obs - sim)**2)
    
    # 计算总平方和 / Calculate total sum of squares
    ss_tot = np.sum((obs - np.mean(obs))**2)
    
    # 计算R² / Calculate R²
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return r2


def calculate_rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    """
    计算均方根误差 RMSE
    Calculate root mean square error RMSE
    
    Parameters
    ----------
    obs : array-like
        观测值 / Observed values
    sim : array-like
        模拟值 / Simulated values
        
    Returns
    -------
    rmse : float
        RMSE值 / RMSE value
        
    Notes
    -----
    RMSE = √[Σ(obs - sim)² / n]
    """
    # 移除缺失值 / Remove missing values
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]
    
    if len(obs) == 0:
        return np.nan
    
    # 计算RMSE / Calculate RMSE
    rmse = np.sqrt(np.mean((obs - sim)**2))
    
    return rmse


def calculate_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """
    计算Nash-Sutcliffe效率系数 NSE
    Calculate Nash-Sutcliffe efficiency NSE
    
    Parameters
    ----------
    obs : array-like
        观测值 / Observed values
    sim : array-like
        模拟值 / Simulated values
        
    Returns
    -------
    nse : float
        NSE值 (-∞, 1] / NSE value (-∞, 1]
        
    Notes
    -----
    NSE = 1 - Σ(obs - sim)² / Σ(obs - mean(obs))²
    """
    # 移除缺失值 / Remove missing values
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]
    
    if len(obs) == 0:
        return np.nan
    
    # 计算NSE / Calculate NSE
    numerator = np.sum((obs - sim)**2)
    denominator = np.sum((obs - np.mean(obs))**2)
    
    nse = 1 - (numerator / denominator) if denominator > 0 else -np.inf
    
    return nse


def calculate_kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """
    计算Kling-Gupta效率系数 KGE
    Calculate Kling-Gupta efficiency KGE
    
    Parameters
    ----------
    obs : array-like
        观测值 / Observed values
    sim : array-like
        模拟值 / Simulated values
        
    Returns
    -------
    kge : float
        KGE值 (-∞, 1] / KGE value (-∞, 1]
        
    Notes
    -----
    KGE = 1 - √[(r-1)² + (α-1)² + (β-1)²]
    其中 / Where:
    r = 相关系数 / correlation coefficient
    α = σ_sim / σ_obs (变异性比 / variability ratio)
    β = μ_sim / μ_obs (偏差比 / bias ratio)
    """
    # 移除缺失值 / Remove missing values
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]
    
    if len(obs) == 0:
        return np.nan
    
    # 计算相关系数 / Calculate correlation
    r = np.corrcoef(obs, sim)[0, 1]
    
    # 计算变异性比 / Calculate variability ratio
    alpha = np.std(sim) / np.std(obs) if np.std(obs) > 0 else 0
    
    # 计算偏差比 / Calculate bias ratio
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else 0
    
    # 计算KGE / Calculate KGE
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return kge


# =============================================================================
# 数据处理函数 (Data Processing Functions)
# =============================================================================

def resample_to_grid(data: xr.DataArray, 
                    target_resolution: float = 0.25,
                    method: str = 'bilinear') -> xr.DataArray:
    """
    将数据重采样到目标分辨率网格
    Resample data to target resolution grid
    
    Parameters
    ----------
    data : xarray.DataArray
        输入数据 / Input data
    target_resolution : float, optional
        目标分辨率 (度) / Target resolution (degrees) (default: 0.25)
    method : str, optional
        插值方法 / Interpolation method (default: 'bilinear')
        可选 / Options: 'bilinear', 'nearest', 'cubic'
        
    Returns
    -------
    resampled_data : xarray.DataArray
        重采样后的数据 / Resampled data
        
    Examples
    --------
    >>> import xarray as xr
    >>> data = xr.DataArray(np.random.rand(180, 360), 
    ...                     dims=['lat', 'lon'],
    ...                     coords={'lat': np.linspace(-90, 90, 180),
    ...                            'lon': np.linspace(-180, 180, 360)})
    >>> resampled = resample_to_grid(data, target_resolution=0.5)
    """
    # 创建目标网格 / Create target grid
    target_lat = np.arange(-90 + target_resolution/2, 
                          90, 
                          target_resolution)
    target_lon = np.arange(-180 + target_resolution/2, 
                          180, 
                          target_resolution)
    
    # 执行插值 / Perform interpolation
    resampled_data = data.interp(
        lat=target_lat,
        lon=target_lon,
        method=method
    )
    
    return resampled_data


def load_catchment_data(data_dir: Union[str, Path],
                       catchment_id: str) -> pd.DataFrame:
    """
    加载单个流域的观测数据
    Load observational data for a single catchment
    
    Parameters
    ----------
    data_dir : str or Path
        数据目录路径 / Data directory path
    catchment_id : str
        流域ID / Catchment ID
        
    Returns
    -------
    df : pandas.DataFrame
        包含日期、流量等的数据框 / DataFrame with date, discharge, etc.
        
    Examples
    --------
    >>> df = load_catchment_data('./data/raw/discharge/', '1234567')
    >>> print(df.head())
    """
    data_dir = Path(data_dir)
    file_path = data_dir / f"{catchment_id}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"找不到流域数据文件 / Catchment data file not found: {file_path}")
    
    # 读取数据 / Read data
    df = pd.read_csv(file_path, parse_dates=['date'])
    
    return df


# =============================================================================
# 可视化辅助函数 (Visualization Helper Functions)
# =============================================================================

def setup_map_projection():
    """
    设置地图投影 (用于全球地图绘制)
    Setup map projection (for global mapping)
    
    Returns
    -------
    projection : cartopy.crs projection object
        地图投影对象 / Map projection object
    """
    try:
        import cartopy.crs as ccrs
        projection = ccrs.PlateCarree()
        return projection
    except ImportError:
        print("警告: cartopy未安装,无法绘制地图 / Warning: cartopy not installed, cannot create maps")
        return None


if __name__ == "__main__":
    # 测试代码 / Test code
    print("Testing utility functions...")
    print("测试辅助函数...")
    
    # 测试Budyko曲线 / Test Budyko curve
    P, Ep, alpha = 1000, 800, 2.5
    RFC = budyko_curve(P, Ep, alpha)
    print(f"\n径流系数 / Runoff coefficient: {RFC:.3f}")
    
    # 测试BFC曲线 / Test BFC curve
    Qbp = 300
    BFC = bfc_curve(P, Ep, alpha, Qbp)
    print(f"基流系数 / Baseflow coefficient: {BFC:.3f}")
    
    # 测试基流分离 / Test baseflow separation
    Q_daily = np.random.lognormal(2, 1, 365)
    Qb, Qq = lyne_hollick_filter(Q_daily)
    print(f"\n平均基流 / Mean baseflow: {Qb.mean():.2f} mm/day")
    print(f"平均快流 / Mean quickflow: {Qq.mean():.2f} mm/day")
    print(f"基流指数 / Baseflow index: {Qb.sum()/Q_daily.sum():.3f}")
    
    print("\n所有测试完成! / All tests completed!")
